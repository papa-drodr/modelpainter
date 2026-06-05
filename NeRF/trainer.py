import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NeRF.model import NeRF
from NeRF.renderer import render_rays


def compute_near_far(
    data_dir: str,
    near_factor: float = 0.1,
    far_factor: float = 1.2,
    bound_factor: float = 1.5,
) -> tuple[float, float, float]:
    """
    Compute near, far, and scene bound from camera positions in transforms.json.

    Args:
        data_dir: directory containing transforms.json
        near_factor: near = min_camera_dist * near_factor
        far_factor: far = max_camera_dist * far_factor
        bound_factor: bound = min_camera_dist * bound_factor

    Returns:
        near: near plane distance
        far: far plane distance
        bound: scene bound for density grid
    """
    transforms_path = Path(data_dir) / "transforms.json"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    distances = []
    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"])
        cam_pos = c2w[:3, 3]
        dist = float(np.linalg.norm(cam_pos))
        distances.append(dist)

    min_dist = float(np.min(distances))
    max_dist = float(np.max(distances))

    near = max(min_dist * near_factor, 0.01)
    far = max_dist * far_factor
    bound = min_dist * bound_factor

    print(
        f"Auto near={near:.4f}, far={far:.4f}, bound={bound:.4f} "
        f"(camera dist range: {min_dist:.4f} ~ {max_dist:.4f})"
    )

    return near, far, bound


def compute_val_psnr(
    model: NeRF,
    val_loader: DataLoader,
    near: float,
    far: float,
    num_samples: int,
    num_fine: int,
    autocast_device: str,
    device: str,
) -> float:
    model.eval()
    total_mse = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            rays_o = batch["rays_o"].to(device)
            rays_d = batch["rays_d"].to(device)
            rgb_gt = batch["rgb"].to(device)
            with torch.amp.autocast(autocast_device, enabled=(device == "cuda")):
                result = render_rays(
                    model, rays_o, rays_d,
                    near=near, far=far,
                    num_samples=num_samples, num_fine=num_fine,
                    perturb=False,
                )
            mse = nn.functional.mse_loss(result["rgb_map"], rgb_gt).item()
            total_mse += mse * len(rays_o)
            total_acc += result["weights"].sum(dim=-1).mean().item() * len(rays_o)
            count += len(rays_o)
    model.train()
    avg_mse = total_mse / max(count, 1)
    avg_acc = total_acc / max(count, 1)
    psnr = float(-10.0 * np.log10(avg_mse + 1e-8))
    print(f"  [val diag] acc={avg_acc:.4f} (0=white bg, 1=fully opaque)", end="")
    return psnr


def train(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-4,
    num_samples: int = 64,
    num_fine: int = 64,
    near: float | None = None,
    far: float | None = None,
    save_every: int = 5,
    img_size: tuple[int, int] = (400, 225),
    resume_from: str | None = None,
    lambda_sparse: float = 1e-4,
    rays_per_epoch: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {device}")

    if near is None or far is None:
        near, far, bound = compute_near_far(data_dir)
    else:
        _, _, bound = compute_near_far(data_dir)

    from NeRF.dataset import NeRFDataset

    print(f"Image size: {img_size[0]}x{img_size[1]}")
    train_dataset = NeRFDataset(data_dir=data_dir, split="train", img_size=img_size)
    val_dataset = NeRFDataset(data_dir=data_dir, split="val", img_size=img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # LR warmup over first 10% of epochs, then cosine annealing
    warmup_epochs = max(1, num_epochs // 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=lr * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    autocast_device = "cuda" if device == "cuda" else "cpu"

    max_batches = (rays_per_epoch // batch_size) if rays_per_epoch else None
    batches_per_epoch = (
        min(len(train_loader), max_batches) if max_batches else len(train_loader)
    )

    print(f"Num params: {sum(p.numel() for p in model.parameters())}")
    print(f"Train rays: {len(train_dataset)} | Val rays: {len(val_dataset)}")
    if rays_per_epoch:
        print(f"Rays per epoch: {rays_per_epoch:,} ({batches_per_epoch} batches)")
    else:
        print(f"Train batches per epoch: {len(train_loader)}")
    print(
        f"Hierarchical sampling: coarse={num_samples}, fine={num_fine} | "
        f"LR warmup: {warmup_epochs} epochs\n"
    )

    start_epoch = 0
    if resume_from:
        try:
            ckpt = torch.load(resume_from, map_location=device)
        except (FileNotFoundError, RuntimeError) as e:
            raise RuntimeError(f"체크포인트 로딩 실패: {e}")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from checkpoint: epoch {start_epoch}\n")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        # coarse loss weight decays from 1.0 → 0.1 over training
        coarse_weight = max(0.1, 1.0 - (epoch - 1) / num_epochs)

        train_mse = 0.0
        for step, batch in enumerate(train_loader):
            if max_batches and step >= max_batches:
                break
            rays_o = batch["rays_o"].to(device)
            rays_d = batch["rays_d"].to(device)
            rgb_gt = batch["rgb"].to(device)

            with torch.amp.autocast(autocast_device, enabled=(device == "cuda")):
                result = render_rays(
                    model, rays_o, rays_d,
                    near=near, far=far,
                    num_samples=num_samples, num_fine=num_fine,
                    perturb=True,
                )
                fine_mse = nn.functional.mse_loss(result["rgb_map"], rgb_gt)
                loss = fine_mse
                if "rgb_map_coarse" in result:
                    coarse_mse = nn.functional.mse_loss(
                        result["rgb_map_coarse"], rgb_gt
                    )
                    loss = loss + coarse_weight * coarse_mse
                if lambda_sparse > 0:
                    loss = loss + lambda_sparse * result["weights"].mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mse += fine_mse.item()

        scheduler.step()
        train_loss /= batches_per_epoch
        train_mse /= batches_per_epoch
        train_psnr = -10.0 * np.log10(train_mse + 1e-8)
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train loss: {train_loss:.4f} ({train_psnr:.2f} dB)",
            end="",
        )

        if epoch % save_every == 0:
            val_psnr = compute_val_psnr(
                model, val_loader, near, far,
                num_samples, num_fine, autocast_device, device,
            )
            print(f" | val PSNR: {val_psnr:.2f} dB")

            ckpt_path = save_path / f"nerf_epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "near": near,
                    "far": far,
                    "bound": bound,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved -> {ckpt_path}")
        else:
            print()

    final_path = save_path / "nerf_final.pth"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "near": near,
            "far": far,
            "bound": bound,
        },
        final_path,
    )
    print(f"\nFinal model saved -> {final_path}")

    return model, bound


def load_model(ckpt_path: str, device: str = "cuda") -> tuple[NeRF, dict]:
    """
    Load NeRF model from checkpoint.

    Args:
        ckpt_path: path to checkpoint file
        device: device to load model on

    Returns:
        model: loaded NeRF model
        ckpt: checkpoint dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model = NeRF().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded from {ckpt_path} (epoch {ckpt['epoch']})")

    return model, ckpt


if __name__ == "__main__":
    # for test
    train(
        data_dir="./data",
        save_dir="./checkpoints",
        num_epochs=20,
        batch_size=1024,
    )
