import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NeRF.model import NeRF
from NeRF.renderer import render_rays


def compute_near_far(data_dir: str) -> tuple[float, float, float]:
    """
    Compute near, far, and scene bound from camera positions in transforms.json.

    Args:
        data_dir: directory containing transforms.json

    Returns:
        near: near plane distance
        far: far plane distance
        bound: scene bound for density grid (max camera distance from origin)
    """
    transforms_path = Path(data_dir) / "transforms.json"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    distances = []
    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"])
        # camera position is the last column of c2w
        cam_pos = c2w[:3, 3]
        dist = float(np.linalg.norm(cam_pos))
        distances.append(dist)

    min_dist = float(np.min(distances))
    max_dist = float(np.max(distances))

    near = max(min_dist * 0.1, 0.01)  # avoid near = 0
    far = max_dist * 1.2
    bound = min_dist * 1.5  # object radius ≈ inner camera radius

    print(
        f"Auto near={near:.4f}, far={far:.4f}, bound={bound:.4f} "
        f"(camera dist range: {min_dist:.4f} ~ {max_dist:.4f})"
    )

    return near, far, bound


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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train NeRF model.

    Args:
        data_dir: directory containing transforms.json and frames/
        save_dir: directory to save model checkpoints
        num_epochs: number of training epochs
        batch_size: number of rays per batch
        lr: learning rate
        num_samples: number of samples per ray
        near: near plane distance (auto-computed from transforms.json if None)
        far: far plane distance (auto-computed from transforms.json if None)
        save_every: save checkpoint every N epochs
        img_size: (width, height) to resize training images
        device: training device
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {device}")

    # auto-compute near/far/bound from scene
    if near is None or far is None:
        near, far, bound = compute_near_far(data_dir)
    else:
        _, _, bound = compute_near_far(data_dir)

    # dataset
    from NeRF.dataset import NeRFDataset

    print(f"Image size: {img_size[0]}x{img_size[1]}")
    train_dataset = NeRFDataset(data_dir=data_dir, split="train", img_size=img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # model
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    print(f"Num params: {sum(p.numel() for p in model.parameters())}")
    print(f"Train rays: {len(train_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Hierarchical sampling: coarse={num_samples}, fine={num_fine}\n")

    # resume from checkpoint
    start_epoch = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from checkpoint: epoch {start_epoch}\n")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            rays_o = batch["rays_o"].to(device)
            rays_d = batch["rays_d"].to(device)
            rgb_gt = batch["rgb"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                result = render_rays(
                    model, rays_o, rays_d,
                    near=near, far=far,
                    num_samples=num_samples, num_fine=num_fine,
                    perturb=True,
                )
                loss = nn.functional.mse_loss(result["rgb_map"], rgb_gt)
                if "rgb_map_coarse" in result:
                    loss = loss + nn.functional.mse_loss(result["rgb_map_coarse"], rgb_gt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        train_psnr = -10.0 * torch.log10(torch.tensor(train_loss))
        print(f"Epoch [{epoch}/{num_epochs}] train loss: {train_loss:.4f} ({train_psnr:.2f} dB)")

        if epoch % save_every == 0:
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
