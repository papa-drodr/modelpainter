import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from NeRF.model import NeRF
from NeRF.renderer import render_rays


def train(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-4,
    num_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    save_every: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train NeRF model.

    Args:
        data_dir: dir containing transforms.json and frames/
        save_dir: dir to save model checkpoints
        num_epochs: number of training epochs
        batch_size: number of rays per batch
        lr: learning rate
        num_samples: number of samples per ray
        near: near plane distance
        far: far plane distance
        save_every: save checkpoint every N epochs
        device: training device
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {device}")

    # dataset
    from NeRF.dataset import NeRFDataset

    train_dataset = NeRFDataset(data_dir=data_dir, split="train")
    val_dataset = NeRFDataset(data_dir=data_dir, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # model
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / num_epochs)
    )

    print(f"Num params: {sum(p.numel() for p in model.parameters())}")
    print(f"Train rays: {len(train_dataset)} / Val rays: {len(val_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}\n")

    for epoch in range(1, num_epochs + 1):
        # train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            rays_o = batch["rays_o"].to(device)
            rays_d = batch["rays_d"].to(device)
            rgb_gt = batch["rgb"].to(device)

            result = render_rays(
                model,
                rays_o,
                rays_d,
                near=near,
                far=far,
                num_samples=num_samples,
                perturb=True,
            )

            # MSE loss between predicted and ground truth RGB
            loss = nn.functional.mse_loss(result["rgb_map"], rgb_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                rays_o = batch["rays_o"].to(device)
                rays_d = batch["rays_d"].to(device)
                rgb_gt = batch["rgb"].to(device)

                result = render_rays(
                    model,
                    rays_o,
                    rays_d,
                    near=near,
                    far=far,
                    num_samples=num_samples,
                    perturb=False,
                )

                loss = nn.functional.mse_loss(result["rgb_map"], rgb_gt)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # PSNR: higher is better (dB)
        train_psnr = -10.0 * torch.log10(torch.tensor(train_loss))
        val_psnr = -10.0 * torch.log10(torch.tensor(val_loss))

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train loss: {train_loss:.4f} ({train_psnr:.2f} dB) | "
            f"val loss: {val_loss:.4f} ({val_psnr:.2f} dB)"
        )

        # save checkpoint
        if epoch % save_every == 0:
            ckpt_path = save_path / f"nerf_epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved -> {ckpt_path}")

    # save final model
    final_path = save_path / "nerf_final.pth"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        final_path,
    )
    print(f"\nFinal model saved -> {final_path}")

    return model


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
