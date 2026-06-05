import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_rays(
    H: int, W: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for each pixel in the image.

    Args:
        H: image height
        W: image width
        focal: focal length
        c2w: camera-to-world matrix (4x4)

    Returns:
        rays_o: ray origins (H, W, 3)
        rays_d: ray directions (H, W, 3)
    """
    # pixel grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )

    # ray direction in camera coordinate
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,  # flip y-axis (OpenGL convention)
            -torch.ones_like(i),  # z points forward
        ],
        dim=-1,
    )  # (H, W, 3)

    # transform to world coordinate using c2w
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)  # (H, W, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # normalize to unit vectors
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d


class NeRFDataset(Dataset):
    """
    NeRF dataset loading transforms.json and corresponding images.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: tuple[int, int] = (800, 450),
    ):
        """
        Args:
            data_dir: directory containing transforms.json and frames/
            split: "train" or "val"
            img_size: (width, height) to resize images
        """
        self.data_dir = Path(data_dir)
        self.img_W, self.img_H = img_size

        # load transforms.json
        transforms_path = self.data_dir / "transforms.json"
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        # compute focal length from camera angle
        camera_angle_x = transforms["camera_angle_x"]
        self.focal = 0.5 * self.img_W / np.tan(0.5 * camera_angle_x)

        frames = transforms["frames"]

        # random train / val split (9:1), fixed seed for reproducibility
        indices = list(range(len(frames)))
        random.seed(42)
        random.shuffle(indices)
        split_idx = int(len(indices) * 0.9)
        if split == "train":
            frames = [frames[i] for i in indices[:split_idx]]
        else:
            frames = [frames[i] for i in indices[split_idx:]]

        print(f"[{split}] {len(frames)} frames loaded.")

        self.rays_o = []
        self.rays_d = []
        self.rgbs = []

        for frame in frames:
            img_path = self.data_dir / (frame["file_path"] + ".jpg")
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

            # load and resize image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.img_W, self.img_H), Image.BILINEAR)
            img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0  # (H, W, 3)

            # generate rays for each pixel
            rays_o, rays_d = get_rays(self.img_H, self.img_W, self.focal, c2w)

            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d)
            self.rgbs.append(img)

        # flatten (N_frames, H, W, 3) -> (N_frames * H * W, 3)
        self.rays_o = torch.stack(self.rays_o).reshape(-1, 3)
        self.rays_d = torch.stack(self.rays_d).reshape(-1, 3)
        self.rgbs = torch.stack(self.rgbs).reshape(-1, 3)

        print(f"Total rays: {len(self.rays_o)}")

    def __len__(self) -> int:
        return len(self.rays_o)

    def __getitem__(self, idx: int) -> dict:
        return {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "rgb": self.rgbs[idx],
        }


if __name__ == "__main__":
    # for test
    dataset = NeRFDataset(data_dir="./data", split="train")
    sample = dataset[0]
    print("rays_o:", sample["rays_o"])
    print("rays_d:", sample["rays_d"])
    print("rgb:", sample["rgb"])
