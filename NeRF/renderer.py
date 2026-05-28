import torch
import torch.nn.functional as F
from NeRF.model import NeRF


def sample_points_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3D points along each ray between near and far bounds.

    Args:
        rays_o: ray origins (N, 3)
        rays_d: ray directions (N, 3)
        near: near plane distance
        far: far plane distance
        num_samples: number of points sampled per ray
        perturb: add random noise to sample positions during training

    Returns:
        pts: sampled 3D points (N, num_samples, 3)
        t_vals: sample distances along ray (N, num_samples)
    """
    N = rays_o.shape[0]

    # evenly spaced t values between near and far
    t_vals = torch.linspace(near, far, num_samples, device=rays_o.device)
    t_vals = t_vals.expand(N, num_samples)  # (N, num_samples)

    if perturb:
        # add uniform noise within each interval
        mid = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        upper = torch.cat([mid, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mid], dim=-1)
        noise = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * noise

    # pts = ray_origin + t * ray_direction
    pts = (
        rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
    )  # (N, num_samples, 3)

    return pts, t_vals


def volume_rendering(
    rgb: torch.Tensor,
    density: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform volume rendering to composite colors along each ray.

    Args:
        rgb: predicted colors at each sample (N, num_samples, 3)
        density: predicted density at each sample (N, num_samples, 1)
        t_vals: sample distances along ray (N, num_samples)
        rays_d: ray directions (N, 3) for computing real distances

    Returns:
        rgb_map: rendered color per ray (N, 3)
        depth_map: rendered depth per ray (N,)
        weights: weight per sample (N, num_samples)
    """
    # compute distance between adjacent samples
    # last interval is set to 1e10 (infinity)
    dists = t_vals[:, 1:] - t_vals[:, :-1]  # (N, num_samples - 1)
    dists = torch.cat(
        [dists, torch.full_like(dists[:, :1], 1e10)], dim=-1
    )  # (N, num_samples)

    # scale by ray direction norm to get real world distances
    dists = dists * rays_d.norm(dim=-1, keepdim=True)  # (N, num_samples)

    # alpha = 1 - exp(-sigma * delta)
    sigma = F.relu(density[..., 0])  # (N, num_samples), clamp negative density
    alpha = 1.0 - torch.exp(-sigma * dists)  # (N, num_samples)

    # transmittance T_i = prod(1 - alpha_j) for j < i
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[
        :, :-1
    ]  # (N, num_samples)

    # weight = T_i * alpha_i
    weights = transmittance * alpha  # (N, num_samples)

    # composite color: sum of weighted RGB
    rgb_map = (weights[..., None] * rgb).sum(dim=1)  # (N, 3)

    # depth map: expected distance along ray
    depth_map = (weights * t_vals).sum(dim=-1)  # (N,)

    return rgb_map, depth_map, weights


def render_rays(
    model: NeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    num_samples: int = 64,
    perturb: bool = True,
) -> dict:
    """
    Full rendering pipeline for a batch of rays.

    Args:
        model: NeRF model
        rays_o: ray origins (N, 3)
        rays_d: ray directions (N, 3)
        near: near plane distance
        far: far plane distance
        num_samples: number of samples per ray
        perturb: add perturbation to samples during training

    Returns:
        dict with rgb_map, depth_map, weights
    """
    # sample points along rays
    pts, t_vals = sample_points_along_rays(
        rays_o, rays_d, near, far, num_samples, perturb
    )

    N, S, _ = pts.shape  # N rays, S samples

    # flatten for model input
    pts_flat = pts.reshape(-1, 3)  # (N*S, 3)
    dirs_flat = rays_d[:, None, :].expand_as(pts).reshape(-1, 3)  # (N*S, 3)

    # NeRF forward
    rgb_flat, density_flat = model(pts_flat, dirs_flat)

    # reshape back
    rgb = rgb_flat.reshape(N, S, 3)  # (N, S, 3)
    density = density_flat.reshape(N, S, 1)  # (N, S, 1)

    # volume rendering
    rgb_map, depth_map, weights = volume_rendering(rgb, density, t_vals, rays_d)

    return {
        "rgb_map": rgb_map,  # (N, 3)
        "depth_map": depth_map,  # (N,)
        "weights": weights,  # (N, S)
    }


if __name__ == "__main__":
    # for test
    model = NeRF()

    N = 1024
    rays_o = torch.randn(N, 3)
    rays_d = torch.randn(N, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # normalize

    result = render_rays(model, rays_o, rays_d)
    print("rgb_map shape:", result["rgb_map"].shape)  # (1024, 3)
    print("depth_map shape:", result["depth_map"].shape)  # (1024,)
    print("weights shape:", result["weights"].shape)  # (1024, 64)
