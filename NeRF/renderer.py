import torch
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
    background: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform volume rendering to composite colors along each ray.

    Args:
        rgb: predicted colors at each sample (N, num_samples, 3)
        density: predicted density at each sample (N, num_samples, 1)
        t_vals: sample distances along ray (N, num_samples)
        rays_d: ray directions (N, 3) for computing real distances
        background: background color blended for unoccupied rays (0=black, 1=white)

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
    sigma = density[..., 0]  # (N, num_samples), non-negative from model
    alpha = 1.0 - torch.exp(-sigma * dists)  # (N, num_samples)

    # transmittance T_i = exp(-sum_{j<i} sigma_j * delta_j), log-space for stability
    log_T = torch.cumsum(-sigma * dists, dim=-1)
    log_T = torch.cat(
        [torch.zeros_like(log_T[:, :1]), log_T[:, :-1]], dim=-1
    )
    transmittance = torch.exp(log_T)  # (N, num_samples)

    # weight = T_i * alpha_i
    weights = transmittance * alpha  # (N, num_samples)

    # composite color: sum of weighted RGB
    rgb_map = (weights[..., None] * rgb).sum(dim=1)  # (N, 3)

    # blend unoccupied ray remainder with background color
    acc = weights.sum(dim=-1, keepdim=True)  # (N, 1) accumulated opacity
    rgb_map = rgb_map + (1.0 - acc) * background

    # depth map: expected distance along ray
    depth_map = (weights * t_vals).sum(dim=-1)  # (N,)

    return rgb_map, depth_map, weights


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    perturb: bool = True,
) -> torch.Tensor:
    """
    Hierarchical sampling: draw fine samples from PDF defined by coarse weights.

    Args:
        bins: bin boundaries (N, num_coarse-1)
        weights: coarse weights as PDF source (N, num_coarse-2)
        num_samples: number of fine samples to draw
        perturb: random sampling (False → uniform grid)

    Returns:
        fine t-values (N, num_samples)
    """
    weights = weights.float() + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[..., :1]), torch.cumsum(pdf, dim=-1)], dim=-1)

    if perturb:
        u = torch.rand(*cdf.shape[:-1], num_samples, device=bins.device)
    else:
        u = torch.linspace(0.0, 1.0, num_samples, device=bins.device)
        u = u.expand(*cdf.shape[:-1], num_samples)
    u = u.contiguous()

    inds = torch.searchsorted(cdf.contiguous(), u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    # direct indexing avoids large (N, num_samples, num_coarse) expand tensors
    row = torch.arange(cdf.shape[0], device=cdf.device).unsqueeze(1)  # (N, 1)
    cdf_g = torch.stack([cdf[row, below], cdf[row, above]], dim=-1)
    bins_f = bins.float()
    bins_g = torch.stack([bins_f[row, below], bins_f[row, above]], dim=-1)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


def render_rays(
    model: NeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    num_samples: int = 64,
    num_fine: int = 64,
    perturb: bool = True,
    background: float = 1.0,
) -> dict:
    """
    Full rendering pipeline with hierarchical (coarse + fine) sampling.

    Args:
        model: NeRF model
        rays_o: ray origins (N, 3)
        rays_d: ray directions (N, 3)
        near: near plane distance
        far: far plane distance
        num_samples: coarse samples per ray
        num_fine: fine samples per ray (0 = coarse only)
        perturb: add perturbation to samples during training
        background: background color for unoccupied rays (0=black, 1=white)

    Returns:
        dict with rgb_map, rgb_map_coarse, depth_map, weights
    """
    # --- coarse pass ---
    pts, t_vals = sample_points_along_rays(
        rays_o, rays_d, near, far, num_samples, perturb
    )
    N, S, _ = pts.shape

    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d[:, None, :].expand_as(pts).reshape(-1, 3)
    rgb_flat, density_flat = model(pts_flat, dirs_flat)
    rgb_c = rgb_flat.reshape(N, S, 3)
    density_c = density_flat.reshape(N, S, 1)
    rgb_map_coarse, depth_map, weights_c = volume_rendering(
        rgb_c, density_c, t_vals, rays_d, background=background
    )

    if num_fine == 0:
        return {
            "rgb_map": rgb_map_coarse,
            "depth_map": depth_map,
            "weights": weights_c,
        }

    # --- fine pass ---
    t_mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])          # (N, S-1)
    t_fine = sample_pdf(t_mids, weights_c[:, 1:-1].detach(), num_fine, perturb)
    t_all, _ = torch.sort(
        torch.cat([t_vals, t_fine], dim=-1), dim=-1
    )  # (N, S+num_fine)

    S_all = t_all.shape[1]
    pts_all = rays_o[:, None, :] + rays_d[:, None, :] * t_all[..., None]
    pts_flat = pts_all.reshape(-1, 3)
    dirs_flat = rays_d[:, None, :].expand(N, S_all, 3).reshape(-1, 3)
    rgb_flat, density_flat = model(pts_flat, dirs_flat)
    rgb_f = rgb_flat.reshape(N, S_all, 3)
    density_f = density_flat.reshape(N, S_all, 1)
    rgb_map, depth_map, weights = volume_rendering(
        rgb_f, density_f, t_all, rays_d, background=background
    )

    return {
        "rgb_map": rgb_map,              # fine result (N, 3)
        "rgb_map_coarse": rgb_map_coarse,  # for coarse loss (N, 3)
        "depth_map": depth_map,
        "weights": weights,
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
    print("weights shape:", result["weights"].shape)  # (1024, 128) = 64 coarse + 64 fine
