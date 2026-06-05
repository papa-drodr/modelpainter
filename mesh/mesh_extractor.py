from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, sparse
from skimage import measure

from NeRF.model import NeRF


def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """
    Laplacian mesh smoothing: move each vertex toward the average of its neighbors.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        iterations: number of smoothing passes

    Returns:
        smoothed vertices (V, 3)
    """
    V = len(vertices)
    # build symmetric adjacency via all face edges
    i0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    i1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([i0, i1])
    cols = np.concatenate([i1, i0])
    data = np.ones(len(rows), dtype=np.float32)

    # row-normalized Laplacian matrix
    L = sparse.csr_matrix((data, (rows, cols)), shape=(V, V))
    row_sums = np.asarray(L.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_diag = sparse.diags(1.0 / row_sums)
    L = inv_diag.dot(L)

    verts = vertices.astype(np.float64)
    for _ in range(iterations):
        verts = L.dot(verts)
    return verts.astype(np.float32)


def extract_density_field(
    model: NeRF,
    resolution: int = 128,
    bound: float = 3.0,
    batch_size: int = 65536,
    device: str = "cuda",
) -> np.ndarray:
    """
    Query NeRF density field on a 3D grid.

    Args:
        model: trained NeRF model
        resolution: grid resolution (resolution^3 points total)
        bound: scene bound (-bound to +bound on each axis)
        batch_size: number of points per forward pass
        device: compute device

    Returns:
        density_grid: 3D numpy array of shape (resolution, resolution, resolution)
    """
    model.eval()

    # create 3D grid of points
    linspace = torch.linspace(-bound, bound, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing="ij")
    pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # (N, 3)

    # dummy direction (density does not depend on direction)
    dirs = torch.zeros_like(pts)

    density_values = []

    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            pts_batch = pts[i : i + batch_size].to(device)
            dirs_batch = dirs[i : i + batch_size].to(device)

            _, density = model(pts_batch, dirs_batch)
            density = density.squeeze(-1)  # (batch,), non-negative from model
            density_values.append(density.cpu())

    density_grid = torch.cat(density_values).reshape(resolution, resolution, resolution)

    return density_grid.numpy()


def extract_mesh(
    model: NeRF,
    output_path: str,
    resolution: int = 128,
    bound: float = 3.0,
    threshold: float | None = None,
    smooth_iter: int = 3,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from NeRF density field using Marching Cubes.

    Args:
        model: trained NeRF model
        output_path: path to save .obj file
        resolution: density grid resolution
        bound: scene bound
        threshold: density isosurface level (auto = 95th percentile if None)
        smooth_iter: Laplacian smoothing iterations (0 = no smoothing)
        device: compute device

    Returns:
        vertices: mesh vertices (V, 3)
        faces: mesh faces (F, 3)
    """
    print(f"Extracting density field at resolution {resolution}^3...")
    density_grid = extract_density_field(model, resolution, bound, device=device)

    dmax = float(density_grid.max())
    if threshold is None:
        threshold = max(float(np.percentile(density_grid, 95)), 0.01)
        print(f"Adaptive threshold: {threshold:.4f} (95th percentile, density max={dmax:.2f})")
    elif threshold >= dmax:
        old = threshold
        threshold = max(float(np.percentile(density_grid, 95)), 0.01)
        print(f"Warning: --threshold {old:.1f} > density max {dmax:.2f}, using adaptive {threshold:.4f}")

    # remove floaters: keep only the largest connected component
    binary = density_grid > threshold
    labeled, num_features = ndimage.label(binary)
    if num_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest = int(np.argmax(sizes)) + 1
        density_grid = np.where(labeled == largest, density_grid, 0.0)
        print(f"Floater removal: kept 1 of {num_features} components")

    print(f"Running Marching Cubes (threshold={threshold})...")
    vertices, faces, _, _ = measure.marching_cubes(density_grid, level=threshold)

    # normalize vertices from grid coords to world coords
    vertices = vertices / (resolution - 1) * 2 * bound - bound  # [-bound, bound]

    if smooth_iter > 0:
        print(f"Applying Laplacian smoothing ({smooth_iter} iterations)...")
        vertices = laplacian_smooth(vertices, faces, iterations=smooth_iter)

    print(f"Mesh extracted: {len(vertices)} vertices, {len(faces)} faces")

    # save as .obj
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_obj(vertices, faces, str(output_path))
    print(f"Mesh saved -> {output_path}")

    return vertices, faces


def _save_obj(vertices: np.ndarray, faces: np.ndarray, path: str):
    """
    Save mesh as .obj file.

    Args:
        vertices: (V, 3)
        faces: (F, 3)
        path: output file path
    """
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # .obj face indices are 1-based
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


if __name__ == "__main__":
    # for test
    from NeRF.trainer import load_model

    model, _ = load_model("./checkpoints/nerf_final.pth")
    vertices, faces = extract_mesh(
        model=model,
        output_path="./output/mesh.obj",
        resolution=128,
        threshold=10.0,
    )
