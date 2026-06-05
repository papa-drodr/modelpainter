from pathlib import Path

import numpy as np
import torch

from NeRF.model import NeRF


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute outward-facing unit normals for each face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)  # avoid divide-by-zero
    return (normals / norms).astype(np.float32)


def bake_face_colors(
    model: NeRF,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cuda",
    batch_size: int = 65536,
    num_dirs: int = 16,
) -> np.ndarray:
    """
    Bake NeRF colors onto mesh faces by averaging over multiple hemisphere directions.

    Queries the NeRF at each face centroid from num_dirs random directions in the
    upper hemisphere (aligned with face normal) and averages to reduce view-dependent
    noise and approximate diffuse surface color.

    Args:
        model: trained NeRF model
        vertices: mesh vertices (V, 3)
        faces: mesh face indices (F, 3)
        device: compute device
        batch_size: number of face centroids per forward pass
        num_dirs: number of random hemisphere directions to average

    Returns:
        face_colors: RGB color per face (F, 3), values in [0, 1]
    """
    model.eval()

    centroids = vertices[faces].mean(axis=1)  # (F, 3)
    normals = _compute_face_normals(vertices, faces)  # (F, 3) unit normals

    F = len(centroids)
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)
    colors_acc = np.zeros((F, 3), dtype=np.float32)

    # sample num_dirs random unit vectors in upper hemisphere per face
    raw = np.random.randn(num_dirs, F, 3).astype(np.float32)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    raw /= np.where(norms < 1e-8, 1.0, norms)
    # flip vectors that point away from face normal
    dots = (raw * normals[None]).sum(axis=-1, keepdims=True)  # (num_dirs, F, 1)
    sampled_dirs = np.where(dots < 0, -raw, raw)  # (num_dirs, F, 3)

    with torch.no_grad():
        for d in range(num_dirs):
            dirs_tensor = torch.tensor(sampled_dirs[d], dtype=torch.float32)
            for i in range(0, F, batch_size):
                pts = centroids_tensor[i : i + batch_size].to(device)
                dirs = dirs_tensor[i : i + batch_size].to(device)
                rgb, _ = model(pts, dirs)
                colors_acc[i : i + batch_size] += rgb.cpu().numpy()

    face_colors = np.clip(colors_acc / num_dirs, 0.0, 1.0)

    print(f"Color baking done: {len(face_colors)} faces ({num_dirs} directions averaged)")

    return face_colors


def save_colored_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    output_path: str,
):
    """
    Save mesh with per-face colors as .obj + .mtl files.

    Args:
        vertices: (V, 3)
        faces: (F, 3)
        face_colors: RGB per face (F, 3), values in [0, 1]
        output_path: path to save .obj file (.mtl saved alongside)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = output_path.with_suffix(".mtl")

    # write .mtl file (one material per face)
    with open(mtl_path, "w") as f:
        for i, color in enumerate(face_colors):
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            f.write(f"newmtl face_{i}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n\n")

    # write .obj file
    with open(output_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for i, face in enumerate(faces):
            f.write(f"usemtl face_{i}\n")
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Colored mesh saved -> {output_path}")


if __name__ == "__main__":
    # for test
    from mesh.mesh_extractor import extract_mesh
    from NeRF.trainer import load_model

    model, _ = load_model("./checkpoints/nerf_final.pth")
    vertices, faces = extract_mesh(model=model, output_path="./output/mesh.obj")
    face_colors = bake_face_colors(model, vertices, faces)
    save_colored_obj(vertices, faces, face_colors, "./output/mesh_colored.obj")
