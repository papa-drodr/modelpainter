from pathlib import Path

import numpy as np
import torch

from NeRF.model import NeRF


def bake_face_colors(
    model: NeRF,
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str = "cuda",
    batch_size: int = 65536,
) -> np.ndarray:
    """
    Bake NeRF colors onto mesh faces by querying color at each face centroid.

    Args:
        model: trained NeRF model
        vertices: mesh vertices (V, 3)
        faces: mesh face indices (F, 3)
        device: compute device
        batch_size: number of face centroids per forward pass

    Returns:
        face_colors: RGB color per face (F, 3), values in [0, 1]
    """
    model.eval()

    centroids = vertices[faces].mean(axis=1)  # (F, 3)
    centroids_tensor = torch.tensor(centroids, dtype=torch.float32)

    # query from 6 cardinal directions and average to avoid winding/normal issues
    view_dirs = torch.tensor([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0, -1, 0],
        [ 0, 0, 1], [ 0, 0, -1],
    ], dtype=torch.float32)

    accumulated = np.zeros((len(centroids), 3), dtype=np.float32)

    with torch.no_grad():
        for d in range(6):
            dir_vec = view_dirs[d].unsqueeze(0).expand(len(centroids), -1)
            colors_d = []
            for i in range(0, len(centroids_tensor), batch_size):
                pts = centroids_tensor[i : i + batch_size].to(device)
                dirs = dir_vec[i : i + batch_size].to(device)
                rgb, _ = model(pts, dirs)
                colors_d.append(rgb.cpu().numpy())
            accumulated += np.concatenate(colors_d, axis=0)

    face_colors = accumulated / 6.0

    print(f"Color baking done: {len(face_colors)} faces (6-dir average)")

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
