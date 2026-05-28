import json
from pathlib import Path

import numpy as np
import pycolmap


def run_reconstruction(image_dir: str, output_dir: str) -> pycolmap.Reconstruction:
    """
    Run incremental SfM reconstruction using pycolmap.

    Args:
        image_dir: dir containing input frames
        output_dir: dir to save colmap database and sparse model

    Returns:
        pycolmap Reconstruction object
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_path = output_path / "colmap.db"

    print("Extracting features...")
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=image_dir,
    )

    print("---Matching features---")
    pycolmap.match_exhaustive(database_path=str(db_path))

    print("---Running incremental mapping---")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=image_dir,
        output_path=str(output_path / "sparse"),
    )

    if not reconstructions:
        raise RuntimeError(
            "Reconstruction failed. Check if frames have enough overlap."
        )

    # select reconstruction with the most registered images
    reconstruction = max(reconstructions.values(), key=lambda r: len(r.images))
    print(f"Reconstruction done: {len(reconstruction.images)} images registered.")

    return reconstruction


def reconstruction_to_transforms(
    reconstruction: pycolmap.Reconstruction,
    image_dir: str,
    output_path: str,
) -> dict:
    """
    Convert pycolmap reconstruction to NeRF transforms.json format.

    Args:
        reconstruction: pycolmap Reconstruction object
        image_dir: dir containing input frames (for file_path reference)
        output_path: path to save transforms.json

    Returns:
        transforms dict
    """
    frames = []
    image_dir_path = Path(image_dir).resolve()

    for _, image in reconstruction.images.items():

        # c2w (camera-to-world) matrix
        # pycolmap stores w2c, so inverse is needed
        R = image.rotation_matrix()  # 3x3 rotation (world-to-camera)
        t = image.tvec  # 3, translation (world-to-camera)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        c2w = np.linalg.inv(w2c)

        # flip y and z axis (OpenCV -> OpenGL convention)
        c2w[:, 1] *= -1
        c2w[:, 2] *= -1

        # store absolute path to avoid resolution issues
        frame_stem = Path(image.name).stem
        frame_path = str(image_dir_path / frame_stem)

        frame = {
            "file_path": frame_path,
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    # compute camera_angle_x from focal length
    # assume PINHOLE model (params[0] = focal length)
    camera = list(reconstruction.cameras.values())[0]
    focal_length = camera.focal_length
    width = camera.width
    camera_angle_x = 2 * np.arctan(width / (2 * focal_length))

    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames,
    }

    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print(f"transforms.json saved -> {output_path}")

    return transforms


def run_colmap_pipeline(image_dir: str, output_dir: str) -> dict:
    """
    Full colmap pipeline: reconstruction -> transforms.json

    Args:
        image_dir: dir containing input frames
        output_dir: dir to save all colmap outputs

    Returns:
        transforms dict
    """
    reconstruction = run_reconstruction(image_dir, output_dir)

    transforms_path = Path(output_dir) / "transforms.json"
    transforms = reconstruction_to_transforms(
        reconstruction=reconstruction,
        image_dir=image_dir,
        output_path=str(transforms_path),
    )

    return transforms


if __name__ == "__main__":
    # for test
    run_colmap_pipeline(
        image_dir="./frames",
        output_dir="./colmap",
    )
