import json
import numpy as np
import pycolmap
from pathlib import Path


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
    output_path: str,
) -> dict:
    """
    Convert pycolmap reconstruction to NeRF transforms.json format.

    Args:
        reconstruction: pycolmap Reconstruction object
        output_path: path to save transforms.json

    Returns:
        transforms dict
    """
    frames = []

    for _, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]

        # c2w (camera-to-world) matrix
        # pycolmap stores w2c, so inverse is needed
        R = image.rotation_matrix()  # 3x3 rotation (world-to-camera)
        t = image.tvec  # 3, translation (world-to-camera)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        c2w = np.linalg.inv(w2c)

        """
        R (3×3)          t (3,)
        [ R00 R01 R02 ]  [ t0 ]
        [ R10 R11 R12 ]  [ t1 ]
        [ R20 R21 R22 ]  [ t2 ]

        w2c[:3,:3] = R| w2c[:3,3] = t
        [ R00 R01 R02 | t0 ]
        [ R10 R11 R12 | t1 ]
        [ R20 R21 R22 | t2 ]
        [  0   0   0  |  1 ]

        R → Rᵀ          t → -Rᵀt
        [ R00 R10 R20 | tx ]
        [ R01 R11 R21 | ty ]
        [ R02 R12 R22 | tz ]
        [  0   0   0  |  1 ]
        """

        # flip y and z axis (OpenCV -> OpenGL convention)
        c2w[:, 1] *= -1
        c2w[:, 2] *= -1

        frame = {
            "file_path": f"./frames/{Path(image.name).stem}",
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
        output_path=str(transforms_path),
    )

    return transforms


if __name__ == "__main__":
    # for test
    run_colmap_pipeline(
        image_dir="./frames",
        output_dir="./colmap",
    )
