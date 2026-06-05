import argparse
from pathlib import Path

import torch

from colmap.reconstructor import run_colmap_pipeline
from frame_extraction.frame_extractor import extract_frames
from mesh.color_baker import bake_face_colors, save_colored_obj
from mesh.mesh_extractor import extract_mesh
from NeRF.trainer import load_model, train


def parse_args():
    parser = argparse.ArgumentParser(
        description="ModelPainter: Video -> 3D Model -> Color Edit"
    )

    parser.add_argument("--video", type=str, required=False, help="Input video path")
    parser.add_argument(
        "--output", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--num_frames", type=int, default=200, help="Number of frames to extract"
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=30.0, help="Blur detection threshold"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="NeRF training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Rays per batch"
    )
    parser.add_argument(
        "--rays_per_epoch",
        type=int,
        default=None,
        help="Max rays sampled per epoch (None = use all). Use to speed up large datasets.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[400, 225],
        metavar=("W", "H"),
        help="Training image size (default: 400 225 / high-res: 800 450)",
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="Marching Cubes resolution"
    )
    parser.add_argument(
        "--near",
        type=float,
        default=None,
        help="Near plane distance (auto-computed if not set)",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=None,
        help="Far plane distance (auto-computed if not set)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Density threshold for mesh extraction (auto 95th percentile if not set)",
    )
    parser.add_argument(
        "--smooth_iter",
        type=int,
        default=3,
        help="Laplacian smoothing iterations after Marching Cubes (0 = off)",
    )
    parser.add_argument(
        "--skip_extract", action="store_true", help="Skip frame extraction"
    )
    parser.add_argument(
        "--skip_colmap", action="store_true", help="Skip COLMAP reconstruction"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip NeRF training and load existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (used with --skip_train)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume training from",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    frames_dir = output_path / "frames"
    colmap_dir = output_path / "colmap"
    ckpt_dir = output_path / "checkpoints"
    mesh_dir = output_path / "mesh"

    device = args.device
    print(f"Device: {device}")

    if not args.skip_extract and args.video is None:
        raise ValueError("--video is required unless --skip_extract is set")

    # step 1: frame extraction
    if args.skip_extract:
        print("\n=== Step 1: Frame Extraction (skipped) ===")
    else:
        print("\n=== Step 1: Frame Extraction ===")
        extract_frames(
            video_path=args.video,
            output_dir=str(frames_dir),
            num_frames=args.num_frames,
            resize=None,
            blur_threshold=args.blur_threshold,
        )

    # step 2: colmap reconstruction
    if args.skip_colmap:
        print("\n=== Step 2: COLMAP Reconstruction (skipped) ===")
    else:
        print("\n=== Step 2: COLMAP Reconstruction ===")
        run_colmap_pipeline(
            image_dir=str(frames_dir),
            output_dir=str(colmap_dir),
        )

    # step 3: nerf training
    if args.skip_train and args.checkpoint:
        print("\n=== Step 3: NeRF Training (skipped) ===")
        print(f"Loading checkpoint: {args.checkpoint}")
        model, ckpt = load_model(args.checkpoint, device=device)
        bound = ckpt.get("bound", 3.0)
    else:
        print("\n=== Step 3: NeRF Training ===")
        model, bound = train(
            data_dir=str(colmap_dir),
            save_dir=str(ckpt_dir),
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            near=args.near,
            far=args.far,
            img_size=tuple(args.img_size),
            resume_from=args.resume,
            rays_per_epoch=args.rays_per_epoch,
            device=device,
        )

    # step 4: mesh extraction
    print("\n=== Step 4: Mesh Extraction ===")
    mesh_path = mesh_dir / "mesh.obj"
    vertices, faces = extract_mesh(
        model=model,
        output_path=str(mesh_path),
        resolution=args.resolution,
        threshold=args.threshold,
        smooth_iter=args.smooth_iter,
        bound=bound,
        device=device,
    )
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(
            "메시 추출 실패: 빈 표면입니다. "
            "--threshold 값을 낮추거나 NeRF 학습을 더 진행하세요."
        )

    # step 5: color baking
    print("\n=== Step 5: Color Baking ===")
    face_colors = bake_face_colors(model, vertices, faces, device=device)
    colored_mesh_path = mesh_dir / "mesh_colored.obj"
    save_colored_obj(vertices, faces, face_colors, str(colored_mesh_path))

    print("\n=== Done ===")
    print(f"Colored mesh -> {colored_mesh_path}")


if __name__ == "__main__":
    main()
