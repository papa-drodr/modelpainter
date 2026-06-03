from pathlib import Path

import cv2 as cv
import numpy as np


def is_blurry(frame: np.ndarray, threshold: float = 30) -> bool:
    """
    Detect if a frame is blurry using Laplacian variance.

    Args:
        frame: input frame (BGR)
        threshold: frames with variance below this are considered blurry

    Returns:
        True if blurry, False otherwise
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var() < threshold


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 200,
    resize: tuple[int, int] | None = (800, 450),
    blur_threshold: float = 30.0,
) -> list[str]:
    """
    Extract frames from video by uniform sampling after filtering blurry frames.

    Steps:
        1. Scan all frames and collect non-blurry frame indices
        2. Uniformly sample num_frames from non-blurry frames
        3. Save sampled frames

    Args:
        video_path: input video path
        output_dir: dir saving frames
        num_frames: number of frames to extract (default: 200)
        resize: (width, height) to resize images (default: (800, 450))
        blur_threshold: frames with Laplacian variance below this are skipped (default: 100.0)

    Returns:
        list of saved frame paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Video not found: {video_path}")

    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"video info: {total_frames} frames / {fps:.1f} fps / {duration:.1f} sec")
    print("Scanning for blurry frames...")

    # step 1: scan all frames and collect non-blurry indices
    non_blurry_indices = []
    for frame_idx in range(total_frames):
        video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            continue
        if not is_blurry(frame, threshold=blur_threshold):
            non_blurry_indices.append(frame_idx)

        if (frame_idx + 1) % 100 == 0:
            print(
                f"  scanned {frame_idx + 1}/{total_frames} frames (non-blurry: {len(non_blurry_indices)})"
            )

    print(f"Non-blurry frames: {len(non_blurry_indices)} / {total_frames}")

    if len(non_blurry_indices) == 0:
        raise RuntimeError("No non-blurry frames found. Try lowering blur_threshold.")

    if num_frames > len(non_blurry_indices):
        print(
            f"Requested {num_frames} frames exceeds non-blurry count ({len(non_blurry_indices)})."
        )
        print(f"Adjusting to {len(non_blurry_indices)}.")
        num_frames = len(non_blurry_indices)

    # step 2: uniform sampling from non-blurry frames
    sample_indices = np.linspace(0, len(non_blurry_indices) - 1, num_frames, dtype=int)
    selected_indices = [non_blurry_indices[i] for i in sample_indices]

    # step 3: save sampled frames
    saved_paths = []
    for i, frame_idx in enumerate(selected_indices):
        video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()

        if not ret:
            print(f"frame {frame_idx} read failed, skip")
            continue

        if resize is not None:
            frame = cv.resize(frame, resize)

        filename = output_path / f"frame_{i:04d}.jpg"
        cv.imwrite(str(filename), frame)
        saved_paths.append(str(filename))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_frames} frames saved")

    video.release()
    print(f"\nTotal {len(saved_paths)} frames saved -> {output_dir}")

    return saved_paths


if __name__ == "__main__":
    # for test
    extract_frames(video_path="test.mp4", output_dir="./frames", num_frames=200)
