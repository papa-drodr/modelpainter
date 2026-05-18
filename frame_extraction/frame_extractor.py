import cv2 as cv
import numpy as np
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 150,
    resize: tuple[int, int] | None = (800, 450),
) -> list[str]:
    """
    Extract the frame from the image by uniform sampling.

    Args:
        video_path: input video path
        output_dir: dir saving frames
        num_frames: number extract frames (default: 150)
        resize: (width, height) to resize images (default: (800, 450))
    Returns:
        list of saved frame paths
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Video not found.: {video_path}")

    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    duration = total_frames / fps

    print(f"video info: {total_frames} frames / {fps:.1f} fps / {duration:.1f} sec")

    if num_frames > total_frames:
        print(f"Requested {num_frames} frames exceeds total ({total_frames}).")
        print(f"Adjusting to {total_frames}.")
        num_frames = total_frames

    # uniform sampling: total_frame -> num_frames select uniformly
    sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    saved_paths = []

    for i, frame_idx in enumerate(sample_indices):
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
            print(f" {i + 1}/{num_frames} frames saved")

    video.release()
    print(f"\nTotal {len(saved_paths)} frames saved -> {output_dir}")

    return saved_paths


if __name__ == "__main__":
    # for test
    extract_frames(video_path="test.mp4", output_dir="./frames", num_frames=150)
