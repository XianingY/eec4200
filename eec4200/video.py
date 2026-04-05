from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video decoding. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return cv2


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for video preprocessing. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return np


def probe_video(video_path: Path | str) -> dict[str, float | int | str]:
    cv2 = _require_cv2()
    path = str(video_path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {
            "path": path,
            "opened": False,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration_sec": 0.0,
        }

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return {
        "path": path,
        "opened": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }


def can_decode_video(video_path: Path | str) -> bool:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False
    ok, _ = capture.read()
    capture.release()
    return bool(ok)


def _read_all_frames(video_path: Path | str):
    cv2 = _require_cv2()
    np = _require_numpy()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def _sample_rgb_frames(video_path: Path | str, sample_positions: Sequence[float]):
    cv2 = _require_cv2()
    np = _require_numpy()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        capture.release()
        frames = _read_all_frames(video_path)
        return [frames[min(len(frames) - 1, max(0, int(round((len(frames) - 1) * position))))] for position in sample_positions]

    sampled_frames = []
    for position in sample_positions:
        index = min(frame_count - 1, max(0, int(round((frame_count - 1) * position))))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if not sampled_frames:
        frames = _read_all_frames(video_path)
        return [frames[min(len(frames) - 1, max(0, int(round((len(frames) - 1) * position))))] for position in sample_positions]
    return sampled_frames


def sample_frame_indices(
    total_frames: int,
    clip_length: int,
    clip_index: int = 0,
    num_clips: int = 1,
    jitter: bool = False,
    rng: random.Random | None = None,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if total_frames == 1:
        return [0] * clip_length

    if total_frames < clip_length:
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (clip_length - total_frames))
        return indices

    stride = (total_frames - clip_length) / max(1, num_clips)
    if num_clips == 1:
        start = max(0, (total_frames - clip_length) // 2)
    else:
        start = int(round(stride * clip_index))
        start = min(start, total_frames - clip_length)

    if jitter and total_frames > clip_length:
        rng = rng or random.Random()
        jitter_extent = max(1, int((total_frames - clip_length) * 0.1))
        start = min(max(0, start + rng.randint(-jitter_extent, jitter_extent)), total_frames - clip_length)

    if clip_length == 1:
        return [start]
    step = (clip_length - 1) / (clip_length - 1)
    return [start + int(round(i * step)) for i in range(clip_length)]


def _apply_luminance_clahe(frame):
    cv2 = _require_cv2()
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _apply_random_photometric(frame, rng: random.Random, brightness_range: tuple[float, float] = (0.85, 1.15), gamma_range: tuple[float, float] = (0.85, 1.15)):
    np = _require_numpy()
    frame = frame.astype(np.float32) / 255.0
    brightness = rng.uniform(*brightness_range)
    gamma = rng.uniform(*gamma_range)
    frame = np.clip(frame * brightness, 0.0, 1.0)
    frame = np.power(frame, gamma)
    return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)


def preprocess_frames(
    frames,
    image_size: int,
    apply_clahe: bool = False,
    apply_random_photometric_aug: bool = False,
    horizontal_flip: bool = False,
    rng: random.Random | None = None,
    photometric_brightness_range: tuple[float, float] = (0.85, 1.15),
    photometric_gamma_range: tuple[float, float] = (0.85, 1.15),
):
    cv2 = _require_cv2()
    np = _require_numpy()
    rng = rng or random.Random()

    processed = []
    for frame in frames:
        current = frame
        if apply_clahe:
            current = _apply_luminance_clahe(current)
        if apply_random_photometric_aug:
            current = _apply_random_photometric(current, rng, brightness_range=photometric_brightness_range, gamma_range=photometric_gamma_range)
        current = cv2.resize(current, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if horizontal_flip:
            current = np.ascontiguousarray(current[:, ::-1, :])
        processed.append(current)
    return np.stack(processed, axis=0)


def load_video_clip(
    video_path: Path | str,
    clip_length: int,
    image_size: int,
    clip_index: int = 0,
    num_clips: int = 1,
    jitter: bool = False,
    apply_clahe: bool = False,
    apply_random_photometric_aug: bool = False,
    random_horizontal_flip: bool = False,
    rng: random.Random | None = None,
    photometric_brightness_range: tuple[float, float] = (0.85, 1.15),
    photometric_gamma_range: tuple[float, float] = (0.85, 1.15),
):
    np = _require_numpy()
    rng = rng or random.Random()
    frames = _read_all_frames(video_path)
    indices = sample_frame_indices(
        total_frames=len(frames),
        clip_length=clip_length,
        clip_index=clip_index,
        num_clips=num_clips,
        jitter=jitter,
        rng=rng,
    )
    selected = frames[indices]
    processed = preprocess_frames(
        selected,
        image_size=image_size,
        apply_clahe=apply_clahe,
        apply_random_photometric_aug=apply_random_photometric_aug,
        horizontal_flip=random_horizontal_flip and rng.random() < 0.5,
        rng=rng,
        photometric_brightness_range=photometric_brightness_range,
        photometric_gamma_range=photometric_gamma_range,
    )
    processed = processed.astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    processed = (processed - mean) / std
    return processed


def extract_reference_frame(video_path: Path | str, relative_position: float = 0.5):
    return _sample_rgb_frames(video_path, [relative_position])[0]


def estimate_video_brightness(video_path: Path | str, sample_positions: Sequence[float] = (0.2, 0.5, 0.8)) -> float:
    np = _require_numpy()
    values = [float(frame.astype(np.float32).mean()) for frame in _sample_rgb_frames(video_path, sample_positions)]
    return float(np.mean(values))
