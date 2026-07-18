"""Motion blur detection and deblur evaluation framework."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.stats import pearsonr


def detect_blur(
    video_path: str, method: str = "laplacian"
) -> dict[str, Any]:
    """Analyze each frame for blur using Laplacian variance and Fourier analysis.

    Args:
        video_path: Path to the video file.
        method: Blur detection method. Currently only ``"laplacian"`` is
            supported and used by default.

    Returns:
        Dictionary with keys:
            - ``blur_scores``: ``(numFrames,)`` array of blur scores (higher = sharper).
            - ``blurry_frames``: list of frame indices below the adaptive threshold.
            - ``threshold``: the threshold used (mean − 2·std of blur scores).
            - ``stats``: dict with mean, std, min, max, p5, p95 of blur scores.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    scores: list[float] = []
    hfrs: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(float(lap_var))

            f_transform = np.fft.fft2(gray.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            radius = min(cy, cx) * 0.5
            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx) ** 2 + (Y - cy) ** 2) > radius ** 2
            total_energy = magnitude.sum()
            high_freq_energy = magnitude[mask].sum() if total_energy > 0 else 0.0
            hfr = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0
            hfrs.append(hfr)
    finally:
        cap.release()

    blur_scores = np.array(scores, dtype=np.float64)
    high_freq_ratios = np.array(hfrs, dtype=np.float64)

    mean_score = float(np.mean(blur_scores))
    std_score = float(np.std(blur_scores))
    threshold = mean_score - 2.0 * std_score

    blurry_frames = np.where(blur_scores < threshold)[0].tolist()

    stats = {
        "mean": mean_score,
        "std": std_score,
        "min": float(np.min(blur_scores)),
        "max": float(np.max(blur_scores)),
        "p5": float(np.percentile(blur_scores, 5)),
        "p95": float(np.percentile(blur_scores, 95)),
        "high_freq_ratios": high_freq_ratios,
    }

    return {
        "blur_scores": blur_scores,
        "blurry_frames": blurry_frames,
        "threshold": threshold,
        "stats": stats,
    }


def detect_motion_from_skeleton(
    skeleton_data: np.ndarray,
    fps: float = 30.0,
    velocity_threshold: float | None = None,
) -> dict[str, Any]:
    """Detect motion intensity from 3D skeleton joint positions.

    High velocity implies likely motion blur in that frame even without the
    original video.

    Args:
        skeleton_data: Array of shape ``(numFrames, numJoints, 3)`` containing
            3-D joint positions.
        fps: Video frame rate in frames per second.
        velocity_threshold: Optional manual threshold. When ``None`` an
            adaptive threshold (mean + 2·std) is used.

    Returns:
        Dictionary with keys:
            - ``velocity_magnitude``: ``(numFrames, numJoints)`` per-joint speed.
            - ``mean_velocity``: ``(numFrames,)`` mean velocity across joints.
            - ``high_motion_frames``: indices of frames exceeding threshold.
            - ``threshold``: the velocity threshold applied.
    """
    if skeleton_data.ndim != 3 or skeleton_data.shape[2] != 3:
        raise ValueError(
            f"Expected skeleton_data shape (numFrames, numJoints, 3), "
            f"got {skeleton_data.shape}"
        )

    diff = np.diff(skeleton_data, axis=0)
    velocity_magnitude = np.linalg.norm(diff, axis=2)

    mean_velocity = velocity_magnitude.mean(axis=1)

    if velocity_threshold is None:
        velocity_threshold = float(np.mean(mean_velocity) + 2.0 * np.std(mean_velocity))

    high_motion_frames = np.where(mean_velocity > velocity_threshold)[0].tolist()

    return {
        "velocity_magnitude": velocity_magnitude,
        "mean_velocity": mean_velocity,
        "high_motion_frames": high_motion_frames,
        "threshold": velocity_threshold,
    }


def evaluate_deblur_impact(
    blur_scores: np.ndarray,
    tracking_quality: np.ndarray,
) -> dict[str, Any]:
    """Correlate blur scores with tracking quality to assess deblur benefit.

    Args:
        blur_scores: ``(numFrames,)`` array of per-frame blur scores (higher = sharper).
        tracking_quality: ``(numFrames,)`` array of a quality metric such as mean
            keypoint confidence.

    Returns:
        Dictionary with keys:
            - ``correlation``: Pearson correlation coefficient between blur and quality.
            - ``recommendation``: one of ``"deblur likely helps"``,
              ``"marginal benefit"``, or ``"not recommended"``.
            - ``blur_quality_threshold``: blur score below which quality degrades.
    """
    if blur_scores.shape[0] != tracking_quality.shape[0]:
        raise ValueError(
            f"Length mismatch: blur_scores ({blur_scores.shape[0]}) vs "
            f"tracking_quality ({tracking_quality.shape[0]})"
        )

    r, p_value = pearsonr(blur_scores, tracking_quality)

    q25 = float(np.percentile(tracking_quality, 25))
    low_quality_mask = tracking_quality < q25
    if low_quality_mask.any():
        blur_quality_threshold = float(np.percentile(blur_scores[low_quality_mask], 75))
    else:
        blur_quality_threshold = float(np.percentile(blur_scores, 25))

    abs_r = abs(r)
    if abs_r > 0.5 and p_value < 0.05:
        recommendation = "deblur likely helps"
    elif abs_r > 0.25 and p_value < 0.1:
        recommendation = "marginal benefit"
    else:
        recommendation = "not recommended"

    return {
        "correlation": float(r),
        "p_value": float(p_value),
        "recommendation": recommendation,
        "blur_quality_threshold": blur_quality_threshold,
    }
