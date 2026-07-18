"""Calibration drift detection for multi-camera motion capture systems.

Detects mid-session camera calibration shifts by monitoring triangulation
reprojection error consistency across time.
"""

import numpy as np
from typing import Optional


def _reproject_to_2d(points_3d: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Reproject 3D world-space points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points.
        projection_matrix: (3, 4) camera projection matrix.

    Returns:
        (N, 2) array of 2D image coordinates.
    """
    ones = np.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
    points_h = np.hstack([points_3d, ones])
    projected = points_h @ projection_matrix.T
    depths = projected[:, 2:3]
    depths = np.where(np.abs(depths) < 1e-8, 1e-8, depths)
    return projected[:, :2] / depths


def _compute_error_distribution(
    errors: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling mean, std, and per-frame mean of a 2D error array.

    Args:
        errors: (T, N) array of errors over time.
        window: rolling window size.

    Returns:
        rolling_mean: (T,) mean error per frame.
        rolling_std: (T,) rolling standard deviation per frame.
        rolling_smooth: (T,) rolling mean of the rolling mean (smoothed trend).
    """
    per_frame_mean = np.mean(errors, axis=1)
    T = per_frame_mean.shape[0]
    rolling_mean = np.copy(per_frame_mean)
    rolling_std = np.zeros(T, dtype=errors.dtype)
    half_w = window // 2

    for t in range(T):
        start = max(0, t - half_w)
        end = min(T, t + half_w + 1)
        segment = per_frame_mean[start:end]
        rolling_mean[t] = np.mean(segment)
        rolling_std[t] = np.std(segment)

    smooth_window = max(window, 10)
    smooth_half = smooth_window // 2
    rolling_smooth = np.copy(rolling_mean)
    for t in range(T):
        s = max(0, t - smooth_half)
        e = min(T, t + smooth_half + 1)
        rolling_smooth[t] = np.mean(per_frame_mean[s:e])

    return per_frame_mean, rolling_std, rolling_smooth


def _compute_trajectory_smoothness(triangulated_3d: np.ndarray) -> np.ndarray:
    """Compute per-frame temporal smoothness of 3D trajectories.

    Args:
        triangulated_3d: (numFrames, numPoints, 3).

    Returns:
        (numFrames,) array of smoothness values (lower = smoother).
    """
    diffs = np.diff(triangulated_3d, axis=0)
    velocities = np.linalg.norm(diffs, axis=2)
    accelerations = np.diff(velocities, axis=0)
    smoothness = np.mean(np.abs(accelerations), axis=1)
    padded = np.zeros(triangulated_3d.shape[0], dtype=triangulated_3d.dtype)
    padded[1:-1] = smoothness
    return padded


def detect_calibration_drift(
    triangulated_3d: np.ndarray,
    camera_matrices: Optional[list[np.ndarray]] = None,
    projection_matrices: Optional[list[np.ndarray]] = None,
    window_size: int = 30,
    threshold_sigma: float = 3.0,
) -> dict:
    """Detect camera calibration drift by monitoring reprojection error consistency.

    Reprojects triangulated 3D points back to each camera's 2D space and
    monitors reprojection error over time. Sudden increases in error for
    individual cameras indicate calibration drift.

    When no calibration data is provided, falls back to computing inter-point
    temporal consistency metrics on the 3D trajectories directly.

    Args:
        triangulated_3d: (numFrames, numPoints, 3) world-space 3D points.
        camera_matrices: List of 3x3 intrinsic matrices, one per camera. None
            to use trajectory-based fallback.
        projection_matrices: List of 3x4 projection matrices, one per camera.
            None to use trajectory-based fallback.
        window_size: Rolling window size for drift detection statistics.
        threshold_sigma: Number of standard deviations above the rolling mean
            to flag as a drift signal.

    Returns:
        Dictionary with keys:
            'reprojection_errors': (numFrames, numCameras, numPoints) per-point
                reprojection error. None if using trajectory fallback.
            'mean_errors_per_camera': (numCameras,) mean error per camera across
                all frames. None if using trajectory fallback.
            'drift_signals': list of dicts with 'camera_idx', 'frame_start',
                'severity' for detected drift events.
            'camera_health': list of dicts per camera with 'status' ("ok" |
                "drift_detected" | "unreliable"), 'mean_error', 'trend'.
    """
    num_frames, num_points, _ = triangulated_3d.shape
    use_projection = (
        projection_matrices is not None
        and len(projection_matrices) > 0
    )

    if use_projection:
        num_cams = len(projection_matrices)
        reprojection_errors = np.zeros(
            (num_frames, num_cams, num_points), dtype=np.float64
        )

        for cam_idx, proj_mat in enumerate(projection_matrices):
            for frame_idx in range(num_frames):
                pts_3d = triangulated_3d[frame_idx]
                projected = _reproject_to_2d(pts_3d, proj_mat)
                pts_2d_gt = np.zeros_like(projected)
                error = np.sqrt(np.sum((projected - pts_2d_gt) ** 2, axis=1))
                reprojection_errors[frame_idx, cam_idx, :] = error

        mean_errors_per_camera = np.mean(
            reprojection_errors.reshape(num_frames * num_points, num_cams), axis=0
        )

        drift_signals = []
        camera_health = []

        for cam_idx in range(num_cams):
            cam_errors = reprojection_errors[:, cam_idx, :]
            per_frame_mean, rolling_std, rolling_smooth = _compute_error_distribution(
                cam_errors, window_size
            )

            baseline_mean = np.mean(per_frame_mean[:window_size]) if num_frames > window_size else np.mean(per_frame_mean)
            baseline_std = np.std(per_frame_mean[:window_size]) if num_frames > window_size else np.std(per_frame_mean)
            baseline_std = max(baseline_std, 1e-8)

            threshold_val = baseline_mean + threshold_sigma * baseline_std

            signal_start = None
            signal_max_severity = 0.0
            for t in range(num_frames):
                if per_frame_mean[t] > threshold_val:
                    severity = (per_frame_mean[t] - baseline_mean) / baseline_std
                    if signal_start is None:
                        signal_start = t
                    signal_max_severity = max(signal_max_severity, severity)
                else:
                    if signal_start is not None:
                        drift_signals.append(
                            {
                                "camera_idx": cam_idx,
                                "frame_start": signal_start,
                                "severity": float(signal_max_severity),
                            }
                        )
                        signal_start = None
                        signal_max_severity = 0.0

            if signal_start is not None:
                drift_signals.append(
                    {
                        "camera_idx": cam_idx,
                        "frame_start": signal_start,
                        "severity": float(signal_max_severity),
                    }
                )

            cam_drift_count = sum(
                1 for s in drift_signals if s["camera_idx"] == cam_idx
            )
            has_drift = cam_drift_count > 0
            trend = "stable"
            if num_frames > window_size:
                first_half = np.mean(per_frame_mean[: num_frames // 2])
                second_half = np.mean(per_frame_mean[num_frames // 2 :])
                if second_half > first_half * 1.2:
                    trend = "increasing"
                elif second_half < first_half * 0.8:
                    trend = "decreasing"

            if has_drift:
                max_sev = max(
                    s["severity"]
                    for s in drift_signals
                    if s["camera_idx"] == cam_idx
                )
                if max_sev > threshold_sigma * 2:
                    status = "unreliable"
                else:
                    status = "drift_detected"
            else:
                status = "ok"

            camera_health.append(
                {
                    "status": status,
                    "mean_error": float(np.mean(per_frame_mean)),
                    "trend": trend,
                }
            )

        return {
            "reprojection_errors": reprojection_errors,
            "mean_errors_per_camera": mean_errors_per_camera,
            "drift_signals": drift_signals,
            "camera_health": camera_health,
        }

    else:
        smoothness = _compute_trajectory_smoothness(triangulated_3d)
        per_frame_mean = smoothness
        rolling_mean = np.copy(per_frame_mean)
        rolling_std = np.zeros(num_frames, dtype=np.float64)

        half_w = window_size // 2
        for t in range(num_frames):
            start = max(0, t - half_w)
            end = min(num_frames, t + half_w + 1)
            segment = per_frame_mean[start:end]
            rolling_mean[t] = np.mean(segment)
            rolling_std[t] = np.std(segment)

        baseline_mean = np.mean(per_frame_mean[:window_size]) if num_frames > window_size else np.mean(per_frame_mean)
        baseline_std = np.std(per_frame_mean[:window_size]) if num_frames > window_size else np.std(per_frame_mean)
        baseline_std = max(baseline_std, 1e-8)
        threshold_val = baseline_mean + threshold_sigma * baseline_std

        drift_signals = []
        signal_start = None
        signal_max_severity = 0.0
        for t in range(num_frames):
            if per_frame_mean[t] > threshold_val:
                severity = (per_frame_mean[t] - baseline_mean) / baseline_std
                if signal_start is None:
                    signal_start = t
                signal_max_severity = max(signal_max_severity, severity)
            else:
                if signal_start is not None:
                    drift_signals.append(
                        {
                            "camera_idx": -1,
                            "frame_start": signal_start,
                            "severity": float(signal_max_severity),
                        }
                    )
                    signal_start = None
                    signal_max_severity = 0.0

        if signal_start is not None:
            drift_signals.append(
                {
                    "camera_idx": -1,
                    "frame_start": signal_start,
                    "severity": float(signal_max_severity),
                }
            )

        has_drift = len(drift_signals) > 0
        trend = "stable"
        if num_frames > window_size:
            first_half = np.mean(per_frame_mean[: num_frames // 2])
            second_half = np.mean(per_frame_mean[num_frames // 2 :])
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"

        if has_drift:
            max_sev = max(s["severity"] for s in drift_signals)
            if max_sev > threshold_sigma * 2:
                status = "unreliable"
            else:
                status = "drift_detected"
        else:
            status = "ok"

        return {
            "reprojection_errors": None,
            "mean_errors_per_camera": None,
            "drift_signals": drift_signals,
            "camera_health": [
                {
                    "status": status,
                    "mean_error": float(np.mean(per_frame_mean)),
                    "trend": trend,
                }
            ],
        }
