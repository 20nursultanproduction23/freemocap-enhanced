"""
#8 DESYNC DETECTION — Detect camera time synchronization issues

Camera time desync causes characteristic artifacts in triangulated 3D data:
1. Systematic depth (Z) errors during lateral motion
2. Increased jitter during fast motion (from triangulation mismatch)
3. Asymmetric motion patterns (one axis smoother than others)

This module provides diagnostic detection of these symptoms.
Since FreeMoCap outputs already-triangulated 3D data (without per-camera
2D points), we can only detect symptoms, not fix root causes.
"""
import numpy as np
from joint_definitions import BODY_IDX, NUM_BODY


def detect_desync_symptoms(skeleton_data, fps=30.0):
    """Detect symptoms of camera time desync in triangulated 3D data.

    Checks for:
    1. Excessive depth (Z) jitter relative to XY jitter
    2. Correlated high-frequency noise across distant joints
    3. Motion-dependent noise (worse during fast movement)

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        symptoms: dict with desync indicators and severity
    """
    body_data = skeleton_data[:, :NUM_BODY, :]
    num_frames = body_data.shape[0]

    valid_frames = np.zeros(num_frames, dtype=bool)
    for f in range(num_frames):
        valid_frames[f] = np.sum(~np.isnan(body_data[f]).any(axis=1)) > 10

    if np.sum(valid_frames) < 20:
        return {"severity": "unknown", "message": "Too few valid frames"}

    xy_jitter = np.zeros(num_frames)
    z_jitter = np.zeros(num_frames)
    frame_speed = np.zeros(num_frames)
    valid_jitter = np.zeros(num_frames, dtype=bool)

    for f in range(1, num_frames):
        if not (valid_frames[f] and valid_frames[f - 1]):
            continue
        curr = body_data[f]
        prev = body_data[f - 1]
        valid = ~np.isnan(curr).any(axis=1) & ~np.isnan(prev).any(axis=1)
        if np.sum(valid) < 5:
            continue

        diff = curr[valid] - prev[valid]
        xy_jitter[f] = np.mean(np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2))
        z_jitter[f] = np.mean(np.abs(diff[:, 2]))
        frame_speed[f] = np.mean(np.linalg.norm(diff, axis=1))
        valid_jitter[f] = True

    valid_both = valid_jitter.copy()

    if np.sum(valid_both) < 10:
        return {"severity": "unknown", "message": "Insufficient jitter data"}

    z_xy_ratio = z_jitter[valid_both] / np.maximum(xy_jitter[valid_both], 1e-8)
    median_ratio = np.median(z_xy_ratio)
    mean_ratio = np.mean(z_xy_ratio)

    fast_mask = frame_speed > np.percentile(frame_speed[valid_both], 75)
    slow_mask = frame_speed < np.percentile(frame_speed[valid_both], 25)

    fast_z_xy = np.median(z_xy_ratio[fast_mask[valid_both]]) if np.any(fast_mask & valid_both) else median_ratio
    slow_z_xy = np.median(z_xy_ratio[slow_mask[valid_both]]) if np.any(slow_mask & valid_both) else median_ratio

    speed_correlation = 0.0
    if np.any(valid_both):
        speeds = frame_speed[valid_both]
        ratios = z_xy_ratio
        if np.std(speeds) > 0 and np.std(ratios) > 0:
            speed_correlation = float(np.corrcoef(speeds, ratios)[0, 1])

    spatial_noise = _compute_spatial_noise_correlation(body_data, valid_frames)

    severity = "low"
    if median_ratio > 1.5 or speed_correlation > 0.3:
        severity = "moderate"
    if median_ratio > 2.5 or speed_correlation > 0.5:
        severity = "high"

    return {
        "severity": severity,
        "z_xy_jitter_ratio_median": float(median_ratio),
        "z_xy_jitter_ratio_mean": float(mean_ratio),
        "fast_motion_ratio": float(fast_z_xy),
        "slow_motion_ratio": float(slow_z_xy),
        "speed_correlation": float(speed_correlation),
        "spatial_noise_correlation": spatial_noise,
        "message": _severity_message(severity, median_ratio, speed_correlation),
    }


def _compute_spatial_noise_correlation(body_data, valid_frames):
    """Check if noise is correlated across spatially distant joints.

    If cameras are desynced, the triangulation error affects all joints
    simultaneously, creating correlated noise across the entire skeleton.
    """
    num_frames = body_data.shape[0]

    wrist_l = BODY_IDX["left_wrist"]
    wrist_r = BODY_IDX["right_wrist"]

    l_vel = np.zeros(num_frames)
    r_vel = np.zeros(num_frames)

    for f in range(1, num_frames):
        if valid_frames[f] and valid_frames[f - 1]:
            l1 = body_data[f - 1, wrist_l]
            l2 = body_data[f, wrist_l]
            r1 = body_data[f - 1, wrist_r]
            r2 = body_data[f, wrist_r]
            if not (np.isnan(l1).any() or np.isnan(l2).any() or
                    np.isnan(r1).any() or np.isnan(r2).any()):
                l_vel[f] = np.linalg.norm(l2 - l1)
                r_vel[f] = np.linalg.norm(r2 - r1)

    both_valid = (l_vel > 0) & (r_vel > 0) & valid_frames
    if np.sum(both_valid) < 10:
        return 0.0

    corr = np.corrcoef(l_vel[both_valid], r_vel[both_valid])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def _severity_message(severity, z_xy_ratio, speed_corr):
    """Generate a human-readable severity message."""
    messages = {
        "low": "No significant desync symptoms detected.",
        "moderate": (
            f"Moderate desync indicators: Z/XY jitter ratio={z_xy_ratio:.2f}, "
            f"speed correlation={speed_corr:.2f}. Camera sync should be verified."
        ),
        "high": (
            f"Strong desync indicators: Z/XY jitter ratio={z_xy_ratio:.2f}, "
            f"speed correlation={speed_corr:.2f}. Camera time sync is likely "
            f"off by >1 frame."
        ),
    }
    return messages.get(severity, "Unknown severity.")
