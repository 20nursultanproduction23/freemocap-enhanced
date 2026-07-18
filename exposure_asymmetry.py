"""
#11 EXPOSURE ASYMMETRY — Detect brightness/exposure-related tracking artifacts

Different camera exposures cause asymmetric tracking quality:
- Overexposed regions lose detail → poor tracking
- Underexposed regions have high noise → jittery tracking
- One camera may consistently track better than another

Since we only have triangulated 3D data (not raw video), we detect
exposure-related artifacts indirectly through:
1. Left/right tracking quality asymmetry
2. Position-dependent noise patterns
3. Correlation between body position and tracking confidence
"""
import numpy as np
from joint_definitions import BODY_IDX, NUM_BODY


def detect_exposure_asymmetry(skeleton_data, fps=30.0):
    """Detect asymmetric tracking quality that may indicate exposure issues.

    Compares left-side vs right-side body joint tracking quality.
    Systematic differences suggest one camera has worse exposure/tracking.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        result: dict with asymmetry metrics and severity assessment
    """
    num_frames = skeleton_data.shape[0]

    left_joints = [
        "left_shoulder", "left_elbow", "left_wrist",
        "left_hip", "left_knee", "left_ankle",
    ]
    right_joints = [
        "right_shoulder", "right_elbow", "right_wrist",
        "right_hip", "right_knee", "right_ankle",
    ]

    left_nan_rate = _compute_side_nan_rate(skeleton_data, left_joints)
    right_nan_rate = _compute_side_nan_rate(skeleton_data, right_joints)

    left_jitter = _compute_side_jitter(skeleton_data, left_joints)
    right_jitter = _compute_side_jitter(skeleton_data, right_joints)

    left_depth_noise = _compute_depth_noise(skeleton_data, left_joints)
    right_depth_noise = _compute_depth_noise(skeleton_data, right_joints)

    nan_diff = abs(left_nan_rate - right_nan_rate)
    jitter_diff = abs(left_jitter - right_jitter)
    depth_diff = abs(left_depth_noise - right_depth_noise)

    worse_side = "left" if left_nan_rate + left_jitter > right_nan_rate + right_jitter else "right"

    severity = "low"
    if nan_diff > 5 or jitter_diff > 5.0 or depth_diff > 10.0:
        severity = "moderate"
    if nan_diff > 15 or jitter_diff > 15.0 or depth_diff > 30.0:
        severity = "high"

    return {
        "left_nan_rate": left_nan_rate,
        "right_nan_rate": right_nan_rate,
        "nan_rate_diff": nan_diff,
        "left_jitter_mm": left_jitter,
        "right_jitter_mm": right_jitter,
        "jitter_diff_mm": jitter_diff,
        "left_depth_noise_mm": left_depth_noise,
        "right_depth_noise_mm": right_depth_noise,
        "depth_noise_diff_mm": depth_diff,
        "worse_side": worse_side,
        "severity": severity,
        "message": _exposure_message(severity, worse_side, nan_diff,
                                      jitter_diff, depth_diff),
    }


def compute_position_dependent_quality(skeleton_data, num_bins=5):
    """Analyze how tracking quality varies with X position.

    If one camera is on the left and another on the right, tracking
    quality should degrade toward the edges where that camera dominates.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        num_bins: number of X-position bins

    Returns:
        result: dict with per-bin quality metrics
    """
    body_data = skeleton_data[:, :NUM_BODY, :]

    nose_x = body_data[:, BODY_IDX["nose"], 0]
    valid_x = ~np.isnan(nose_x)

    if np.sum(valid_x) < num_bins * 5:
        return {"bins": [], "message": "Insufficient valid data"}

    x_min = np.nanmin(nose_x)
    x_max = np.nanmax(nose_x)
    bin_edges = np.linspace(x_min, x_max, num_bins + 1)

    bins = []
    for b in range(num_bins):
        mask = (nose_x >= bin_edges[b]) & (nose_x < bin_edges[b + 1])
        if b == num_bins - 1:
            mask = (nose_x >= bin_edges[b]) & (nose_x <= bin_edges[b + 1])

        bin_data = body_data[mask]
        if bin_data.shape[0] < 3:
            continue

        nan_rate = float(np.mean(np.isnan(bin_data).any(axis=2).any(axis=1))) * 100

        velocities = []
        for f in range(1, bin_data.shape[0]):
            prev = bin_data[f - 1]
            curr = bin_data[f]
            valid = ~np.isnan(prev).any(axis=1) & ~np.isnan(curr).any(axis=1)
            if np.sum(valid) > 3:
                velocities.append(float(np.mean(np.linalg.norm(
                    curr[valid] - prev[valid], axis=1
                ))))
        mean_jitter = float(np.mean(velocities)) if velocities else 0.0

        bins.append({
            "x_center": float((bin_edges[b] + bin_edges[b + 1]) / 2),
            "frame_count": int(np.sum(mask)),
            "nan_rate_pct": nan_rate,
            "mean_jitter_mm": mean_jitter,
        })

    return {
        "bins": bins,
        "x_range": [float(x_min), float(x_max)],
        "message": (f"Quality varies across X range — check if worse "
                    f"edges correspond to camera positions"),
    }


def _compute_side_nan_rate(skeleton_data, joint_names):
    total = 0
    nan_count = 0
    for name in joint_names:
        idx = BODY_IDX[name]
        nan_count += int(np.sum(np.isnan(skeleton_data[:, idx, 0])))
        total += skeleton_data.shape[0]
    return (nan_count / total * 100) if total > 0 else 0.0


def _compute_side_jitter(skeleton_data, joint_names):
    displacements = []
    for name in joint_names:
        idx = BODY_IDX[name]
        pos = skeleton_data[:, idx, :]
        valid = ~np.isnan(pos).any(axis=1)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            continue
        for i in range(len(valid_idx) - 1):
            f1, f2 = valid_idx[i], valid_idx[i + 1]
            disp = np.linalg.norm(pos[f2] - pos[f1])
            displacements.append(disp)
    return float(np.mean(displacements)) if displacements else 0.0


def _compute_depth_noise(skeleton_data, joint_names):
    z_stds = []
    for name in joint_names:
        idx = BODY_IDX[name]
        z = skeleton_data[:, idx, 2]
        valid = ~np.isnan(z)
        if np.sum(valid) > 10:
            z_stds.append(float(np.std(z[valid])))
    return float(np.mean(z_stds)) if z_stds else 0.0


def _exposure_message(severity, worse_side, nan_diff, jitter_diff, depth_diff):
    if severity == "high":
        return (f"STRONG exposure/tracking asymmetry: {worse_side} side worse "
                f"(NaN diff={nan_diff:.1f}%, jitter diff={jitter_diff:.1f}mm, "
                f"depth noise diff={depth_diff:.1f}mm). Check camera exposures.")
    elif severity == "moderate":
        return (f"MODERATE asymmetry detected: {worse_side} side slightly worse. "
                f"Verify camera settings are matched.")
    return "No significant exposure asymmetry detected."
