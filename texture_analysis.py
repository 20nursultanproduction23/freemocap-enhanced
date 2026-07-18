"""
Problem #18: Texture/clothing false positive analysis.

Analyzes spatial patterns in tracking quality to detect texture-induced
artifacts where high-contrast clothing or background patterns trick
MediaPipe into confident but incorrect detections.
"""

from typing import Any

import numpy as np

BODY_REGIONS: dict[str, list[int]] = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "torso": [11, 12, 23, 24],
    "left_arm": [13, 15, 17, 19, 21],
    "right_arm": [14, 16, 18, 20, 22],
    "left_leg": [25, 27, 29, 31],
    "right_leg": [26, 28, 30, 32],
}

JOINT_TO_REGION: dict[int, str] = {}
for _region_name, _indices in BODY_REGIONS.items():
    for _idx in _indices:
        JOINT_TO_REGION[_idx] = _region_name

SIBLING_PAIRS: list[tuple[int, int]] = [
    (11, 12),
    (23, 24),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def _classify_body_region(joint_index: int) -> str:
    if joint_index in JOINT_TO_REGION:
        return JOINT_TO_REGION[joint_index]
    return "unknown"


def _compute_spatial_quality(
    skeleton_data: np.ndarray,
    confidence_data: np.ndarray | None,
) -> dict[str, float]:
    num_frames, num_points, _ = skeleton_data.shape

    region_confidences: dict[str, list[float]] = {name: [] for name in BODY_REGIONS}

    if confidence_data is not None:
        for region_name, joint_indices in BODY_REGIONS.items():
            valid_indices = [i for i in joint_indices if i < num_points]
            if valid_indices:
                region_confs = confidence_data[:, valid_indices]
                region_confidences[region_name] = region_confs.mean(axis=1).tolist()
    else:
        for region_name, joint_indices in BODY_REGIONS.items():
            valid_indices = [i for i in joint_indices if i < num_points]
            if not valid_indices:
                continue
            region_positions = skeleton_data[:, valid_indices, :]
            diffs = np.diff(region_positions, axis=0)
            frame_disp = np.linalg.norm(diffs, axis=2).mean(axis=1)
            max_disp = frame_disp.max()
            if max_disp > 1e-8:
                stability = 1.0 - np.clip(frame_disp / max_disp, 0, 1)
            else:
                stability = np.ones(len(frame_disp))
            region_confidences[region_name] = stability.tolist()

    spatial_quality: dict[str, float] = {}
    for region_name in BODY_REGIONS:
        values = region_confidences[region_name]
        if values:
            spatial_quality[region_name] = float(np.mean(values))
        else:
            spatial_quality[region_name] = 0.0

    return spatial_quality


def _compute_temporal_jitter(
    skeleton_data: np.ndarray,
    fps: float,
) -> dict[str, float]:
    num_frames, num_points, _ = skeleton_data.shape

    if num_frames < 3:
        return {name: 0.0 for name in BODY_REGIONS}

    velocity = np.diff(skeleton_data, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)

    if num_frames < 4:
        acceleration = np.diff(speed, axis=0)
    else:
        acceleration = np.diff(speed, axis=0)

    if acceleration.shape[0] > 0:
        accel_jerk = np.abs(acceleration)
        per_joint_jerk = accel_jerk.mean(axis=0)
    else:
        per_joint_jerk = np.zeros(num_points)

    region_jitter: dict[str, list[float]] = {name: [] for name in BODY_REGIONS}

    for region_name, joint_indices in BODY_REGIONS.items():
        valid_indices = [i for i in joint_indices if i < num_points]
        if valid_indices:
            region_jitter[region_name] = per_joint_jerk[valid_indices].tolist()

    temporal_jitter: dict[str, float] = {}
    for region_name in BODY_REGIONS:
        values = region_jitter[region_name]
        if values:
            temporal_jitter[region_name] = float(np.mean(values))
        else:
            temporal_jitter[region_name] = 0.0

    return temporal_jitter


def analyze_texture_artifacts(
    skeleton_data: np.ndarray,
    fps: float = 30.0,
    confidence_data: np.ndarray | None = None,
) -> dict[str, Any]:
    skeleton_data = np.asarray(skeleton_data, dtype=np.float64)
    if confidence_data is not None:
        confidence_data = np.asarray(confidence_data, dtype=np.float64)

    spatial_quality = _compute_spatial_quality(skeleton_data, confidence_data)
    temporal_jitter = _compute_temporal_jitter(skeleton_data, fps)

    spatial_values = list(spatial_quality.values())
    jitter_values = list(temporal_jitter.values())

    if spatial_values:
        mean_spatial = float(np.mean(spatial_values))
        std_spatial = float(np.std(spatial_values))
    else:
        mean_spatial = 0.0
        std_spatial = 0.0

    if jitter_values:
        mean_jitter = float(np.mean(jitter_values))
        std_jitter = float(np.std(jitter_values))
    else:
        mean_jitter = 0.0
        std_jitter = 0.0

    suspicious_regions: list[tuple[str, str]] = []

    for region_name in BODY_REGIONS:
        quality = spatial_quality[region_name]
        jitter = temporal_jitter[region_name]

        high_jitter = jitter > mean_jitter + 1.5 * std_jitter if std_jitter > 1e-8 else jitter > mean_jitter * 1.5
        low_confidence = quality < mean_spatial - 1.0 * std_spatial if std_spatial > 1e-8 else quality < mean_spatial * 0.7
        low_quality_overall = quality < 0.3

        if high_jitter and low_confidence:
            suspicious_regions.append(
                (region_name, "high jitter with low confidence — likely texture interference")
            )
        elif high_jitter and not low_confidence:
            suspicious_regions.append(
                (region_name, "high jitter despite OK confidence — possible texture artifact")
            )
        elif low_quality_overall:
            suspicious_regions.append(
                (region_name, "consistently low tracking quality in this region")
            )

    region_scores: list[float] = []
    for region_name in BODY_REGIONS:
        q = spatial_quality[region_name]
        j = temporal_jitter[region_name]
        normalized_jitter = j / (mean_jitter * 2 + 1e-8) if mean_jitter > 0 else j
        region_scores.append(max(0.0, normalized_jitter - q))

    if region_scores:
        risk_score = float(np.mean(region_scores))
    else:
        risk_score = 0.0

    torso_jitter = temporal_jitter.get("torso", 0.0)
    torso_quality = spatial_quality.get("torso", 0.0)

    extremity_regions = ["left_arm", "right_arm", "left_leg", "right_leg"]
    extremity_jitters = [temporal_jitter.get(r, 0.0) for r in extremity_regions]
    avg_extremity_jitter = float(np.mean(extremity_jitters)) if extremity_jitters else 0.0

    torso_vs_extremity_ratio = (
        torso_jitter / (avg_extremity_jitter + 1e-8) if avg_extremity_jitter > 0 else 0.0
    )

    if torso_vs_extremity_ratio > 2.0 and torso_quality < mean_spatial:
        risk_score += 0.3

    sibling_ratios: list[float] = []
    num_pts = skeleton_data.shape[1]
    velocity = np.diff(skeleton_data, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)
    mean_speed = speed.mean(axis=0) if speed.shape[0] > 0 else np.zeros(num_pts)

    for left_idx, right_idx in SIBLING_PAIRS:
        if left_idx < num_pts and right_idx < num_pts:
            left_jitter = float(mean_speed[left_idx]) if left_idx < len(mean_speed) else 0.0
            right_jitter = float(mean_speed[right_idx]) if right_idx < len(mean_speed) else 0.0
            if max(left_jitter, right_jitter) > 1e-8:
                ratio = min(left_jitter, right_jitter) / max(left_jitter, right_jitter)
                sibling_ratios.append(ratio)

    if sibling_ratios:
        symmetry = float(np.mean(sibling_ratios))
        if symmetry < 0.3:
            risk_score += 0.2
            suspicious_regions.append(
                ("global", "strong asymmetry between bilateral joints — possible selective texture interference")
            )

    if risk_score < 0.25:
        overall_risk = "low"
    elif risk_score < 0.5:
        overall_risk = "medium"
    else:
        overall_risk = "high"

    return {
        "spatial_quality_map": spatial_quality,
        "temporal_jitter_per_region": temporal_jitter,
        "suspicious_regions": suspicious_regions,
        "overall_texture_risk": overall_risk,
    }
