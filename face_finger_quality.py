"""
#13 FACE/FINGER QUALITY — Analyze and improve face mesh and hand tracking

MediaPipe's face mesh (478 points) and hand landmarks (21 each) are often
noisy or incomplete, especially under occlusion or motion blur.

This module provides:
1. Quality metrics for face mesh and hand tracking
2. Detection of face mesh collapse (flat/collapsed mesh)
3. Hand tracking jitter measurement
4. Quality-aware filtering (more aggressive on low-quality regions)
"""
import numpy as np
from joint_definitions import (
    BODY_IDX, NUM_BODY, NUM_RIGHT_HAND, NUM_LEFT_HAND, NUM_FACE, NUM_TOTAL,
)


def compute_hand_quality(skeleton_data, fps=30.0):
    """Compute quality metrics for hand tracking.

    Analyzes wrist→hand boundary consistency, temporal smoothness,
    and spatial coherence of hand landmarks.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        result: dict with per-hand quality metrics
    """
    num_frames = skeleton_data.shape[0]
    results = {}

    hand_slices = {
        "right": slice(NUM_BODY, NUM_BODY + NUM_RIGHT_HAND),
        "left": slice(NUM_BODY + NUM_RIGHT_HAND,
                      NUM_BODY + NUM_RIGHT_HAND + NUM_LEFT_HAND),
    }

    wrist_indices = {
        "right": BODY_IDX["right_wrist"],
        "left": BODY_IDX["left_wrist"],
    }

    for side, hand_slice in hand_slices.items():
        hand_data = skeleton_data[:, hand_slice, :]
        wrist_data = skeleton_data[:, wrist_indices[side], :]

        hand_valid = ~np.isnan(hand_data).any(axis=2)
        hand_valid_count = hand_valid.sum(axis=1)
        coverage = float(np.mean(hand_valid_count) / hand_data.shape[1] * 100)

        wrist_valid = ~np.isnan(wrist_data).any(axis=1)

        boundary_errors = []
        for f in range(num_frames):
            if not (wrist_valid[f] and np.any(hand_valid[f])):
                continue
            wrist_pos = wrist_data[f]
            hand_positions = hand_data[f]
            valid_hand = hand_positions[hand_valid[f]]
            if len(valid_hand) == 0:
                continue

            wrist_to_hand = np.linalg.norm(valid_hand - wrist_pos, axis=1)
            mean_dist = np.mean(wrist_to_hand)

            hand_center = np.mean(valid_hand, axis=0)
            hand_spread = np.mean(np.linalg.norm(valid_hand - hand_center, axis=1))

            if mean_dist > 0 and hand_spread > 0:
                ratio = hand_spread / mean_dist
                if ratio < 0.3 or ratio > 3.0:
                    boundary_errors.append(f)

        temporal_jitter = _compute_hand_temporal_jitter(hand_data, fps)

        mean_jitter = float(np.mean(temporal_jitter)) if len(temporal_jitter) > 0 else 0.0
        p95_jitter = float(np.percentile(temporal_jitter, 95)) if len(temporal_jitter) > 0 else 0.0

        quality_score = 100.0
        if coverage < 50:
            quality_score -= (50 - coverage) * 0.5
        if mean_jitter > 20:
            quality_score -= (mean_jitter - 20) * 0.3
        if len(boundary_errors) > num_frames * 0.1:
            quality_score -= 15
        quality_score = max(0, quality_score)

        severity = "good"
        if quality_score < 60:
            severity = "poor"
        elif quality_score < 80:
            severity = "moderate"

        results[side] = {
            "coverage_pct": coverage,
            "boundary_error_frames": len(boundary_errors),
            "boundary_error_rate": float(len(boundary_errors) / num_frames * 100),
            "temporal_jitter_mean_mm": mean_jitter,
            "temporal_jitter_p95_mm": p95_jitter,
            "quality_score": quality_score,
            "severity": severity,
        }

    return results


def compute_face_quality(skeleton_data, fps=30.0):
    """Compute quality metrics for face mesh tracking.

    Analyzes face mesh density, temporal smoothness, and geometric
    consistency (mesh shouldn't flatten or collapse).

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate

    Returns:
        result: dict with face quality metrics
    """
    num_frames = skeleton_data.shape[0]
    face_start = NUM_BODY + NUM_RIGHT_HAND + NUM_LEFT_HAND
    face_end = face_start + NUM_FACE
    face_data = skeleton_data[:, face_start:face_end, :]

    face_valid = ~np.isnan(face_data).any(axis=2)
    face_valid_count = face_valid.sum(axis=1)
    coverage = float(np.mean(face_valid_count) / NUM_FACE * 100)

    depth_spread = np.zeros(num_frames)
    for f in range(num_frames):
        valid_face = face_data[f][face_valid[f]]
        if len(valid_face) > 10:
            z_range = np.ptp(valid_face[:, 2])
            xy_range = np.ptp(valid_face[:, :2])
            depth_spread[f] = z_range / max(xy_range, 1e-6)

    valid_spread = depth_spread[depth_spread > 0]
    mean_depth_ratio = float(np.mean(valid_spread)) if len(valid_spread) > 0 else 0.0
    collapse_frames = int(np.sum(valid_spread < 0.05)) if len(valid_spread) > 0 else 0

    nose_idx = BODY_IDX["nose"]
    with np.errstate(invalid='ignore'):
        face_center = np.nanmean(face_data, axis=1)
    nose_pos = skeleton_data[:, nose_idx, :]
    valid_nose = ~np.isnan(nose_pos).any(axis=1) & ~np.isnan(face_center).any(axis=1)

    face_nose_dist = np.zeros(num_frames)
    if np.any(valid_nose):
        face_nose_dist[valid_nose] = np.linalg.norm(
            face_center[valid_nose] - nose_pos[valid_nose], axis=1
        )

    valid_dist = face_nose_dist[valid_nose] if np.any(valid_nose) else np.array([0])
    nose_consistency = float(np.std(valid_dist)) if len(valid_dist) > 5 else 0.0

    temporal_jitter = np.zeros(num_frames)
    for f in range(1, num_frames):
        if face_valid[f].sum() > 100 and face_valid[f - 1].sum() > 100:
            valid_both = face_valid[f] & face_valid[f - 1]
            if valid_both.sum() > 50:
                diff = face_data[f, valid_both] - face_data[f - 1, valid_both]
                temporal_jitter[f] = float(np.mean(np.linalg.norm(diff, axis=1)))

    valid_jitter = temporal_jitter[temporal_jitter > 0]
    mean_jitter = float(np.mean(valid_jitter)) if len(valid_jitter) > 0 else 0.0

    quality_score = 100.0
    if coverage < 50:
        quality_score -= (50 - coverage) * 0.5
    if collapse_frames > 0:
        quality_score -= min(collapse_frames * 2, 30)
    if nose_consistency > 30:
        quality_score -= min((nose_consistency - 30) * 0.5, 20)
    if mean_jitter > 15:
        quality_score -= min((mean_jitter - 15) * 0.5, 15)
    quality_score = max(0, quality_score)

    severity = "good"
    if quality_score < 60:
        severity = "poor"
    elif quality_score < 80:
        severity = "moderate"

    return {
        "coverage_pct": coverage,
        "valid_frames": int(face_valid_count.sum()),
        "total_face_points": NUM_FACE,
        "mean_depth_ratio": mean_depth_ratio,
        "collapse_frames": collapse_frames,
        "nose_consistency_mm": nose_consistency,
        "temporal_jitter_mean_mm": mean_jitter,
        "quality_score": quality_score,
        "severity": severity,
        "message": _face_message(severity, coverage, collapse_frames,
                                  mean_jitter),
    }


def _compute_hand_temporal_jitter(hand_data, fps):
    num_frames = hand_data.shape[0]
    jitter = []
    for f in range(1, num_frames):
        curr_valid = ~np.isnan(hand_data[f]).any(axis=1)
        prev_valid = ~np.isnan(hand_data[f - 1]).any(axis=1)
        both_valid = curr_valid & prev_valid
        if np.sum(both_valid) > 3:
            diff = hand_data[f, both_valid] - hand_data[f - 1, both_valid]
            jitter.append(float(np.mean(np.linalg.norm(diff, axis=1))))
    return jitter


def _face_message(severity, coverage, collapse_frames, jitter):
    if severity == "poor":
        return (f"POOR face quality: {coverage:.0f}% coverage, "
                f"{collapse_frames} collapse frames, "
                f"{jitter:.1f}mm jitter. Consider alternative face tracker.")
    elif severity == "moderate":
        return (f"MODERATE face quality: {coverage:.0f}% coverage. "
                f"Some noise but usable for basic applications.")
    return f"GOOD face quality: {coverage:.0f}% coverage, stable mesh."
