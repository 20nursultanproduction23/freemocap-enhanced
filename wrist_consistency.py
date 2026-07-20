"""
#6 WRIST CONSISTENCY — Ensure body/hand boundary is smooth

MediaPipe outputs 553 markers: body[0:33], right_hand[33:54],
left_hand[54:75], face[75:553]. The wrist marker from body (index 15/16)
should match the base of the corresponding hand skeleton (index 0).

When they diverge (different triangulation noise), it creates a visible
discontinuity at the wrist. This module detects and corrects such cases.
"""
import numpy as np
from joint_definitions import BODY_IDX, NUM_BODY, NUM_RIGHT_HAND, NUM_LEFT_HAND

BODY_LEFT_WRIST_IDX = BODY_IDX["left_wrist"]
BODY_RIGHT_WRIST_IDX = BODY_IDX["right_wrist"]

RH_BASE_IDX = NUM_BODY
LH_BASE_IDX = NUM_BODY + NUM_RIGHT_HAND


def check_wrist_consistency(skeleton_data, max_mismatch_mm=50.0):
    """Check if body wrist positions match hand base positions.

    MediaPipe hand skeleton has 21 landmarks where landmark[0] is the
    wrist/base. This should match the body's left_wrist/right_wrist.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        max_mismatch_mm: threshold for flagging inconsistency

    Returns:
        diagnostics: dict with per-side mismatch stats
    """
    diagnostics = {}

    num_keypoints = skeleton_data.shape[1]
    has_hands = num_keypoints > NUM_BODY

    for side, body_idx, hand_base_idx in [
        ("right", BODY_RIGHT_WRIST_IDX, RH_BASE_IDX),
        ("left", BODY_LEFT_WRIST_IDX, LH_BASE_IDX),
    ]:
        body_wrist = skeleton_data[:, body_idx, :]
        if not has_hands:
            diagnostics[side] = {
                "mean_mismatch_mm": 0, "max_mismatch_mm": 0,
                "median_mismatch_mm": 0, "flagged_frames": 0,
                "p95_mismatch_mm": 0, "status": "no_hand_data",
            }
            continue
        hand_base = skeleton_data[:, hand_base_idx, :]

        valid = ~np.isnan(body_wrist).any(axis=1) & ~np.isnan(hand_base).any(axis=1)

        if np.sum(valid) == 0:
            diagnostics[side] = {"mean_mismatch": 0, "max_mismatch": 0,
                                 "flagged_frames": 0, "status": "no_valid_data"}
            continue

        mismatch = np.linalg.norm(body_wrist[valid] - hand_base[valid], axis=1)

        flagged = np.where(valid)[0][mismatch > max_mismatch_mm]

        diagnostics[side] = {
            "mean_mismatch_mm": float(np.mean(mismatch)),
            "max_mismatch_mm": float(np.max(mismatch)),
            "median_mismatch_mm": float(np.median(mismatch)),
            "flagged_frames": len(flagged),
            "p95_mismatch_mm": float(np.percentile(mismatch, 95)),
            "status": "ok" if len(flagged) == 0 else "inconsistent",
        }

    return diagnostics


def _estimate_confidence_weight(skeleton_data, body_idx, hand_idx, window=5):
    num_frames = skeleton_data.shape[0]
    body_wrist = skeleton_data[:, body_idx, :]
    hand_base = skeleton_data[:, hand_idx, :]

    body_valid = ~np.isnan(body_wrist).any(axis=1)
    hand_valid = ~np.isnan(hand_base).any(axis=1)

    body_disp = np.full(num_frames, np.nan)
    hand_disp = np.full(num_frames, np.nan)

    body_diff = np.linalg.norm(np.diff(body_wrist, axis=0), axis=1)
    hand_diff = np.linalg.norm(np.diff(hand_base, axis=0), axis=1)

    bv = body_valid[1:] & body_valid[:-1]
    hv = hand_valid[1:] & hand_valid[:-1]

    body_disp[1:] = np.where(bv, body_diff, np.nan)
    hand_disp[1:] = np.where(hv, hand_diff, np.nan)

    body_smooth = np.full(num_frames, np.nan)
    hand_smooth = np.full(num_frames, np.nan)

    for i in range(num_frames):
        start = max(0, i - window + 1)
        body_smooth[i] = np.nanmean(body_disp[start:i + 1])
        hand_smooth[i] = np.nanmean(hand_disp[start:i + 1])

    conf_body = np.ones(num_frames, dtype=np.float64)
    conf_hand = np.ones(num_frames, dtype=np.float64)

    both_smooth = ~np.isnan(body_smooth) & ~np.isnan(hand_smooth)
    if np.any(both_smooth):
        bs = body_smooth[both_smooth]
        hs = hand_smooth[both_smooth]
        total = bs + hs
        safe_total = np.where(total > 0, total, 1.0)
        conf_body[both_smooth] = hs / safe_total
        conf_hand[both_smooth] = bs / safe_total

    only_body = body_valid & ~hand_valid
    only_hand = hand_valid & ~body_valid
    neither = ~body_valid & ~hand_valid

    conf_body[only_body] = 1.0
    conf_hand[only_body] = 0.0
    conf_body[only_hand] = 0.0
    conf_hand[only_hand] = 1.0
    conf_body[neither] = 0.0
    conf_hand[neither] = 0.0

    total = conf_body + conf_hand
    total[total == 0] = 1.0
    conf_body /= total
    conf_hand /= total

    return conf_body, conf_hand


def fix_wrist_consistency(skeleton_data, strategy="confidence_weighted", dissolve_radius=None):
    """Fix wrist/hand base inconsistency with optional confidence-weighted blending.

    When body wrist and hand base diverge, the hand skeleton is re-anchored
    to produce a smooth body-to-hand transition.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        strategy: how to resolve inconsistency:
            - "confidence_weighted" (default): blend positions weighted by
              temporal smoothness; apply dissolving offset so fingertips
              retain independent tracking
            - "body_wins": snap hand base to body wrist
            - "hand_wins": snap body wrist to hand base
            - "average": use midpoint
        dissolve_radius: float or None. Maximum radius over which the
            correction dissolves. If None, auto-computed from mean hand
            landmark spread.

    Returns:
        fixed_data: (numFrames, numTrackedPoints, 3) array
        fix_info: dict with fix statistics
    """
    fixed = skeleton_data.copy()
    fix_info = {"right_fixes": 0, "left_fixes": 0}

    num_keypoints = skeleton_data.shape[1]
    if num_keypoints <= NUM_BODY:
        return fixed, fix_info

    for side, body_idx, hand_base_idx in [
        ("right", BODY_RIGHT_WRIST_IDX, RH_BASE_IDX),
        ("left", BODY_LEFT_WRIST_IDX, LH_BASE_IDX),
    ]:
        body_wrist = skeleton_data[:, body_idx, :]
        hand_base = skeleton_data[:, hand_base_idx, :]

        valid = ~np.isnan(body_wrist).any(axis=1) & ~np.isnan(hand_base).any(axis=1)

        if np.sum(valid) == 0:
            continue

        mismatch = np.linalg.norm(body_wrist[valid] - hand_base[valid], axis=1)

        fix_threshold = np.median(mismatch) + 3 * np.std(mismatch)
        fix_frames = np.where(valid)[0][mismatch > fix_threshold]

        if len(fix_frames) == 0:
            continue

        hand_start = RH_BASE_IDX if side == "right" else LH_BASE_IDX
        hand_count = NUM_RIGHT_HAND if side == "right" else NUM_LEFT_HAND

        conf_body = None
        conf_hand = None
        side_dissolve = dissolve_radius

        if strategy == "confidence_weighted":
            conf_body, conf_hand = _estimate_confidence_weight(
                skeleton_data, body_idx, hand_base_idx, window=5
            )

            if side_dissolve is None:
                hand_all = skeleton_data[:, hand_start:hand_start + hand_count, :]
                hand_valid = ~np.isnan(hand_all).any(axis=2).all(axis=1)
                if np.any(hand_valid):
                    diffs = hand_all - np.where(hand_valid[:, None, None], hand_all[:, 0:1, :], 0.0)
                    dists = np.linalg.norm(diffs, axis=2)
                    side_dissolve = float(np.mean(dists[hand_valid]))
                else:
                    side_dissolve = 1.0
                if side_dissolve == 0:
                    side_dissolve = 1.0

        for frame in fix_frames:
            if strategy == "confidence_weighted":
                cb = conf_body[frame]
                ch = conf_hand[frame]
                total_conf = cb + ch
                if total_conf == 0:
                    continue
                blended = (cb * body_wrist[frame] + ch * hand_base[frame]) / total_conf
                delta = blended - hand_base[frame]
                hand_frame = skeleton_data[frame, hand_start:hand_start + hand_count, :]
                base_pos = hand_frame[0]
                dists = np.linalg.norm(hand_frame - base_pos, axis=1)
                factors = 1.0 / (1.0 + dists / side_dissolve)
                for h_idx in range(hand_count):
                    marker_idx = hand_start + h_idx
                    if not np.isnan(fixed[frame, marker_idx, :]).any():
                        fixed[frame, marker_idx, :] += delta * factors[h_idx]
            else:
                if strategy == "body_wins":
                    new_base = body_wrist[frame]
                elif strategy == "hand_wins":
                    new_base = hand_base[frame]
                else:
                    new_base = (body_wrist[frame] + hand_base[frame]) / 2
                delta = new_base - hand_base[frame]
                for h_idx in range(hand_count):
                    marker_idx = hand_start + h_idx
                    if not np.isnan(fixed[frame, marker_idx, :]).any():
                        fixed[frame, marker_idx, :] += delta

        fix_info[f"{side}_fixes"] = len(fix_frames)

    return fixed, fix_info
