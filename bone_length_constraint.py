"""
#3 BREATHING BONES — Enforce rigid bone length constraints

MediaPipe triangulates each joint independently, so bone lengths fluctuate
frame to frame (the "breathing bones" artifact). 

Uses forward-kinematics BFS from a root joint: walk the skeleton tree
from root to leaves, placing each child at exactly the target distance
from its already-corrected parent. This gives EXACT bone lengths in a
single pass with no iteration or oscillation.

Only enforces truly rigid bones (limb segments, heels). Non-rigid bones
(cross-body distances, foot flexion) are left untouched.
"""
import numpy as np
from collections import deque
from joint_definitions import BODY_IDX, NUM_BODY, BODY_LANDMARK_NAMES

RIGID_BONE_NAMES = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

SKELETON_TREE = {
    "left_hip": ["left_knee"],
    "right_hip": ["right_knee"],
    "left_knee": ["left_ankle"],
    "right_knee": ["right_ankle"],
    "left_shoulder": ["left_elbow"],
    "right_shoulder": ["right_elbow"],
    "left_elbow": ["left_wrist"],
    "right_elbow": ["right_wrist"],
}


def _compute_target_lengths(skeleton_data):
    """Compute median bone lengths for rigid bones across valid frames."""
    target_lengths = {}
    for parent_name, child_name in RIGID_BONE_NAMES:
        pi = BODY_IDX[parent_name]
        ci = BODY_IDX[child_name]
        diffs = skeleton_data[:, pi, :] - skeleton_data[:, ci, :]
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        valid = ~np.isnan(lengths) & (lengths > 1e-8)
        if np.any(valid):
            target_lengths[(parent_name, child_name)] = np.median(lengths[valid])
    return target_lengths


def detect_bone_length_outliers(skeleton_data, max_relative_error=0.60):
    """Mark frames with extreme bone length errors as NaN.

    When a bone length deviates more than max_relative_error from the median,
    it indicates a tracking failure. Setting those joints to NaN lets the
    gap filling step interpolate reasonable positions instead of keeping
    wildly incorrect data.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        max_relative_error: bones with error above this fraction are
            marked NaN (default 0.60 = 60%)

    Returns:
        cleaned: copy of input with outlier joints set to NaN
        diagnostics: dict with statistics
    """
    cleaned = skeleton_data.copy()
    target_lengths = _compute_target_lengths(skeleton_data)
    diagnostics = {"frames_marked_nan": 0, "bones_flagged": {}}

    for parent_name, child_name in RIGID_BONE_NAMES:
        pi = BODY_IDX[parent_name]
        ci = BODY_IDX[child_name]
        target = target_lengths.get((parent_name, child_name))
        if target is None or target < 1e-8:
            continue

        diffs = cleaned[:, pi, :] - cleaned[:, ci, :]
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        valid = ~np.isnan(lengths) & (lengths > 1e-8)

        errors = np.full(len(lengths), np.nan)
        errors[valid] = np.abs(lengths[valid] - target) / target

        outlier_mask = errors > max_relative_error
        n_outliers = int(np.sum(outlier_mask))
        diagnostics["bones_flagged"][f"{parent_name}->{child_name}"] = n_outliers

        if n_outliers > 0:
            outlier_frames = np.where(outlier_mask)[0]
            for f in outlier_frames:
                cleaned[f, pi, :] = np.nan
                cleaned[f, ci, :] = np.nan
            diagnostics["frames_marked_nan"] += n_outliers

    return cleaned, diagnostics


def enforce_rigid_bones_bfs(
    skeleton_data,
    num_passes=1,
    max_correction_fraction=0.30,
    correction_strength=0.2,
):
    """Enforce rigid bone lengths using forward-kinematics BFS.
    
    For each frame:
    1. Start from root joints (hips, shoulders) — use original positions
    2. Walk tree outward, placing each child at EXACT target distance
       from its (already corrected) parent
    3. Only correct bones where relative error < max_correction_fraction
    
    This gives exact bone lengths in a single pass with no oscillation.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        num_passes: unused (kept for API compat)
        max_correction_fraction: skip bones where relative error exceeds
            this (indicates occlusion/tracking failure)
        correction_strength: unused (kept for API compat)
        
    Returns:
        constrained_data: (numFrames, numTrackedPoints, 3) array
    """
    constrained = skeleton_data.copy()
    num_frames = skeleton_data.shape[0]
    
    target_lengths = _compute_target_lengths(skeleton_data)
    
    print(f"  Bone IK (FK-BFS): {len(target_lengths)} rigid bones, "
          f"max_correction={max_correction_fraction*100:.0f}%...")
    
    roots = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    
    non_rigid_children = {
        "left_ankle": ["left_heel", "left_foot_index"],
        "right_ankle": ["right_heel", "right_foot_index"],
    }
    
    corrected_count = 0
    skipped_count = 0
    
    for frame_idx in range(num_frames):
        original_positions = {}
        for name in BODY_IDX:
            idx = BODY_IDX[name]
            original_positions[name] = constrained[frame_idx, idx, :].copy()
        
        positions = {name: original_positions[name].copy() for name in BODY_IDX}
        
        valid = {name: not np.any(np.isnan(positions[name])) for name in BODY_IDX}
        
        if sum(valid.values()) < 4:
            continue
        
        visited = set()
        queue = deque()
        
        for root in roots:
            if valid[root]:
                queue.append(root)
                visited.add(root)
        
        while queue:
            parent_name = queue.popleft()
            parent_pos = positions[parent_name]
            
            for child_name in SKELETON_TREE.get(parent_name, []):
                if child_name in visited:
                    continue
                visited.add(child_name)
                
                if not valid[child_name]:
                    queue.append(child_name)
                    continue
                
                target_key = (parent_name, child_name)
                target_len = target_lengths.get(target_key)
                if target_len is None or target_len < 1e-8:
                    queue.append(child_name)
                    continue
                
                child_pos = positions[child_name]
                vec = child_pos - parent_pos
                current_len = np.linalg.norm(vec)
                
                if current_len < 1e-8:
                    queue.append(child_name)
                    continue
                
                relative_error = abs(current_len - target_len) / target_len
                if relative_error > max_correction_fraction:
                    skipped_count += 1
                    queue.append(child_name)
                    continue
                
                direction = vec / current_len
                new_child_pos = parent_pos + direction * target_len
                positions[child_name] = new_child_pos
                corrected_count += 1
                
                if child_name in non_rigid_children:
                    offset = new_child_pos - child_pos
                    for nr_child in non_rigid_children[child_name]:
                        if valid[nr_child]:
                            positions[nr_child] = original_positions[nr_child] + offset
                
                queue.append(child_name)
        
        for name in BODY_IDX:
            idx = BODY_IDX[name]
            constrained[frame_idx, idx, :] = positions[name]
    
    print(f"    Corrected {corrected_count} bones, skipped {skipped_count}")
    
    return constrained


def correct_foot_gaps_after_interpolation(skeleton_data, original_nan_mask=None):
    """After gap interpolation, re-anchor heel/foot_index relative to ankle.

    When long NaN gaps are linearly interpolated independently for each
    marker, ankle->heel and ankle->foot_index distances may not match the
    original bone lengths. This function:
    1. Computes median offset vectors from ankle to heel/foot_index
       using only valid (non-NaN, non-interpolated) frames
    2. For ALL frames where ankle is valid, re-places heel/foot_index
       at the median offset from ankle

    This guarantees consistent foot bone distances across all frames.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        original_nan_mask: (numFrames, numTrackedPoints) boolean array
            indicating which frames were NaN before gap filling.
            If provided, only re-anchor frames that were originally NaN.
            If None, re-anchor ALL frames.

    Returns:
        corrected: (numFrames, numTrackedPoints, 3) array
        info: dict with correction statistics
    """
    corrected = skeleton_data.copy()
    num_frames = skeleton_data.shape[0]
    info = {"left_heel_fixes": 0, "left_toe_fixes": 0,
            "right_heel_fixes": 0, "right_toe_fixes": 0}

    foot_pairs = [
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_foot_index"),
    ]

    for ankle_name, child_name in foot_pairs:
        ankle_idx = BODY_IDX[ankle_name]
        child_idx = BODY_IDX[child_name]

        ankle_pos = skeleton_data[:, ankle_idx, :]
        child_pos = skeleton_data[:, child_idx, :]

        valid = (~np.isnan(ankle_pos).any(axis=1) &
                 ~np.isnan(child_pos).any(axis=1))

        if np.sum(valid) < 10:
            continue

        offsets = child_pos[valid] - ankle_pos[valid]
        median_offset = np.median(offsets, axis=0)
        distances = np.linalg.norm(offsets, axis=1)
        median_dist = np.median(distances)

        if median_dist < 1e-8:
            continue

        ankle_valid = ~np.isnan(ankle_pos).any(axis=1)

        for frame_idx in range(num_frames):
            if not ankle_valid[frame_idx]:
                continue
            if np.isnan(child_pos[frame_idx]).any():
                continue

            if original_nan_mask is not None:
                was_nan = original_nan_mask[frame_idx, child_idx]
                if not was_nan:
                    continue

            corrected[frame_idx, child_idx, :] = (
                ankle_pos[frame_idx] + median_offset
            )
            key = "left" if "left" in ankle_name else "right"
            child_suffix = "heel" if "heel" in child_name else "toe"
            fix_key = "%s_%s_fixes" % (key, child_suffix)
            info[fix_key] = info.get(fix_key, 0) + 1

    total_fixes = sum(v for k, v in info.items() if k.endswith("_fixes"))
    if total_fixes > 0:
        print("    Foot gap corrections: %s (total: %d)" % (info, total_fixes))

    return corrected, info
