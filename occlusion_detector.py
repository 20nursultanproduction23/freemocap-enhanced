"""
#2 OCCLUSION & OUTLIER DETECTION — Identify and interpolate missing/corrupt data

Before any filtering, we need to identify:
1. NaN gaps (complete detection failures from occlusion)
2. Outlier spikes (jumps from wrong triangulation)
3. Confidence drops (unreliable detections)

This module provides detection and interpolation to prepare data for filtering.
"""
import numpy as np
from scipy import signal
from joint_definitions import BODY_IDX, NUM_BODY


def detect_and_interpolate_gaps(
    skeleton_data,
    max_gap=10,
    outlier_threshold=5.0,
    fps=30.0,
):
    """Detect NaN gaps and outlier spikes, then interpolate.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        max_gap: maximum gap length to interpolate (longer gaps stay as NaN)
        outlier_threshold: number of standard deviations for outlier detection
        fps: frames per second
        
    Returns:
        cleaned_data: (numFrames, numTrackedPoints, 3) array
        diagnostics: dict with detection statistics
    """
    cleaned = skeleton_data.copy()
    diagnostics = {
        "total_nan_before": int(np.sum(np.isnan(skeleton_data))),
        "outliers_detected": 0,
        "gaps_interpolated": 0,
    }
    
    num_frames, num_tracked, num_dims = skeleton_data.shape
    
    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = cleaned[:, tracked_idx, dim]
            
            nan_mask = np.isnan(channel)
            valid_count = np.sum(~nan_mask)
            
            if valid_count < 3:
                continue
            
            valid_indices = np.where(~nan_mask)[0]
            
            gap_lengths = np.diff(valid_indices) - 1
            for gap_idx in range(len(gap_lengths)):
                gap_len = gap_lengths[gap_idx]
                if 0 < gap_len <= max_gap:
                    gap_start = valid_indices[gap_idx] + 1
                    gap_end = valid_indices[gap_idx + 1]
                    x_valid = valid_indices[[gap_idx, gap_idx + 1]]
                    y_valid = channel[x_valid]
                    
                    x_gap = np.arange(gap_start, gap_end)
                    channel[gap_start:gap_end] = np.interp(x_gap, x_valid, y_valid)
                    diagnostics["gaps_interpolated"] += int(gap_len)
            
            diff = np.diff(channel[~nan_mask])
            if len(diff) > 5:
                median_diff = np.median(diff)
                mad = np.median(np.abs(diff - median_diff))
                if mad > 0:
                    outlier_mask_channel = np.abs(diff - median_diff) > outlier_threshold * mad * 1.4826
                    outlier_indices = np.where(outlier_mask_channel)[0]
                    
                    for oi in outlier_indices:
                        valid_idx = np.where(~nan_mask)[0]
                        if oi < len(valid_idx) - 1:
                            idx_before = valid_idx[oi]
                            idx_after = valid_idx[oi + 1]
                            channel[idx_before:idx_after + 1] = np.nan
                            diagnostics["outliers_detected"] += 1
            
            cleaned[:, tracked_idx, dim] = channel
    
    diagnostics["total_nan_after"] = int(np.sum(np.isnan(cleaned)))
    
    return cleaned, diagnostics


def detect_outliers(skeleton_data, outlier_threshold=5.0):
    """Detect outlier spikes and mark them as NaN (without filling gaps).

    Uses median absolute deviation on frame-to-frame derivatives to find
    sudden jumps that indicate tracking failures.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        outlier_threshold: MAD multiplier for outlier detection

    Returns:
        cleaned_data: copy of input with outliers replaced by NaN
        diagnostics: dict with detection statistics
    """
    cleaned = skeleton_data.copy()
    diagnostics = {"outliers_detected": 0}

    num_frames, num_tracked, num_dims = skeleton_data.shape

    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = cleaned[:, tracked_idx, dim]
            nan_mask = np.isnan(channel)
            valid_indices = np.where(~nan_mask)[0]

            if len(valid_indices) < 10:
                continue

            diff = np.diff(channel[valid_indices])
            median_diff = np.median(diff)
            mad = np.median(np.abs(diff - median_diff))
            if mad <= 0:
                continue

            outlier_mask = np.abs(diff - median_diff) > outlier_threshold * mad * 1.4826
            outlier_local_idx = np.where(outlier_mask)[0]

            for oi in outlier_local_idx:
                idx_before = valid_indices[oi]
                idx_after = valid_indices[oi + 1]
                channel[idx_before:idx_after + 1] = np.nan
                diagnostics["outliers_detected"] += 1

            cleaned[:, tracked_idx, dim] = channel

    return cleaned, diagnostics


def compute_detection_confidence(skeleton_data):
    """Compute per-frame confidence based on joint visibility and consistency.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        
    Returns:
        confidence: (numFrames,) array, values 0-1
    """
    num_frames = skeleton_data.shape[0]
    confidence = np.zeros(num_frames)
    
    body_slice = slice(0, NUM_BODY)
    
    for frame_idx in range(num_frames):
        body_frame = skeleton_data[frame_idx, body_slice, :]
        total_joints = body_frame.shape[0]
        valid_joints = np.sum(~np.isnan(body_frame).any(axis=1))
        visibility_score = valid_joints / total_joints
        
        if frame_idx > 0 and frame_idx < num_frames - 1:
            prev = skeleton_data[frame_idx - 1, body_slice, :]
            curr = body_frame
            next_f = skeleton_data[frame_idx + 1, body_slice, :]
            
            valid = (~np.isnan(prev).any(axis=1) & 
                    ~np.isnan(curr).any(axis=1) & 
                    ~np.isnan(next_f).any(axis=1))
            
            if valid.sum() > 3:
                velocity = np.abs(curr[valid] - prev[valid])
                accel = np.abs(next_f[valid] - 2 * curr[valid] + prev[valid])
                
                speed = np.mean(velocity)
                jerk = np.mean(accel)
                
                smoothness_score = 1.0 / (1.0 + speed * 10 + jerk * 5)
            else:
                smoothness_score = 0.5
        else:
            smoothness_score = 0.5
        
        confidence[frame_idx] = 0.6 * visibility_score + 0.4 * smoothness_score
    
    return confidence


def detect_left_right_swaps(skeleton_data, velocity_threshold=None):
    """Detect and fix left/right body part swaps.
    
    MediaPipe sometimes confuses left and right when the person is
    facing away from the camera or in unusual poses.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        velocity_threshold: velocity threshold for swap detection
        
    Returns:
        corrected_data: (numFrames, numTrackedPoints, 3) array
        swap_frames: list of frame indices where swaps were detected
    """
    corrected = skeleton_data.copy()
    swap_frames = []
    
    left_shoulder = skeleton_data[:, BODY_IDX["left_shoulder"], :]
    right_shoulder = skeleton_data[:, BODY_IDX["right_shoulder"], :]
    
    valid = (~np.isnan(left_shoulder).any(axis=1) & 
            ~np.isnan(right_shoulder).any(axis=1))
    
    if np.sum(valid) < 10:
        return corrected, swap_frames
    
    mid_shoulder_x = np.zeros(skeleton_data.shape[0])
    mid_shoulder_x[valid] = (left_shoulder[valid, 0] + right_shoulder[valid, 0]) / 2
    
    swap_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]
    
    for frame_idx in range(1, skeleton_data.shape[0]):
        if not valid[frame_idx]:
            continue
        
        left_x = left_shoulder[frame_idx, 0]
        right_x = right_shoulder[frame_idx, 0]
        
        is_swapped = left_x > right_x
        
        if frame_idx > 1 and valid[frame_idx - 1]:
            prev_left_x = left_shoulder[frame_idx - 1, 0]
            prev_right_x = right_shoulder[frame_idx - 1, 0]
            prev_swapped = prev_left_x > prev_right_x
            
            if is_swapped != prev_swapped:
                shoulder_dist = abs(left_x - right_x)
                prev_shoulder_dist = abs(prev_left_x - prev_right_x)
                
                if shoulder_dist > 0.1 * prev_shoulder_dist:
                    swap_frames.append(frame_idx)
    
    return corrected, swap_frames


def fill_nan_gaps_cubic(skeleton_data, max_gap=30):
    """Fill NaN gaps using cubic spline interpolation.
    
    Better than linear for motion data - preserves velocity continuity.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        max_gap: maximum gap length to fill
        
    Returns:
        filled_data: (numFrames, numTrackedPoints, 3) array
    """
    from scipy.interpolate import CubicSpline
    
    filled = skeleton_data.copy()
    num_frames, num_tracked, num_dims = skeleton_data.shape
    
    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = skeleton_data[:, tracked_idx, dim]
            nan_mask = np.isnan(channel)
            
            if not np.any(nan_mask):
                continue
            
            valid = ~nan_mask
            if np.sum(valid) < 4:
                continue
            
            valid_indices = np.where(valid)[0]
            valid_values = channel[valid]
            
            gap_lengths = np.diff(valid_indices) - 1
            large_gaps = np.where(gap_lengths > max_gap)[0]
            
            if len(large_gaps) > 0:
                boundary_indices = set([0, len(valid_indices) - 1])
                for lg in large_gaps:
                    boundary_indices.add(lg)
                    boundary_indices.add(lg + 1)
                boundary_indices = sorted(boundary_indices)
                valid_indices_subset = valid_indices[boundary_indices]
                valid_values_subset = valid_values[boundary_indices]
            else:
                valid_indices_subset = valid_indices
                valid_values_subset = valid_values
            
            if len(valid_indices_subset) < 4:
                interp_func = lambda x: np.interp(x, valid_indices_subset, valid_values_subset)
            else:
                try:
                    cs = CubicSpline(valid_indices_subset, valid_values_subset, bc_type='natural')
                    interp_func = cs
                except Exception:
                    interp_func = lambda x: np.interp(x, valid_indices_subset, valid_values_subset)
            
            nan_indices = np.where(nan_mask)[0]
            filled[nan_indices, tracked_idx, dim] = interp_func(nan_indices)
    
    return filled
