"""
#5 FOOT SLIDING — Foot contact detection + IK pinning

When a foot is stationary on the ground, noise makes it appear to slide.
This module detects foot-ground contacts and pins foot positions
to a fixed point during contact, with smooth blend in/out.

Algorithm:
1. Compute foot velocity (frame-to-frame displacement)
2. Detect contact when velocity falls below threshold
3. Optionally use ground plane to improve contact detection
   (foot is close to ground AND velocity is low)
4. Pin foot position to average position during contact
5. Smooth blend on contact start/end to avoid pops
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from joint_definitions import (
    BODY_IDX, LEFT_FOOT_MARKERS, RIGHT_FOOT_MARKERS, NUM_BODY,
)


def detect_and_fix_foot_sliding(
    skeleton_data,
    velocity_threshold=None,
    min_contact_frames=3,
    fps=30.0,
    blend_frames=5,
    ground_normal=None,
    ground_point=None,
):
    """Detect foot contacts and pin foot positions to eliminate sliding.
    
    Uses per-segment pinning: each contact segment is pinned to its own
    local average position, not a global average across all contacts.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        velocity_threshold: velocity below which foot is considered planted.
            If None, auto-calibrated using percentile of speed distribution.
        min_contact_frames: minimum consecutive frames for a valid contact
        fps: frames per second
        blend_frames: number of frames for smooth blend in/out
        ground_normal: (3,) ground plane normal (optional, for height-based contact)
        ground_point: (3,) point on ground plane (optional)
        
    Returns:
        corrected_data: (numFrames, numTrackedPoints, 3) array
    """
    corrected = skeleton_data.copy()
    
    for foot_name, foot_markers in [("left", LEFT_FOOT_MARKERS), ("right", RIGHT_FOOT_MARKERS)]:
        print(f"  Processing {foot_name} foot...")
        
        foot_indices = [BODY_IDX[m] for m in foot_markers]
        
        foot_positions = skeleton_data[:, foot_indices, :]  # (numFrames, 3, 3)
        foot_center = np.nanmean(foot_positions, axis=1)    # (numFrames, 3)
        
        velocities = _compute_velocity(foot_center, fps)
        speed = np.sqrt(np.sum(velocities ** 2, axis=1))   # (numFrames,)
        
        if velocity_threshold is None:
            valid_speeds = speed[~np.isnan(speed)]
            if len(valid_speeds) > 0:
                threshold = np.percentile(valid_speeds, 25)
                threshold = max(threshold, 0.5)
            else:
                threshold = 1.0
        else:
            threshold = velocity_threshold
        
        contact_mask = speed < threshold
        contact_mask = _filter_contact_segments(contact_mask, min_contact_frames)

        # Refine with ground plane if available
        if ground_normal is not None and ground_point is not None:
            height_mask = _compute_ground_height_mask(
                foot_center, ground_normal, ground_point, max_height_mm=30.0
            )
            contact_mask = contact_mask & height_mask
            contact_mask = _filter_contact_segments(contact_mask, min_contact_frames)
        
        num_contact = np.sum(contact_mask)
        print(f"    Contact frames: {num_contact}/{len(contact_mask)} (threshold: {threshold:.2f})")
        
        if num_contact == 0:
            continue
        
        corrected_center = foot_center.copy()
        
        segments = _find_contact_segments(contact_mask)
        
        for seg_start, seg_end in segments:
            seg_positions = foot_center[seg_start:seg_end]
            seg_center = np.nanmean(seg_positions, axis=0)
            
            actual_blend_start = min(blend_frames, (seg_end - seg_start) // 2)
            actual_blend_end = min(blend_frames, (seg_end - seg_start) // 2)
            
            for i in range(actual_blend_start):
                frame_idx = seg_start + i
                alpha = (i + 1) / (actual_blend_start + 1)
                corrected_center[frame_idx] = (
                    foot_center[frame_idx] * (1 - alpha) + seg_center * alpha
                )
            
            corrected_center[seg_start + actual_blend_start:seg_end - actual_blend_end] = seg_center
            
            for i in range(actual_blend_end):
                frame_idx = seg_end - 1 - i
                if frame_idx < seg_start:
                    break
                alpha = (i + 1) / (actual_blend_end + 1)
                corrected_center[frame_idx] = (
                    foot_center[frame_idx] * (1 - alpha) + seg_center * alpha
                )
        
        offset = corrected_center - foot_center
        for marker_idx in foot_indices:
            mask = ~np.isnan(corrected[:, marker_idx, 0])
            corrected[mask, marker_idx, :] += offset[mask]
    
    return corrected


def _compute_velocity(positions, fps):
    """Compute velocity from position data using central differences.
    
    Args:
        positions: (numFrames, 3) array
        fps: frames per second
        
    Returns:
        velocity: (numFrames, 3) array
    """
    num_frames = len(positions)
    velocity = np.zeros_like(positions)
    dt = 1.0 / fps
    
    for i in range(num_frames):
        if i == 0:
            if num_frames > 1:
                velocity[i] = (positions[1] - positions[0]) / dt
            else:
                velocity[i] = np.zeros(3)
        elif i == num_frames - 1:
            velocity[i] = (positions[-1] - positions[-2]) / dt
        else:
            velocity[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    
    nan_mask = np.isnan(positions).any(axis=1)
    velocity[nan_mask] = np.nan
    
    return velocity


def _filter_contact_segments(contact_mask, min_frames):
    """Remove short contact segments (noise spikes)."""
    result = contact_mask.copy()
    
    in_segment = False
    segment_start = 0
    
    for i in range(len(contact_mask)):
        if contact_mask[i] and not in_segment:
            in_segment = True
            segment_start = i
        elif not contact_mask[i] and in_segment:
            in_segment = False
            if i - segment_start < min_frames:
                result[segment_start:i] = False
    
    if in_segment and len(contact_mask) - segment_start < min_frames:
        result[segment_start:] = False
    
    return result


def _find_contact_segments(contact_mask):
    """Find start/end indices of contact segments.
    
    Returns:
        list of (start, end) tuples (end is exclusive)
    """
    segments = []
    in_segment = False
    segment_start = 0
    
    for i in range(len(contact_mask)):
        if contact_mask[i] and not in_segment:
            in_segment = True
            segment_start = i
        elif not contact_mask[i] and in_segment:
            in_segment = False
            segments.append((segment_start, i))
    
    if in_segment:
        segments.append((segment_start, len(contact_mask)))
    
    return segments


def _compute_ground_height_mask(foot_center, ground_normal, ground_point, max_height_mm=30.0):
    """Create a mask of frames where foot is close to the ground plane.
    
    Distance from point to plane: |n . (p - p0)| / |n|
    
    Args:
        foot_center: (numFrames, 3) foot center positions
        ground_normal: (3,) ground plane normal
        ground_point: (3,) point on ground plane
        max_height_mm: maximum distance from ground to count as "near ground"
        
    Returns:
        mask: (numFrames,) boolean, True where foot is near ground
    """
    n = ground_normal / np.linalg.norm(ground_normal)
    p0 = ground_point
    
    diff = foot_center - p0
    distances = np.abs(np.dot(diff, n))
    
    mask = distances < max_height_mm
    
    # Also mark NaN frames as not near ground
    nan_mask = np.isnan(foot_center).any(axis=1)
    mask[nan_mask] = False
    
    return mask
