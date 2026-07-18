"""
#1 JITTER FILTER — One Euro Filter + Butterworth combination

The One Euro Filter is an adaptive low-pass filter designed for noisy signals.
- At low speeds (slow movement): strong filtering, removes jitter
- At high speeds (fast movement): weak filtering, preserves motion

This is superior to plain Butterworth for motion capture because it adapts
to the signal dynamics rather than uniformly cutting frequencies.
"""
import numpy as np
from OneEuroFilter import OneEuroFilter as _OneEuroFilter
from scipy import signal


def apply_one_euro_filter(
    skeleton_data,
    fps=30.0,
    min_cutoff=1.0,
    beta=0.007,
    dcutoff=1.0,
    z_cutoff_ratio=None,
):
    """Apply One Euro Filter to all landmarks across all frames.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: frames per second
        min_cutoff: minimum cutoff frequency for XY (lower = more smoothing)
        beta: speed coefficient (higher = less lag for fast movements)
        dcutoff: cutoff frequency for the derivative
        z_cutoff_ratio: if set, Z axis cutoff = min_cutoff * z_cutoff_ratio
            (values > 1.0 = less smoothing on Z axis, < 1.0 = more smoothing)
            Default None uses same cutoff for all axes.
        
    Returns:
        filtered_data: (numFrames, numTrackedPoints, 3) array
    """
    num_frames, num_tracked, num_dims = skeleton_data.shape
    filtered = np.empty_like(skeleton_data)
    
    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = skeleton_data[:, tracked_idx, dim]
            
            if np.all(np.isnan(channel)):
                filtered[:, tracked_idx, dim] = channel
                continue
            
            dim_min_cutoff = min_cutoff
            if z_cutoff_ratio is not None and dim == 2:
                dim_min_cutoff = min_cutoff * z_cutoff_ratio
            
            oef = _OneEuroFilter(
                freq=fps,
                mincutoff=dim_min_cutoff,
                beta=beta,
                dcutoff=dcutoff,
            )
            
            for frame_idx in range(num_frames):
                if np.isnan(channel[frame_idx]):
                    filtered[frame_idx, tracked_idx, dim] = np.nan
                else:
                    filtered[frame_idx, tracked_idx, dim] = oef(channel[frame_idx], frame_idx / fps)
    
    return filtered


def apply_butterworth_filter(
    skeleton_data,
    cutoff=6.0,
    order=4,
    fps=30.0,
    z_cutoff_ratio=None,
):
    """Apply zero-phase Butterworth low-pass filter (same as FreeMoCap's default).
    
    Uses scipy.signal.filtfilt for zero-phase filtering (no temporal lag).
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        cutoff: cutoff frequency in Hz for XY
        order: filter order
        fps: sampling rate in Hz
        z_cutoff_ratio: if set, Z axis cutoff = cutoff * z_cutoff_ratio
            (values > 1.0 = less smoothing on Z axis)
        
    Returns:
        filtered_data: (numFrames, numTrackedPoints, 3) array
    """
    num_frames, num_tracked, num_dims = skeleton_data.shape
    nyquist = 0.5 * fps
    
    filtered = np.empty_like(skeleton_data)
    
    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = skeleton_data[:, tracked_idx, dim]
            
            if np.all(np.isnan(channel)):
                filtered[:, tracked_idx, dim] = channel
                continue
            
            dim_cutoff = cutoff
            if z_cutoff_ratio is not None and dim == 2:
                dim_cutoff = cutoff * z_cutoff_ratio
            
            normal_cutoff = dim_cutoff / nyquist
            normal_cutoff = min(normal_cutoff, 0.99)
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            
            nan_mask = np.isnan(channel)
            if np.any(nan_mask):
                valid = ~nan_mask
                channel_interp = channel.copy()
                channel_interp[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(valid)[0],
                    channel[valid]
                )
            else:
                channel_interp = channel
            
            if num_frames > 35:
                filtered[:, tracked_idx, dim] = signal.filtfilt(b, a, channel_interp)
            else:
                filtered[:, tracked_idx, dim] = channel_interp
            
            filtered[nan_mask, tracked_idx, dim] = np.nan
    
    return filtered


def apply_combined_filter(
    skeleton_data,
    fps=30.0,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.007,
    butterworth_cutoff=6.0,
    butterworth_order=4,
    z_cutoff_ratio=None,
):
    """Apply One Euro Filter followed by Butterworth for maximum jitter removal.
    
    Pipeline:
    1. One Euro Filter (adaptive, removes most jitter while preserving motion)
    2. Butterworth (uniform cleanup of remaining high-frequency noise)
    
    This combination gives better results than either filter alone:
    - One Euro preserves fast intentional movements
    - Butterworth provides uniform spectral cleanup
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: frames per second
        one_euro_min_cutoff: One Euro min cutoff (lower = more smoothing)
        one_euro_beta: One Euro speed coefficient
        butterworth_cutoff: Butterworth cutoff frequency in Hz
        butterworth_order: Butterworth filter order
        z_cutoff_ratio: if set, Z axis cutoff = XY_cutoff * z_cutoff_ratio
            (values > 1.0 = less smoothing on Z, < 1.0 = more smoothing on Z)
        
    Returns:
        filtered_data: (numFrames, numTrackedPoints, 3) array
    """
    print("  [1/2] Applying One Euro Filter (adaptive jitter removal)...")
    step1 = apply_one_euro_filter(
        skeleton_data,
        fps=fps,
        min_cutoff=one_euro_min_cutoff,
        beta=one_euro_beta,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    
    print("  [2/2] Applying Butterworth low-pass filter (spectral cleanup)...")
    step2 = apply_butterworth_filter(
        step1,
        cutoff=butterworth_cutoff,
        order=butterworth_order,
        fps=fps,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    
    return step2
