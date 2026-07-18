"""
#14 FRAME DROP DETECTION — Detect camera frame drops from motion patterns

When USB bandwidth is insufficient, cameras drop frames silently.
FreeMoCap may interpolate over these gaps, hiding the problem.

This module detects frame drops by analyzing:
1. Sudden position jumps (interpolated frames smooth over real gaps)
2. Velocity spikes inconsistent with human motion limits
3. NaN burst patterns correlated across all body joints simultaneously
4. Irregular inter-frame motion intervals (expected: uniform at constant fps)
"""
import numpy as np
from joint_definitions import BODY_IDX, NUM_BODY


HUMAN_MAX_VELOCITY_MM_S = 5000.0


def detect_frame_drops(skeleton_data, fps=30.0, velocity_threshold=None,
                        jump_threshold=100.0):
    """Detect likely frame drops from motion discontinuities.

    Frame drops create characteristic artifacts:
    - After interpolation: smoothed-over gaps with reduced velocity
    - Without interpolation: sudden jumps > physiological limits
    - Correlated NaN bursts across ALL body joints (camera failure)

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate
        velocity_threshold: max physiological velocity in mm/s (default: auto)
        jump_threshold: min frame-to-frame jump in mm to flag as drop

    Returns:
        result: dict with drop detection statistics and flagged frames
    """
    num_frames = skeleton_data.shape[0]
    body_data = skeleton_data[:, :NUM_BODY, :]

    if velocity_threshold is None:
        velocity_threshold = HUMAN_MAX_VELOCITY_MM_S

    frame_valid = np.array([
        np.sum(~np.isnan(body_data[f]).any(axis=1)) >= 10
        for f in range(num_frames)
    ])

    velocity = np.zeros(num_frames)
    max_joint_disp = np.zeros(num_frames)

    for f in range(1, num_frames):
        if not (frame_valid[f] and frame_valid[f - 1]):
            continue
        curr = body_data[f]
        prev = body_data[f - 1]
        valid = ~np.isnan(curr).any(axis=1) & ~np.isnan(prev).any(axis=1)
        if np.sum(valid) < 5:
            continue

        disp = np.linalg.norm(curr[valid] - prev[valid], axis=1)
        velocity[f] = np.mean(disp) * fps
        max_joint_disp[f] = np.max(disp)

    all_nan_frame = np.all(np.isnan(body_data).any(axis=2), axis=1)

    nan_bursts = []
    in_burst = False
    burst_start = 0
    for f in range(num_frames):
        if all_nan_frame[f] and not in_burst:
            in_burst = True
            burst_start = f
        elif not all_nan_frame[f] and in_burst:
            in_burst = False
            nan_bursts.append({"start": burst_start, "end": f - 1,
                               "length": f - burst_start})
    if in_burst:
        nan_bursts.append({"start": burst_start, "end": num_frames - 1,
                           "length": num_frames - burst_start})

    jump_frames = np.where(max_joint_disp > jump_threshold)[0]
    jump_frames = jump_frames[jump_frames > 0]

    velocity_exceed = np.where(
        (velocity > velocity_threshold) & frame_valid
    )[0]

    speed_stats = velocity[velocity > 0]
    if len(speed_stats) > 10:
        median_speed = float(np.median(speed_stats))
        p95_speed = float(np.percentile(speed_stats, 95))
    else:
        median_speed = 0.0
        p95_speed = 0.0

    drop_confidence = 0.0
    drop_indicators = 0
    if len(nan_bursts) >= 2:
        drop_indicators += 1
    if len(jump_frames) > num_frames * 0.01:
        drop_indicators += 1
    if p95_speed > velocity_threshold * 0.8:
        drop_indicators += 1

    if drop_indicators >= 3:
        drop_confidence = 0.9
    elif drop_indicators == 2:
        drop_confidence = 0.6
    elif drop_indicators == 1:
        drop_confidence = 0.3

    return {
        "num_frames": num_frames,
        "fps": fps,
        "total_nan_bursts": len(nan_bursts),
        "nan_bursts": nan_bursts,
        "all_nan_frame_count": int(np.sum(all_nan_frame)),
        "jump_frames_count": len(jump_frames),
        "jump_frames": jump_frames.tolist()[:50],
        "velocity_exceed_count": len(velocity_exceed),
        "velocity_exceed_frames": velocity_exceed.tolist()[:50],
        "median_speed_mm_s": median_speed,
        "p95_speed_mm_s": p95_speed,
        "velocity_threshold_mm_s": velocity_threshold,
        "drop_confidence": drop_confidence,
        "message": _drop_message(drop_confidence, len(nan_bursts),
                                 len(jump_frames), num_frames),
    }


def _drop_message(confidence, burst_count, jump_count, total_frames):
    if confidence >= 0.8:
        return (f"HIGH frame drop confidence ({confidence:.0%}). "
                f"{burst_count} NaN bursts, {jump_count} velocity jumps. "
                f"Recording setup likely has USB bandwidth issues.")
    elif confidence >= 0.5:
        return (f"MODERATE frame drop indicators ({confidence:.0%}). "
                f"Some evidence of frame gaps in recording.")
    elif confidence >= 0.2:
        return (f"LOW frame drop indicators ({confidence:.0%}). "
                f"Minor anomalies detected, likely acceptable.")
    return "No significant frame drop indicators detected."


def _detect_timestamp_unit(timestamps):
    """Auto-detect whether timestamps are in seconds or milliseconds.

    Uses median inter-frame interval: if > 10, assumed milliseconds.
    """
    arr = np.asarray(timestamps, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]
    diffs = np.diff(arr)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return "seconds"
    median_diff = float(np.median(diffs))
    return "milliseconds" if median_diff > 10 else "seconds"


def detect_frame_drops_from_timestamps(timestamps, fps_nominal=30.0,
                                       tolerance_ms=2.0):
    """Detect frame drops from raw camera timestamps (direct measurement).

    Args:
        timestamps: (numFrames,) or (numCameras, numFrames) array.
            Values in seconds or milliseconds (auto-detected).
        fps_nominal: expected frame rate.
        tolerance_ms: extra tolerance in ms beyond ideal interval before
            flagging a gap as a drop.

    Returns:
        dict with keys:
            drop_frames, drop_count, actual_fps, interval_stats, confidence
    """
    arr = np.asarray(timestamps, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]

    unit = _detect_timestamp_unit(arr)
    if unit == "milliseconds":
        intervals_ms = np.diff(arr)
    else:
        intervals_ms = np.diff(arr) * 1000.0

    ideal_interval_ms = 1000.0 / fps_nominal
    threshold_ms = ideal_interval_ms + tolerance_ms

    drop_mask = intervals_ms > threshold_ms
    drop_frames = (np.where(drop_mask)[0] + 1).tolist()

    positive_intervals = intervals_ms[intervals_ms > 0]
    if len(positive_intervals) > 0:
        median_interval = float(np.median(positive_intervals))
        actual_fps = 1000.0 / median_interval if median_interval > 0 else 0.0
        interval_stats = {
            "mean_ms": float(np.mean(positive_intervals)),
            "std_ms": float(np.std(positive_intervals)),
            "min_ms": float(np.min(positive_intervals)),
            "max_ms": float(np.max(positive_intervals)),
            "median_ms": median_interval,
        }
    else:
        actual_fps = fps_nominal
        interval_stats = {
            "mean_ms": 0.0, "std_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0, "median_ms": 0.0,
        }

    return {
        "drop_frames": drop_frames,
        "drop_count": len(drop_frames),
        "actual_fps": actual_fps,
        "interval_stats": interval_stats,
        "confidence": 0.95,
    }


def detect_frame_drops_multi_camera(timestamps_per_camera, fps_nominal=30.0,
                                    tolerance_ms=2.0):
    """Detect frame drops across multiple cameras with correlation analysis.

    Args:
        timestamps_per_camera: list of (numFrames,) arrays, one per camera.
        fps_nominal: expected frame rate.
        tolerance_ms: tolerance for gap detection in ms.

    Returns:
        dict with per_camera_drops, correlated_drops,
        camera_specific_drops, overall_confidence.
    """
    per_camera = []
    for cam_idx, ts in enumerate(timestamps_per_camera):
        result = detect_frame_drops_from_timestamps(
            ts, fps_nominal=fps_nominal, tolerance_ms=tolerance_ms
        )
        result["camera_index"] = cam_idx
        per_camera.append(result)

    if len(per_camera) < 2:
        return {
            "per_camera_drops": per_camera,
            "correlated_drops": [],
            "camera_specific_drops": [],
            "overall_confidence": per_camera[0]["confidence"] if per_camera else 0.0,
        }

    ideal_interval_ms = 1000.0 / fps_nominal
    tolerance_s = tolerance_ms / 1000.0
    sync_window = ideal_interval_ms * 0.5 / 1000.0

    all_drop_frames = [set(cam["drop_frames"]) for cam in per_camera]

    correlated = []
    camera_specific = []

    all_indices = set()
    for s in all_drop_frames:
        all_indices.update(s)

    cam_times_all = []
    for ts in timestamps_per_camera:
        arr = np.asarray(ts, dtype=float)
        unit = _detect_timestamp_unit(arr)
        if unit == "milliseconds":
            arr = arr / 1000.0
        cam_times_all.append(arr)

    for frame_idx in sorted(all_indices):
        cam_count = sum(1 for s in all_drop_frames if frame_idx in s)
        if cam_count >= 2:
            ts_est = None
            for arr in cam_times_all:
                if frame_idx < len(arr):
                    ts_est = float(arr[frame_idx])
                    break
            correlated.append({
                "frame_index": frame_idx,
                "timestamp_approx_s": ts_est,
                "cameras_detecting": cam_count,
                "total_cameras": len(per_camera),
            })
        elif cam_count == 1:
            cam_id = next(
                i for i, s in enumerate(all_drop_frames) if frame_idx in s
            )
            camera_specific.append({
                "frame_index": frame_idx,
                "camera_index": cam_id,
            })

    total_drops = sum(c["drop_count"] for c in per_camera)
    if len(correlated) > 0:
        overall_confidence = min(
            0.98,
            0.90 + 0.02 * min(len(correlated), 4)
        )
    elif total_drops > 0:
        overall_confidence = 0.70
    else:
        overall_confidence = 0.95

    return {
        "per_camera_drops": per_camera,
        "correlated_drops": correlated,
        "camera_specific_drops": camera_specific,
        "overall_confidence": overall_confidence,
    }


def fuse_detection(skeleton_drops, timestamp_drops=None):
    """Combine indirect (motion-based) and direct (timestamp-based) detection.

    Args:
        skeleton_drops: dict returned by detect_frame_drops().
        timestamp_drops: dict returned by detect_frame_drops_from_timestamps()
            or detect_frame_drops_multi_camera(). Optional.

    Returns:
        unified result dict with fused confidence and message.
    """
    skel_conf = skeleton_drops.get("drop_confidence", 0.0)
    skel_jumps = skeleton_drops.get("jump_frames_count", 0)
    skel_nan = skeleton_drops.get("total_nan_bursts", 0)

    if timestamp_drops is None:
        fused_confidence = skel_conf
        source = "motion_only"
    else:
        ts_conf = timestamp_drops.get(
            "overall_confidence",
            timestamp_drops.get("confidence", 0.0),
        )
        ts_drops = timestamp_drops.get(
            "drop_count",
            len(timestamp_drops.get("correlated_drops", []))
            if "correlated_drops" in timestamp_drops
            else 0,
        )

        if ts_drops > 0 and skel_conf > 0.3:
            fused_confidence = min(
                0.99, max(ts_conf, skel_conf) + 0.03
            )
            source = "both_agree"
        elif ts_drops > 0:
            fused_confidence = ts_conf
            source = "timestamps_only"
        elif skel_conf > 0.3:
            fused_confidence = skel_conf * 0.7
            source = "motion_only_possible_artifact"
        else:
            fused_confidence = max(ts_conf, skel_conf)
            source = "neither"

    if fused_confidence >= 0.9:
        msg = (f"HIGH fused confidence ({fused_confidence:.0%}). "
               f"Direct timestamp and/or motion evidence confirms frame drops.")
    elif fused_confidence >= 0.6:
        msg = (f"MODERATE fused confidence ({fused_confidence:.0%}). "
               f"Frame drops likely; verify with raw timestamp data.")
    elif fused_confidence >= 0.3:
        msg = (f"LOW fused confidence ({fused_confidence:.0%}). "
               f"Possible minor frame gaps.")
    else:
        msg = "No significant frame drop evidence from either method."

    return {
        "fused_confidence": fused_confidence,
        "source": source,
        "message": msg,
        "skeleton_confidence": skel_conf,
        "timestamp_confidence": (
            timestamp_drops.get(
                "overall_confidence",
                timestamp_drops.get("confidence", 0.0),
            )
            if timestamp_drops
            else None
        ),
        "skeleton_jumps": skel_jumps,
        "skeleton_nan_bursts": skel_nan,
        "timestamp_drop_count": (
            timestamp_drops.get("drop_count", 0) if timestamp_drops else None
        ),
    }
