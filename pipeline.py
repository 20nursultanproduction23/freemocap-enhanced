"""
FreeMoCap Enhanced Post-Processing Pipeline

Combines all improvements:
  #1 OneEuroFilter + Butterworth for jitter removal (with Z-weighted filtering)
  #2 Gap detection + linear interpolation for occlusions
  #3 Rigid bone length enforcement (IK) + foot gap correction
  #5 Foot contact detection + pinning for foot sliding fix
  #6 Wrist consistency check (body/hand boundary)
  #10 Floor plane estimation and alignment (optional)

Pipeline order (v3.3 — enhanced with new modules):
  0.  Rolling shutter compensation (optional, if non-global-shutter cameras)
  0.5 Outlier frame detection (flags problem frames upfront)
  1.  Outlier spike detection & removal (make spikes NaN)
  2.  Bone-length-based outlier detection (mark extreme tracking failures NaN)
  2.5 Detector drift monitoring (tracks quality over time)
  3.  Linear NaN gap filling (bounded)
  3.5 Foot gap correction (re-anchor heel/foot_index to ankle)
  4.  Rigid bone length enforcement — FIRST pass + bone length solver
  5.  One Euro + Butterworth jitter filtering (Z-weighted)
  6.  Rigid bone length enforcement — SECOND pass + bone length solver
  6.5 Wrist consistency check (body/hand boundary)
  7.  Floor plane alignment (optional)
  8.  Foot contact detection & pinning
  8.5 RTS smoother (optional, zero-lag backward smoothing)
"""
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

from joint_definitions import NUM_BODY, NUM_TOTAL, NUM_FACE, NUM_RIGHT_HAND, NUM_LEFT_HAND
from occlusion_detector import detect_outliers, detect_and_interpolate_gaps
from filter_jitter import apply_combined_filter, apply_one_euro_filter, apply_butterworth_filter
from bone_length_constraint import enforce_rigid_bones_bfs, detect_bone_length_outliers, correct_foot_gaps_after_interpolation
from foot_sliding import detect_and_fix_foot_sliding
from floor_plane import estimate_ground_plane, align_to_ground
from bone_length_solver import correct_bone_lengths
from rts_smoother import smooth_skeleton
from outlier_frame_detector import OutlierFrameDetector
from rolling_shutter_compensation import RollingShutterCompensator
from detector_drift_monitor import DetectorDriftMonitor


def run_full_pipeline(
    skeleton_data,
    fps=30.0,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.007,
    butterworth_cutoff=6.0,
    butterworth_order=2,
    max_gap=10,
    outlier_threshold=5.0,
    enforce_bones=True,
    fix_foot_sliding=True,
    bone_iterations=10,
    velocity_threshold=None,
    skip_bones_for_hands=True,
    align_floor=False,
    target_height=None,
    z_cutoff_ratio=1.5,
    use_rolling_shutter_compensation=False,
    rolling_shutter_readout_ms=33.0,
    use_outlier_frame_detection=False,
    use_bone_length_solver=True,
    bone_solver_threshold=0.20,
    bone_solver_passes=4,
    use_drift_monitor=False,
    use_rts_smoother=False,
    rts_process_noise=1.0,
    rts_observation_noise=10.0,
):
    """Run the full enhanced post-processing pipeline.

    Pipeline order (v3.4 — integrated with all new modules):
    0.  Rolling shutter compensation (optional, if non-global-shutter cameras)
    0.5 Outlier frame detection (flags problem frames upfront)
    1.  Outlier spike detection (make spikes NaN)
    2.  Bone-length outlier detection (mark extreme tracking failures NaN)
    2.5 Detector drift monitoring (tracks quality over time)
    3.  Linear NaN gap filling (bounded)
    3.5 Foot gap correction (re-anchor heel/foot_index to ankle)
    4.  Rigid bone length enforcement — FIRST pass + bone length solver
    5.  One Euro + Butterworth jitter filtering (Z-weighted)
    6.  Rigid bone length enforcement — SECOND pass + bone length solver
    6.5 Wrist consistency check (body/hand boundary)
    7.  Floor plane alignment (optional)
    8.  Foot contact detection & pinning
    8.5 RTS smoother (optional, zero-lag backward smoothing)

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) numpy array
        fps: video frame rate
        one_euro_min_cutoff: One Euro min cutoff (lower = more smoothing)
        one_euro_beta: One Euro speed coefficient (higher = less lag for fast motion)
        butterworth_cutoff: Butterworth cutoff in Hz
        butterworth_order: Butterworth filter order
        max_gap: max NaN gap to interpolate (frames)
        outlier_threshold: std-dev threshold for outlier detection
        enforce_bones: whether to enforce rigid bone lengths
        fix_foot_sliding: whether to fix foot sliding
        bone_iterations: IK constraint iterations (unused, kept for API compat)
        velocity_threshold: foot contact velocity threshold (None = auto)
        skip_bones_for_hands: if True, only apply bone constraints to body
        align_floor: whether to align skeleton to estimated ground plane
        target_height: if provided, scale skeleton to this height in mm
        z_cutoff_ratio: Z axis cutoff multiplier (1.5 = Z gets 1.5x higher cutoff
            = less smoothing on depth axis, which has higher triangulation noise)
        use_rolling_shutter_compensation: if True, pre-correct rolling shutter skew
        rolling_shutter_readout_ms: sensor readout time in ms (for rolling shutter)
        use_outlier_frame_detection: if True, flag problem frames before processing
        use_bone_length_solver: if True, run bone length correction at stages 4 & 6
        bone_solver_threshold: max fractional bone deviation before correction (0.20=20%)
        bone_solver_passes: number of bone solver iterations
        use_drift_monitor: if True, monitor detector quality drift over time
        use_rts_smoother: if True, apply zero-lag RTS Kalman smoother as final step
        rts_process_noise: RTS process noise scale (higher = trust measurements more)
        rts_observation_noise: RTS observation noise scale (higher = smooth more)

    Returns:
        processed_data: (numFrames, numTrackedPoints, 3) numpy array
        report: dict with processing statistics
    """
    print("=" * 60)
    print("  FreeMoCap Enhanced Post-Processing Pipeline (v3.4)")
    print("=" * 60)

    start_time = time.time()
    report = {"stages": {}, "pipeline_version": "3.4-integrated"}

    num_frames = skeleton_data.shape[0]
    print(f"\n  Input: {num_frames} frames, {skeleton_data.shape[1]} tracked points")
    print(f"  FPS: {fps}")

    nan_before = int(np.sum(np.isnan(skeleton_data)))
    print(f"  NaN values before: {nan_before}")

    data = skeleton_data.copy()

    # Stage 0: Rolling shutter compensation (optional, pre-processing)
    if use_rolling_shutter_compensation:
        t0 = time.time()
        print("\n[Stage 0/9] Rolling Shutter Compensation")
        rs_compensator = RollingShutterCompensator(
            readout_time_ms=rolling_shutter_readout_ms,
            fps=fps,
        )
        data = rs_compensator.compensate_sequence(data)
        t1 = time.time()
        report["stages"]["rolling_shutter"] = {
            "time_seconds": t1 - t0,
            "readout_time_ms": rolling_shutter_readout_ms,
        }
        print(f"  Readout time: {rolling_shutter_readout_ms:.1f}ms")
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 0.5: Outlier frame detection (informational — flags problem frames)
    if use_outlier_frame_detection:
        t0 = time.time()
        print("\n[Stage 0.5/9] Outlier Frame Detection")
        outlier_detector = OutlierFrameDetector()
        outlier_summary = outlier_detector.analyze(data, fps=fps)
        t1 = time.time()
        report["stages"]["outlier_frame_detection"] = {
            "time_seconds": t1 - t0,
            "total_flagged": outlier_summary.total_flagged,
            "severity_counts": outlier_summary.severity_counts,
            "category_counts": outlier_summary.category_counts,
            "worst_frames": outlier_summary.worst_frames[:10],
        }
        print(f"  Flagged frames: {outlier_summary.total_flagged}/{outlier_summary.total_frames}")
        for sev, cnt in outlier_summary.severity_counts.items():
            print(f"    {sev}: {cnt}")
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 1: Outlier detection + small gap interpolation (combined)
    t0 = time.time()
    print("\n[Stage 1/9] Outlier Detection & Initial Gap Fill")
    data, gap_diag = detect_and_interpolate_gaps(
        data, max_gap=max_gap, outlier_threshold=outlier_threshold, fps=fps
    )
    t1 = time.time()
    report["stages"]["outlier_removal"] = gap_diag
    report["stages"]["outlier_removal"]["time_seconds"] = t1 - t0
    nan_after_outliers = int(np.sum(np.isnan(data)))
    print(f"  Outliers detected: {gap_diag['outliers_detected']}, gaps filled: {gap_diag['gaps_interpolated']}")
    print(f"  NaN after: {nan_after_outliers}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 2: Bone-length-based outlier detection
    # Frames with extreme bone length errors indicate tracking failure
    t0 = time.time()
    print("\n[Stage 2/9] Bone Length Outlier Detection")
    data, bone_outlier_diag = detect_bone_length_outliers(data, max_relative_error=0.60)
    t1 = time.time()
    report["stages"]["bone_outlier_detection"] = bone_outlier_diag
    report["stages"]["bone_outlier_detection"]["time_seconds"] = t1 - t0
    nan_after_bone_outliers = int(np.sum(np.isnan(data)))
    print(f"  Frames with extreme bone errors: {bone_outlier_diag['frames_marked_nan']}")
    for bone, count in bone_outlier_diag["bones_flagged"].items():
        if count > 0:
            print(f"    {bone}: {count} frames")
    print(f"  NaN after: {nan_after_bone_outliers}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 2.5: Detector drift monitoring (quality tracking over time)
    drift_report = None
    if use_drift_monitor:
        t0 = time.time()
        print("\n[Stage 2.5/9] Detector Drift Monitoring")
        drift_monitor = DetectorDriftMonitor(baseline_frames=min(30, num_frames // 4 + 1))
        for frame_idx in range(num_frames):
            frame_data = data[frame_idx]
            valid_keypoints = ~np.any(np.isnan(frame_data), axis=1)
            completeness = float(np.sum(valid_keypoints)) / max(1, len(valid_keypoints))
            synthetic_detection = {
                "bbox": [0, 0, 1920, 1080],
                "confidence": completeness,
                "keypoints": frame_data[~np.any(np.isnan(frame_data), axis=1)].tolist() if np.any(valid_keypoints) else [],
            }
            drift_monitor.record_frame(
                frame_idx,
                [synthetic_detection],
                timestamp_s=frame_idx / fps,
            )
        drift_report = drift_monitor.get_report()
        t1 = time.time()
        report["stages"]["drift_monitor"] = {
            "time_seconds": t1 - t0,
            "overall_drift_score": drift_report.overall_drift_score,
            "num_alerts": len(drift_report.alerts),
            "summary": drift_report.summary,
        }
        print(f"  Drift score: {drift_report.overall_drift_score:.2f}")
        print(f"  Alerts: {len(drift_report.alerts)}")
        print(f"  Summary: {drift_report.summary[:100]}")
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 3: Linear gap fill (remaining NaN from outlier detection)
    t0 = time.time()
    print("\n[Stage 3/9] Linear NaN Gap Filling")
    original_nan_mask = np.isnan(data).any(axis=2)
    data = _fill_nan_gaps_linear(data, max_gap=max_gap)
    t1 = time.time()
    nan_after_gaps = int(np.sum(np.isnan(data)))
    print(f"  NaN after gap fill: {nan_after_gaps} (filled {nan_after_outliers - nan_after_gaps})")
    report["stages"]["gap_filling"] = {"time_seconds": t1 - t0, "nan_filled": nan_after_outliers - nan_after_gaps}
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 3.5: Foot gap correction (re-anchor heel/foot_index to ankle)
    t0 = time.time()
    print("\n[Stage 3.5/9] Foot Gap Correction (ankle offset propagation)")
    data, foot_gap_info = correct_foot_gaps_after_interpolation(data, original_nan_mask=original_nan_mask)
    t1 = time.time()
    total_foot_fixes = sum(v for k, v in foot_gap_info.items() if k.endswith("_fixes"))
    report["stages"]["foot_gap_correction"] = foot_gap_info
    report["stages"]["foot_gap_correction"]["time_seconds"] = t1 - t0
    print(f"  Total foot corrections: {total_foot_fixes}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 4: Bone enforcement — FIRST pass (on mostly-valid data, before filter)
    if enforce_bones:
        t0 = time.time()
        print("\n[Stage 4/9] Rigid Bone Length Enforcement (1st pass — pre-filter)")
        data = enforce_rigid_bones_bfs(data)
        t1 = time.time()
        report["stages"]["bone_enforcement_pre"] = {"time_seconds": t1 - t0}
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 4b: Bone length solver (post-bone-enforcement, pre-filter)
    if use_bone_length_solver:
        t0 = time.time()
        print("\n[Stage 4b/9] Bone Length Solver (1st pass — pre-filter)")
        data, bone_solver_report = correct_bone_lengths(
            data,
            threshold=bone_solver_threshold,
            num_passes=bone_solver_passes,
            fps=fps,
        )
        t1 = time.time()
        report["stages"]["bone_solver_pre"] = {
            "time_seconds": t1 - t0,
            "num_corrections": bone_solver_report["num_corrections"],
        }
        print(f"  Corrections: {bone_solver_report['num_corrections']}")
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 4: Combined jitter filter (OneEuro + Butterworth)
    # Applied to ALL marker groups: body (0:33), right_hand (33:54),
    # left_hand (54:75), face (75:553)
    t0 = time.time()
    print("\n[Stage 5/9] Combined Jitter Filter (OneEuro + Butterworth)")
    print("    Filtering body (33), right_hand (21), left_hand (21), face (478)...")
    body_filtered = apply_combined_filter(
        data[:, :NUM_BODY, :],
        fps=fps,
        one_euro_min_cutoff=one_euro_min_cutoff,
        one_euro_beta=one_euro_beta,
        butterworth_cutoff=butterworth_cutoff,
        butterworth_order=butterworth_order,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    data[:, :NUM_BODY, :] = body_filtered

    # Filter right hand (indices 33:54) — use higher cutoff (hands move faster)
    rh_start = NUM_BODY
    rh_end = NUM_BODY + NUM_RIGHT_HAND
    rh_filtered = apply_one_euro_filter(
        data[:, rh_start:rh_end, :],
        fps=fps,
        min_cutoff=one_euro_min_cutoff * 1.5,
        beta=one_euro_beta * 1.2,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    data[:, rh_start:rh_end, :] = rh_filtered

    # Filter left hand (indices 54:75)
    lh_start = rh_end
    lh_end = rh_end + NUM_LEFT_HAND
    lh_filtered = apply_one_euro_filter(
        data[:, lh_start:lh_end, :],
        fps=fps,
        min_cutoff=one_euro_min_cutoff * 1.5,
        beta=one_euro_beta * 1.2,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    data[:, lh_start:lh_end, :] = lh_filtered

    # Filter face (indices 75:553) — use gentler filter (face is noisy but high-res)
    face_start = lh_end
    face_end = lh_end + NUM_FACE
    face_filtered = apply_one_euro_filter(
        data[:, face_start:face_end, :],
        fps=fps,
        min_cutoff=one_euro_min_cutoff * 2.0,
        beta=one_euro_beta * 0.5,
        z_cutoff_ratio=z_cutoff_ratio,
    )
    data[:, face_start:face_end, :] = face_filtered
    t1 = time.time()
    report["stages"]["jitter_filter"] = {"time_seconds": t1 - t0}
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 5: Bone enforcement — SECOND pass (fix filter-induced drift)
    # Use higher tolerance (50%) because filters can distort bone lengths
    # by up to ~40% in frames with fast motion
    if enforce_bones:
        t0 = time.time()
        print("\n[Stage 6/9] Rigid Bone Length Enforcement (2nd pass — post-filter, 50% tol)")
        data = enforce_rigid_bones_bfs(data, max_correction_fraction=0.50)
        t1 = time.time()
        report["stages"]["bone_enforcement_post"] = {"time_seconds": t1 - t0}
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 6b: Bone length solver (post-filter, final bone correction)
    if use_bone_length_solver:
        t0 = time.time()
        print("\n[Stage 6b/9] Bone Length Solver (2nd pass — post-filter)")
        data, bone_solver_report_post = correct_bone_lengths(
            data,
            threshold=bone_solver_threshold,
            num_passes=bone_solver_passes,
            fps=fps,
        )
        t1 = time.time()
        report["stages"]["bone_solver_post"] = {
            "time_seconds": t1 - t0,
            "num_corrections": bone_solver_report_post["num_corrections"],
        }
        print(f"  Corrections: {bone_solver_report_post['num_corrections']}")
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 5.5: Wrist consistency — snap hand base to body wrist
    from wrist_consistency import check_wrist_consistency, fix_wrist_consistency
    t0 = time.time()
    print("\n[Stage 7/9] Wrist Consistency Check")
    wrist_diag = check_wrist_consistency(data)
    for side in ["right", "left"]:
        s = wrist_diag[side]
        print(f"  {side}: mean={s.get('mean_mismatch_mm', 0):.1f}mm "
              f"max={s.get('max_mismatch_mm', 0):.1f}mm "
              f"flagged={s.get('flagged_frames', 0)} "
              f"status={s.get('status', '?')}")
    data, fix_info = fix_wrist_consistency(data, strategy="confidence_weighted")
    t1 = time.time()
    report["stages"]["wrist_consistency"] = {"diagnostics": wrist_diag, "fixes": fix_info}
    report["stages"]["wrist_consistency"]["time_seconds"] = t1 - t0
    print(f"  Fixes applied: right={fix_info['right_fixes']}, left={fix_info['left_fixes']}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Stage 5.5: Floor plane alignment (optional)
    ground_normal_used = None
    ground_point_used = None
    if align_floor:
        t0 = time.time()
        print("\n[Stage 7.5/9] Floor Plane Alignment")
        ground_normal_used, ground_point_used, floor_info = estimate_ground_plane(data, fps=fps)
        print(f"  Ground normal: [{ground_normal_used[0]:.3f}, {ground_normal_used[1]:.3f}, {ground_normal_used[2]:.3f}]")
        print(f"  Contact frames: left={len(floor_info.get('left_frames', []))}, right={len(floor_info.get('right_frames', []))}")
        data, align_info = align_to_ground(
            data,
            ground_normal=ground_normal_used,
            ground_point=ground_point_used,
            target_height=target_height,
        )
        if "scale_factor" in align_info:
            print(f"  Scaled {align_info['original_height_mm']:.0f}mm -> {target_height}mm (x{align_info['scale_factor']:.3f})")
        if "rotation_angle_deg" in align_info:
            print(f"  Rotated {align_info['rotation_angle_deg']:.1f} degrees")
        t1 = time.time()
        report["stages"]["floor_alignment"] = {"time_seconds": t1 - t0, "info": align_info}
        print(f"  Time: {t1 - t0:.2f}s")

    # Stage 8: Foot contact detection & pinning
    if fix_foot_sliding:
        t0 = time.time()
        print("\n[Stage 8/9] Foot Contact Detection & Sliding Fix")
        data = detect_and_fix_foot_sliding(
            data, fps=fps, velocity_threshold=velocity_threshold,
        )
        t1 = time.time()
        report["stages"]["foot_sliding"] = {"time_seconds": t1 - t0}
        print(f"  Time: {t1 - t0:.2f}s")
    else:
        print("\n[Stage 8/9] Foot Sliding Fix: SKIPPED")

    # Stage 8.5: RTS Kalman smoother (optional, zero-lag backward smoothing)
    if use_rts_smoother:
        t0 = time.time()
        print("\n[Stage 8.5/9] RTS Kalman Smoother (zero-lag backward pass)")
        rts_result = smooth_skeleton(
            data,
            fps=fps,
            process_noise=rts_process_noise,
            observation_noise=rts_observation_noise,
            interpolate_gaps=True,
        )
        data = rts_result["smoothed"]
        t1 = time.time()
        report["stages"]["rts_smoother"] = {
            "time_seconds": t1 - t0,
            "nan_frames": rts_result["num_nan_frames"],
        }
        print(f"  NaN frames handled: {rts_result['num_nan_frames']}")
        print(f"  Time: {t1 - t0:.2f}s")

    total_time = time.time() - start_time

    nan_after = int(np.sum(np.isnan(data)))

    report["total_time_seconds"] = total_time
    report["nan_before"] = nan_before
    report["nan_after"] = nan_after
    report["nan_reduced"] = nan_before - nan_after

    print("\n" + "=" * 60)
    print(f"  Pipeline v3.4 complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  NaN: {nan_before} -> {nan_after} (reduced {nan_before - nan_after})")
    print("=" * 60)

    return data, report


def _fill_nan_gaps_linear(skeleton_data, max_gap=10):
    """Fill NaN gaps using linear interpolation (bounded, no overshoot).

    Unlike cubic spline, linear interpolation never overshoots, so it
    preserves bone length consistency from the IK stage.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        max_gap: maximum gap length to fill (longer gaps stay as NaN)

    Returns:
        filled_data: (numFrames, numTrackedPoints, 3) array
    """
    filled = skeleton_data.copy()
    num_frames, num_tracked, num_dims = skeleton_data.shape

    total_filled = 0

    for tracked_idx in range(num_tracked):
        for dim in range(num_dims):
            channel = filled[:, tracked_idx, dim]
            nan_mask = np.isnan(channel)

            if not np.any(nan_mask):
                continue

            valid = ~nan_mask
            if np.sum(valid) < 2:
                continue

            valid_indices = np.where(valid)[0]
            valid_values = channel[valid]

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
                    total_filled += int(gap_len)

            filled[:, tracked_idx, dim] = channel

    return filled


def process_freemocap_recording(
    recording_folder_path,
    output_suffix="_enhanced",
    save_output=True,
    **pipeline_kwargs,
):
    """Process a complete FreeMoCap recording folder.

    Reads the raw 3D data, applies enhanced post-processing,
    and saves results alongside original files.

    Args:
        recording_folder_path: path to the recording folder
        output_suffix: suffix for output filenames
        save_output: whether to save processed data to disk
        **pipeline_kwargs: additional arguments for run_full_pipeline

    Returns:
        processed_data: (numFrames, numTrackedPoints, 3) numpy array
        report: dict with processing statistics
    """
    output_data_dir = os.path.join(recording_folder_path, "output_data")

    raw_3d_path = os.path.join(
        output_data_dir, "raw_data",
        "mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ.npy"
    )

    if not os.path.exists(raw_3d_path):
        alt_paths = [
            os.path.join(output_data_dir, "mediapipe_skeleton_3d.npy"),
            os.path.join(output_data_dir, "raw_data", "mediapipe_3dData_numFrames_numTrackedPoints_reprojectionError.npy"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                raw_3d_path = alt
                break
        else:
            raise FileNotFoundError(
                f"Could not find 3D skeleton data in {recording_folder_path}. "
                f"Expected: {raw_3d_path}"
            )

    print(f"Loading: {raw_3d_path}")
    skeleton_data = np.load(raw_3d_path)
    print(f"Loaded: shape={skeleton_data.shape}, dtype={skeleton_data.dtype}")

    fps = pipeline_kwargs.pop("fps", 30.0)

    processed_data, report = run_full_pipeline(
        skeleton_data, fps=fps, **pipeline_kwargs
    )

    if save_output:
        output_dir = os.path.join(output_data_dir, "raw_data")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"mediapipe_3dData_numFrames_numTrackedPoints_spatialXYZ{output_suffix}.npy"
        )
        np.save(output_path, processed_data)
        print(f"\nSaved enhanced data: {output_path}")

        skeleton_path = os.path.join(
            output_data_dir,
            f"mediapipe_skeleton_3d{output_suffix}.npy"
        )
        np.save(skeleton_path, processed_data)
        print(f"Saved enhanced skeleton: {skeleton_path}")

    return processed_data, report
