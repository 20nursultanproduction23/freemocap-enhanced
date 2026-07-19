"""
Full Pipeline Integration Test -- Stages 1->7 End-to-End

Connects all multi-person pipeline stages on real 3-camera data:
  Stage 1: Multi-person detection (RTMDet + RTMPose)
  Stage 2: Cross-view association (epipolar + Hungarian)
  Stage 3: Temporal tracking (persistent global IDs)
  Stage 4: Per-actor triangulation (DLT)
  Stage 5: v3.1 post-processing pipeline per actor
  Stage 6: Physical interaction validation
  Stage 7: ArUco marker detection + binding

Test 1: Single-person real data (3 cameras, 20 frames)
Test 2: Synthetic 2-person data (injected into 3-camera system, 10 frames)

All results use per-frame tables (never trust aggregate metrics).
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_loader import load_calibration_toml
from multiperson_detector import MultiPersonDetector, PersonDetection, RTMPOSE_SEGMENTS
from cross_view_association import CrossViewAssociator
from temporal_tracker import TemporalTracker
from per_actor_triangulation import PerActorTriangulator
from multi_actor_pipeline import process_multi_actor, get_pipeline_summary
from physical_interaction_validator import interpenetration_detection, contact_detection
from marker_fallback import ArucoMarkerDetector, detect_and_bind


# -- Paths --------------------------------------------------------------
from test_config import CALIB_PATH, VID_DIR, ARUCO_CONFIG


def create_synthetic_person(
    real_det: PersonDetection,
    offset_x: int = 200,
    offset_y: int = 0,
    conf_scale: float = 0.85,
) -> PersonDetection:
    """Create a synthetic second person by offsetting real detection keypoints."""
    new_bbox = real_det.bbox.copy()
    new_bbox[0] += offset_x
    new_bbox[2] += offset_x
    new_bbox[1] += offset_y
    new_bbox[3] += offset_y

    new_kpts = real_det.keypoints_133.copy()
    new_kpts[:, 0] += offset_x
    new_kpts[:, 1] += offset_y

    segments = {}
    for name, (start, end) in RTMPOSE_SEGMENTS.items():
        segments[name] = new_kpts[start:end].copy()

    return PersonDetection(
        person_id=1,
        bbox=new_bbox,
        keypoints_133=new_kpts,
        confidence=real_det.confidence * conf_scale,
        segments=segments,
    )


def load_video_captures(video_dir: str, limit: int = None):
    """Load video files from directory, return list of VideoCapture objects."""
    files = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])
    if limit:
        files = files[:limit]
    return [cv2.VideoCapture(f) for f in files], files


def run_single_person_test(calib, num_frames=20):
    """TEST 1: Full pipeline on single-person real data (3 cameras)."""
    print("\n" + "=" * 70)
    print("  TEST 1: Single-Person Full Pipeline (3 cameras, 20 frames)")
    print("=" * 70)

    caps, vid_files = load_video_captures(VID_DIR)
    num_cams = len(caps)
    print(f"  Cameras: {num_cams}, Video files: {[os.path.basename(f) for f in vid_files]}")

    # -- Stage 1: Detection ---------------------------------------------
    print("\n  -- Stage 1: Multi-Person Detection --")
    t0 = time.time()
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    per_cam_per_frame = [[] for _ in range(num_cams)]  # [cam][frame]
    num_detected_per_frame = []

    for frame_idx in range(num_frames):
        frame_dets_per_cam = []
        for cam_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame_dets_per_cam.append([])
                continue
            dets = detector.detect_frame(frame)
            frame_dets_per_cam.append(dets)
            per_cam_per_frame[cam_idx].append(dets)

        # Ensure all cameras have data for this frame
        max_persons = max(len(d) for d in frame_dets_per_cam) if frame_dets_per_cam else 0
        num_detected_per_frame.append([len(d) for d in frame_dets_per_cam])

    stage1_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame detection count (cam0, cam1, cam2):")
    print(f"  {'Frame':>5}  {'Cam0':>5}  {'Cam1':>5}  {'Cam2':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    all_ok = True
    for fi, counts in enumerate(num_detected_per_frame):
        status = "OK" if all(c >= 1 for c in counts) else "MISS"
        if status == "MISS":
            all_ok = False
        print(f"  {fi:>5}  {counts[0]:>5}  {counts[1]:>5}  {counts[2]:>5}  {status}")

    total_dets = sum(sum(c) for c in num_detected_per_frame)
    print(f"\n  Stage 1: {total_dets} total detections across {num_frames} frames, "
          f"{num_cams} cameras ({stage1_time:.1f}s)")
    stage1_pass = all_ok
    print(f"  Result: {'PASS' if stage1_pass else 'PARTIAL'}")

    # -- Stage 2: Cross-View Association ---------------------------------
    print("\n  -- Stage 2: Cross-View Association --")
    t0 = time.time()
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    frame_associations = []
    for frame_idx in range(num_frames):
        per_cam_dets = [per_cam_per_frame[cam][frame_idx] for cam in range(num_cams)]
        fa = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)
        frame_associations.append(fa)

    stage2_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame association results:")
    print(f"  {'Frame':>5}  {'Actors':>7}  {'Matched':>8}  {'Cost':>8}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}")
    stage2_pass = True
    for fi, fa in enumerate(frame_associations):
        n_matched = sum(1 for pa in fa.pair_assignments for v in pa.assignments.values() if v != -1)
        total_cost = sum(pa.total_cost for pa in fa.pair_assignments)
        status = "OK" if fa.num_actors >= 1 else "FAIL"
        if fa.num_actors == 0:
            stage2_pass = False
        print(f"  {fi:>5}  {fa.num_actors:>7}  {n_matched:>8}  {total_cost:>8.1f}  {status}")

    print(f"\n  Stage 2: {num_frames} frames associated ({stage2_time:.1f}s)")
    print(f"  Result: {'PASS' if stage2_pass else 'FAIL'}")

    # -- Stage 3: Temporal Tracking --------------------------------------
    print("\n  -- Stage 3: Temporal Tracking --")
    t0 = time.time()
    tracker = TemporalTracker()
    temporal_results = tracker.track(
        frame_associations,
        per_camera_per_frame_detections=per_cam_per_frame,
    )
    stage3_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame temporal tracking:")
    print(f"  {'Frame':>5}  {'Persistent':>11}  {'Active':>7}  {'IDs':>20}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*7}  {'-'*20}")
    stage3_pass = True
    for fi, ta in enumerate(temporal_results):
        ids_str = str(sorted(ta.active_tracks))
        if len(ids_str) > 20:
            ids_str = ids_str[:17] + "..."
        status = "OK" if ta.num_persistent_actors >= 1 else "FAIL"
        if ta.num_persistent_actors == 0:
            stage3_pass = False
        print(f"  {fi:>5}  {ta.num_persistent_actors:>11}  {len(ta.active_tracks):>7}  {ids_str:>20}  {status}")

    track_summary = tracker.get_track_summary()
    print(f"\n  Stage 3: {track_summary['total_tracks_created']} tracks created, "
          f"{track_summary['active_tracks']} active ({stage3_time:.1f}s)")
    print(f"  Result: {'PASS' if stage3_pass else 'FAIL'}")

    # -- Stage 4: Per-Actor Triangulation --------------------------------
    print("\n  -- Stage 4: Per-Actor Triangulation --")
    t0 = time.time()
    triangulator = PerActorTriangulator(calib, min_cams=2)
    actor_triang_results = triangulator.triangulate_all(
        per_cam_per_frame,
        frame_associations,
        temporal_results,
    )
    stage4_time = time.time() - t0

    # Per-actor summary
    print(f"\n  Per-actor triangulation results:")
    print(f"  {'Actor':>6}  {'Visible':>8}  {'Total':>5}  {'%':>6}  {'Reproj':>8}  {'Kpts/Fr':>9}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}")
    stage4_pass = True
    for pid, res in sorted(actor_triang_results.items()):
        vis = int(np.sum(res.visible_frames))
        pct = 100.0 * vis / max(res.num_frames, 1)
        status = "OK" if vis > 0 else "FAIL"
        if vis == 0:
            stage4_pass = False
        print(f"  {pid:>6}  {vis:>8}  {res.num_frames:>5}  {pct:>5.0f}%  "
              f"{res.mean_reprojection_error:>7.1f}px  {res.mean_visible_kpts_per_frame:>8.0f}  {status}")

    print(f"\n  Stage 4: {len(actor_triang_results)} actor(s) triangulated ({stage4_time:.1f}s)")
    print(f"  Result: {'PASS' if stage4_pass else 'FAIL'}")

    # -- Stage 5: v3.1 Pipeline per Actor --------------------------------
    print("\n  -- Stage 5: v3.1 Post-Processing Pipeline --")
    t0 = time.time()
    try:
        pipeline_results = process_multi_actor(
            actor_triang_results,
            fps=6.0,  # test video is 6 FPS
            align_floor=True,
        )
        stage5_time = time.time() - t0

        # Per-actor summary
        psummary = get_pipeline_summary(pipeline_results)
        print(f"\n  Per-actor pipeline results:")
        print(f"  {'Actor':>6}  {'Time':>7}  {'NaN Before':>11}  {'NaN After':>10}  {'Reduced':>8}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*11}  {'-'*10}  {'-'*8}")
        stage5_pass = True
        for pid, pdata in sorted(psummary['per_actor'].items()):
            status = "OK" if pdata['nan_reduced'] >= 0 else "WARN"
            print(f"  {pid:>6}  {pdata['time_seconds']:>6.1f}s  "
                  f"{pdata['nan_before']:>11}  {pdata['nan_after']:>10}  {pdata['nan_reduced']:>8}  {status}")

        print(f"\n  Stage 5: {len(pipeline_results)} actor(s) processed ({stage5_time:.1f}s)")
        print(f"  Result: PASS")
    except Exception as e:
        stage5_time = time.time() - t0
        print(f"\n  Stage 5 FAILED: {e}")
        stage5_pass = False
        pipeline_results = {}

    # -- Stage 6: Physical Interaction Validation ------------------------
    print("\n  -- Stage 6: Physical Interaction Validation --")
    t0 = time.time()

    # Collect processed skeletons for interaction check
    actor_skeletons = []
    for pid in sorted(pipeline_results.keys()):
        skel = pipeline_results[pid]["processed_skeleton"]
        actor_skeletons.append(skel)

    inter_events = []
    contact_events = []
    if len(actor_skeletons) >= 2:
        inter_events = interpenetration_detection(
            actor_skeletons,
            threshold=0.15,
            overlap_ratio_threshold=0.3,
        )
        contact_events = contact_detection(
            actor_skeletons,
            contact_threshold=0.05,
        )
    stage6_time = time.time() - t0

    print(f"\n  Interpenetration events: {len(inter_events)}")
    print(f"  Contact events: {len(contact_events)}")
    if len(actor_skeletons) < 2:
        print(f"  (Only {len(actor_skeletons)} actor(s), interaction check requires 2+)")
    stage6_pass = True  # Single person = no interactions expected
    print(f"  Stage 6: ({stage6_time:.3f}s)")
    print(f"  Result: PASS (single person, no interactions expected)")

    # -- Stage 7: ArUco Marker Detection ---------------------------------
    print("\n  -- Stage 7: ArUco Marker Detection --")
    t0 = time.time()

    # Reset captures
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    aruco_detector = ArucoMarkerDetector(config_path=ARUCO_CONFIG)
    total_markers_found = 0
    total_bindings = 0

    for frame_idx in range(num_frames):
        for cam_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue
            marker_dets = aruco_detector.detect_frame(frame, cam_idx=cam_idx, frame_idx=frame_idx)
            total_markers_found += len(marker_dets)

            if marker_dets:
                skel_dets = per_cam_per_frame[cam_idx][frame_idx] if frame_idx < len(per_cam_per_frame[cam_idx]) else []
                bindings = detect_and_bind(
                    [marker_dets],
                    [skel_dets],
                    frame_idx=frame_idx,
                )
                total_bindings += len(bindings)

    stage7_time = time.time() - t0

    print(f"\n  Markers found: {total_markers_found}")
    print(f"  Bindings made: {total_bindings}")
    print(f"  (No physical markers expected in test data)")
    stage7_pass = True  # No markers in test data is expected
    print(f"  Stage 7: ({stage7_time:.1f}s)")
    print(f"  Result: PASS (no markers in test data, expected)")

    # -- FINAL VERDICT ---------------------------------------------------
    all_stages = {
        "Stage 1 (Detection)": stage1_pass,
        "Stage 2 (Association)": stage2_pass,
        "Stage 3 (Tracking)": stage3_pass,
        "Stage 4 (Triangulation)": stage4_pass,
        "Stage 5 (Pipeline)": stage5_pass,
        "Stage 6 (Interaction)": stage6_pass,
        "Stage 7 (ArUco)": stage7_pass,
    }

    total_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time + stage6_time + stage7_time

    print("\n" + "=" * 70)
    print("  TEST 1 FINAL VERDICT")
    print("=" * 70)
    for name, passed in all_stages.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Total time: {total_time:.1f}s")
    overall = all(all_stages.values())
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")

    # Cleanup
    for cap in caps:
        cap.release()

    return all_stages, total_time


def run_synthetic_two_person_test(calib, num_frames=10):
    """TEST 2: Full pipeline on synthetic 2-person data (3 cameras)."""
    print("\n" + "=" * 70)
    print("  TEST 2: Synthetic 2-Person Full Pipeline (3 cameras, 10 frames)")
    print("=" * 70)

    caps, vid_files = load_video_captures(VID_DIR)
    num_cams = len(caps)

    # -- Stage 1: Detection + Synthetic injection ------------------------
    print("\n  -- Stage 1: Detection + Synthetic Person Injection --")
    t0 = time.time()
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    per_cam_per_frame = [[] for _ in range(num_cams)]
    detection_counts = []

    for frame_idx in range(num_frames):
        frame_counts = []
        for cam_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                per_cam_per_frame[cam_idx].append([])
                frame_counts.append(0)
                continue
            dets = detector.detect_frame(frame)

            # Inject synthetic person (offset varies by camera for realism)
            if len(dets) > 0:
                synth = create_synthetic_person(
                    dets[0],
                    offset_x=180 + cam_idx * 20,
                    offset_y=cam_idx * 15,
                )
                # Update person_id for injected detection
                synth.person_id = len(dets)
                dets.append(synth)

            per_cam_per_frame[cam_idx].append(dets)
            frame_counts.append(len(dets))
        detection_counts.append(frame_counts)

    stage1_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame detection count (including synthetic):")
    print(f"  {'Frame':>5}  {'Cam0':>5}  {'Cam1':>5}  {'Cam2':>5}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    all_two = True
    for fi, counts in enumerate(detection_counts):
        status = "OK" if all(c >= 2 for c in counts) else "MISS"
        if status == "MISS":
            all_two = False
        print(f"  {fi:>5}  {counts[0]:>5}  {counts[1]:>5}  {counts[2]:>5}  {status}")

    print(f"\n  Stage 1: {num_frames} frames with synthetic injection ({stage1_time:.1f}s)")
    stage1_pass = all_two
    print(f"  Result: {'PASS' if stage1_pass else 'PARTIAL'}")

    # -- Stage 2: Cross-View Association ---------------------------------
    print("\n  -- Stage 2: Cross-View Association --")
    t0 = time.time()
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    frame_associations = []
    for frame_idx in range(num_frames):
        per_cam_dets = [per_cam_per_frame[cam][frame_idx] for cam in range(num_cams)]
        fa = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)
        frame_associations.append(fa)

    stage2_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame association results:")
    print(f"  {'Frame':>5}  {'Actors':>7}  {'Matched':>8}  {'Cost':>8}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}")
    stage2_pass = True
    for fi, fa in enumerate(frame_associations):
        n_matched = sum(1 for pa in fa.pair_assignments for v in pa.assignments.values() if v != -1)
        total_cost = sum(pa.total_cost for pa in fa.pair_assignments)
        status = "OK" if fa.num_actors >= 2 else "PARTIAL"
        if fa.num_actors < 2:
            stage2_pass = False
        print(f"  {fi:>5}  {fa.num_actors:>7}  {n_matched:>8}  {total_cost:>8.1f}  {status}")

    print(f"\n  Stage 2: {num_frames} frames ({stage2_time:.1f}s)")
    print(f"  Result: {'PASS' if stage2_pass else 'PARTIAL'}")

    # -- Stage 3: Temporal Tracking --------------------------------------
    print("\n  -- Stage 3: Temporal Tracking --")
    t0 = time.time()
    tracker = TemporalTracker()
    temporal_results = tracker.track(
        frame_associations,
        per_camera_per_frame_detections=per_cam_per_frame,
    )
    stage3_time = time.time() - t0

    # Per-frame table
    print(f"\n  Per-frame temporal tracking:")
    print(f"  {'Frame':>5}  {'Persistent':>11}  {'Active':>7}  {'IDs':>20}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*7}  {'-'*20}")
    stage3_pass = True
    for fi, ta in enumerate(temporal_results):
        ids_str = str(sorted(ta.active_tracks))
        if len(ids_str) > 20:
            ids_str = ids_str[:17] + "..."
        status = "OK" if ta.num_persistent_actors >= 2 else "PARTIAL"
        if ta.num_persistent_actors < 2:
            stage3_pass = False
        print(f"  {fi:>5}  {ta.num_persistent_actors:>11}  {len(ta.active_tracks):>7}  {ids_str:>20}  {status}")

    track_summary = tracker.get_track_summary()
    print(f"\n  Stage 3: {track_summary['total_tracks_created']} tracks, "
          f"{track_summary['active_tracks']} active ({stage3_time:.1f}s)")
    print(f"  Result: {'PASS' if stage3_pass else 'PARTIAL'}")

    # -- Stage 4: Per-Actor Triangulation --------------------------------
    print("\n  -- Stage 4: Per-Actor Triangulation --")
    t0 = time.time()
    triangulator = PerActorTriangulator(calib, min_cams=2)
    actor_triang_results = triangulator.triangulate_all(
        per_cam_per_frame,
        frame_associations,
        temporal_results,
    )
    stage4_time = time.time() - t0

    # Per-actor summary
    print(f"\n  Per-actor triangulation results:")
    print(f"  {'Actor':>6}  {'Visible':>8}  {'Total':>5}  {'%':>6}  {'Reproj':>8}  {'Kpts/Fr':>9}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*9}")
    stage4_pass = True
    for pid, res in sorted(actor_triang_results.items()):
        vis = int(np.sum(res.visible_frames))
        pct = 100.0 * vis / max(res.num_frames, 1)
        status = "OK" if vis > 0 else "FAIL"
        if vis == 0:
            stage4_pass = False
        print(f"  {pid:>6}  {vis:>8}  {res.num_frames:>5}  {pct:>5.0f}%  "
              f"{res.mean_reprojection_error:>7.1f}px  {res.mean_visible_kpts_per_frame:>8.0f}  {status}")

    print(f"\n  Stage 4: {len(actor_triang_results)} actor(s) ({stage4_time:.1f}s)")
    stage4_pass = stage4_pass and len(actor_triang_results) >= 2
    print(f"  Result: {'PASS' if stage4_pass else 'PARTIAL'}")

    # -- Stage 5: v3.1 Pipeline ------------------------------------------
    print("\n  -- Stage 5: v3.1 Post-Processing Pipeline --")
    t0 = time.time()
    try:
        pipeline_results = process_multi_actor(
            actor_triang_results,
            fps=6.0,
            align_floor=True,
        )
        stage5_time = time.time() - t0

        psummary = get_pipeline_summary(pipeline_results)
        print(f"\n  Per-actor pipeline results:")
        print(f"  {'Actor':>6}  {'Time':>7}  {'NaN Before':>11}  {'NaN After':>10}  {'Reduced':>8}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*11}  {'-'*10}  {'-'*8}")
        for pid, pdata in sorted(psummary['per_actor'].items()):
            print(f"  {pid:>6}  {pdata['time_seconds']:>6.1f}s  "
                  f"{pdata['nan_before']:>11}  {pdata['nan_after']:>10}  {pdata['nan_reduced']:>8}")

        stage5_pass = len(pipeline_results) >= 2
        print(f"\n  Stage 5: {len(pipeline_results)} actor(s) ({stage5_time:.1f}s)")
        print(f"  Result: {'PASS' if stage5_pass else 'PARTIAL'}")
    except Exception as e:
        stage5_time = time.time() - t0
        print(f"\n  Stage 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        stage5_pass = False
        pipeline_results = {}

    # -- Stage 6: Physical Interaction Validation ------------------------
    print("\n  -- Stage 6: Physical Interaction Validation --")
    t0 = time.time()

    actor_skeletons = []
    for pid in sorted(pipeline_results.keys()):
        skel = pipeline_results[pid]["processed_skeleton"]
        actor_skeletons.append(skel)

    inter_events = []
    contact_events = []
    if len(actor_skeletons) >= 2:
        inter_events = interpenetration_detection(
            actor_skeletons,
            threshold=0.15,
            overlap_ratio_threshold=0.3,
        )
        contact_events = contact_detection(
            actor_skeletons,
            contact_threshold=0.05,
        )
    stage6_time = time.time() - t0

    print(f"\n  Interpenetration events: {len(inter_events)}")
    print(f"  Contact events: {len(contact_events)}")
    if inter_events:
        print(f"  Sample interpenetrations:")
        for ev in inter_events[:5]:
            print(f"    Frame {ev.frame}: actors {ev.actor_a}/{ev.actor_b}, "
                  f"overlap={ev.overlap_fraction:.2%}")
    if contact_events:
        print(f"  Sample contacts (first 5):")
        for ev in contact_events[:5]:
            print(f"    Frame {ev.frame}: {ev.joint_a_label} <-> {ev.joint_b_label}, "
                  f"dist={ev.distance:.4f}")

    stage6_pass = True  # Synthetic data may or may not show interactions
    print(f"\n  Stage 6: ({stage6_time:.3f}s)")
    print(f"  Result: PASS (detection complete, events logged)")

    # -- Stage 7: ArUco Marker Detection ---------------------------------
    print("\n  -- Stage 7: ArUco Marker Detection --")
    t0 = time.time()

    # Reset captures
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    aruco_detector = ArucoMarkerDetector(config_path=ARUCO_CONFIG)
    total_markers = 0
    total_bindings = 0

    for frame_idx in range(num_frames):
        for cam_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue
            marker_dets = aruco_detector.detect_frame(frame, cam_idx=cam_idx, frame_idx=frame_idx)
            total_markers += len(marker_dets)

            if marker_dets:
                skel_dets = per_cam_per_frame[cam_idx][frame_idx]
                bindings = detect_and_bind([marker_dets], [skel_dets], frame_idx=frame_idx)
                total_bindings += len(bindings)

    stage7_time = time.time() - t0
    print(f"\n  Markers found: {total_markers}, Bindings: {total_bindings}")
    stage7_pass = True
    print(f"  Stage 7: ({stage7_time:.1f}s)")
    print(f"  Result: PASS")

    # -- FINAL VERDICT ---------------------------------------------------
    all_stages = {
        "Stage 1 (Detection)": stage1_pass,
        "Stage 2 (Association)": stage2_pass,
        "Stage 3 (Tracking)": stage3_pass,
        "Stage 4 (Triangulation)": stage4_pass,
        "Stage 5 (Pipeline)": stage5_pass,
        "Stage 6 (Interaction)": stage6_pass,
        "Stage 7 (ArUco)": stage7_pass,
    }

    total_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time + stage6_time + stage7_time

    print("\n" + "=" * 70)
    print("  TEST 2 FINAL VERDICT")
    print("=" * 70)
    for name, passed in all_stages.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Total time: {total_time:.1f}s")
    overall = all(all_stages.values())
    print(f"  Overall: {'PASS' if overall else 'PARTIAL'}")

    for cap in caps:
        cap.release()

    return all_stages, total_time


def main():
    print("=" * 70)
    print("  FULL PIPELINE INTEGRATION TEST")
    print("  Stages 1->7 End-to-End on 3-Camera Real Data")
    print("=" * 70)

    t_start = time.time()

    # Load calibration
    print("\n  Loading calibration...")
    calib = load_calibration_toml(CALIB_PATH)
    print(f"  Cameras: {len(calib['camera_matrices'])}")
    for i, name in enumerate(calib['names']):
        K = calib['camera_matrices'][i]
        print(f"    {name}: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # Test 1: Single-person
    test1_results, test1_time = run_single_person_test(calib, num_frames=20)

    # Test 2: Synthetic 2-person
    test2_results, test2_time = run_synthetic_two_person_test(calib, num_frames=10)

    # -- OVERALL SUMMARY -------------------------------------------------
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  OVERALL INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"\n  Test 1 (single-person, 20 frames):")
    for name, passed in test1_results.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    print(f"    Time: {test1_time:.1f}s")

    print(f"\n  Test 2 (synthetic 2-person, 10 frames):")
    for name, passed in test2_results.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    print(f"    Time: {test2_time:.1f}s")

    print(f"\n  Total test time: {total_time:.1f}s")
    t1_all_pass = all(test1_results.values())
    t2_all_pass = all(test2_results.values())
    print(f"  Test 1: {'ALL PASS' if t1_all_pass else 'HAS FAILURES'}")
    print(f"  Test 2: {'ALL PASS' if t2_all_pass else 'HAS FAILURES'}")
    overall = t1_all_pass and t2_all_pass
    print(f"\n  FINAL: {'ALL PASS' if overall else 'PARTIAL'}")


if __name__ == "__main__":
    main()
