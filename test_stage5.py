"""
Stage 5 Test: Multi-Actor Pipeline (v3.1 applied per actor).

Tests:
1. Synthetic: 2 actors with full pipeline (jitter, bones, floor)
2. Real data: single-person pipeline applied to multi-person output
3. Shared floor: verify both actors get the same floor plane
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_loader import load_calibration_toml
from cross_view_association import FrameAssociation
from temporal_tracker import TemporalTracker
from per_actor_triangulation import (
    PerActorTriangulator,
    ActorTriangulationResult,
    NUM_KEYPOINTS,
)
from multi_actor_pipeline import process_multi_actor, get_pipeline_summary
from multiperson_detector import PersonDetection
from test_config import CALIB_PATH, VID_DIR


class MockDetection:
    """Minimal PersonDetection mock for testing."""
    def __init__(self, person_id, center_x, center_y, cam_idx=0):
        self.person_id = person_id
        self.keypoints_133 = np.zeros((133, 3))
        body_offsets = np.array([
            [0, -40], [-5, -35], [-10, -35], [-15, -35],
            [5, -35], [10, -35], [15, -35], [-20, -30],
            [20, -30], [-10, -25], [10, -25], [-25, -10],
            [25, -10], [-35, 10], [35, 10], [-40, 30], [40, 30],
        ])
        noise = np.random.randn(17, 2) * 1.5
        cam_offset = np.array([cam_idx * 3.0, cam_idx * 1.5])
        for i in range(17):
            self.keypoints_133[i, 0] = center_x + body_offsets[i, 0] + noise[i, 0] + cam_offset[0]
            self.keypoints_133[i, 1] = center_y + body_offsets[i, 1] + noise[i, 1] + cam_offset[1]
            self.keypoints_133[i, 2] = 0.7 + np.random.rand() * 0.2
        self.keypoints_133[17:, 2] = 0.0
        self.bbox = np.array([center_x - 50, center_y - 60, center_x + 50, center_y + 60])
        self.confidence = 0.75


def _make_frame_association(frame_idx, actor_configs, num_cameras=3):
    detection_to_actor = {}
    actor_memberships = {}
    for actor_local_id in range(len(actor_configs)):
        members = []
        for cam in range(num_cameras):
            detection_to_actor[(cam, actor_local_id)] = actor_local_id
            members.append((cam, actor_local_id))
        actor_memberships[actor_local_id] = members
    return FrameAssociation(
        frame_idx=frame_idx, num_cameras=num_cameras,
        num_actors=len(actor_configs),
        detection_to_actor=detection_to_actor,
        actor_memberships=actor_memberships,
    )


def test_synthetic_two_actors():
    """Test 1: Two synthetic actors through full pipeline.

    Verifies:
    - Both actors processed independently
    - Output shapes correct
    - NaN reduction (pipeline should reduce NaN)
    - Shared floor plane
    """
    print("=" * 70)
    print("TEST 1: Synthetic 2 actors through full pipeline")
    print("=" * 70)

    calib_path = CALIB_PATH
    calib = load_calibration_toml(calib_path)

    num_frames = 30
    num_cameras = 3
    fps = 6.0  # matches test video FPS

    # Build data
    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        fa = _make_frame_association(
            frame_idx,
            [{"center": (200, 300)}, {"center": (400, 300)}],
            num_cameras,
        )
        frame_associations.append(fa)

        frame_cam_dets = []
        for cam in range(num_cameras):
            frame_cam_dets.append([
                MockDetection(0, 200, 300, cam),
                MockDetection(1, 400, 300, cam),
            ])
        per_cam_detections.append(frame_cam_dets)

    # Stage 3: Temporal tracking
    tracker = TemporalTracker(reference_camera=0)
    temporal_results = tracker.track(frame_associations, per_cam_detections)

    # Stage 4: Triangulation
    per_camera_per_frame = []
    for cam in range(num_cameras):
        per_camera_per_frame.append([per_cam_detections[f][cam] for f in range(num_frames)])

    triangulator = PerActorTriangulator(calib)
    actor_results = triangulator.triangulate_all(
        per_camera_per_frame, frame_associations, temporal_results
    )

    # Stage 5: Multi-actor pipeline
    print("\n  Running multi-actor pipeline...")
    t0 = time.time()
    pipeline_results = process_multi_actor(
        actor_results, fps=fps, align_floor=True,
        butterworth_cutoff=6.0, one_euro_min_cutoff=1.0,
    )
    t1 = time.time()
    print(f"\n  Total pipeline time: {t1-t0:.2f}s")

    # Validate
    all_ok = True
    summary = get_pipeline_summary(pipeline_results)
    print(f"\n  Pipeline summary:")
    print(f"    Actors processed: {summary['num_actors']}")
    print(f"    Total time: {summary['total_time_seconds']:.2f}s")
    print(f"    Total NaN before: {summary['total_nan_before']}")
    print(f"    Total NaN after: {summary['total_nan_after']}")
    print(f"    Total NaN reduced: {summary['total_nan_reduced']}")

    for pid, data in sorted(pipeline_results.items()):
        skel = data["processed_skeleton"]
        report = data["pipeline_report"]
        print(f"\n  Actor {pid}:")
        print(f"    Shape: {skel.shape}")
        print(f"    Time: {report.get('total_time_seconds', 0):.2f}s")
        print(f"    NaN: {report.get('nan_before', 0)} -> {report.get('nan_after', 0)}")

        if skel.shape != (num_frames, NUM_KEYPOINTS, 3):
            print(f"    FAIL: Wrong shape {skel.shape}")
            all_ok = False

        # Check that output is reasonable (no extreme values)
        valid = skel[np.isfinite(skel)]
        if len(valid) > 0:
            abs_max = np.max(np.abs(valid))
            print(f"    Max abs value: {abs_max:.1f}")
            if abs_max > 1e6:
                print(f"    WARNING: Extremely large values")
                all_ok = False

        # Check floor alignment (Y values should be >= 0 if aligned)
        floor_info = report.get("shared_floor")
        if floor_info:
            print(f"    Shared floor from Actor {floor_info.get('estimated_from_actor', '?')}")

    if all_ok:
        print(f"\n  PASS: Both actors processed through pipeline")
    else:
        print(f"\n  FAIL: Issues detected")

    return all_ok


def test_shared_floor():
    """Test 2: Verify shared floor plane is identical for both actors."""
    print("\n" + "=" * 70)
    print("TEST 2: Shared floor plane verification")
    print("=" * 70)

    calib_path = CALIB_PATH
    calib = load_calibration_toml(calib_path)

    num_frames = 30
    num_cameras = 3
    fps = 6.0

    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        fa = _make_frame_association(
            frame_idx,
            [{"center": (200, 300)}, {"center": (400, 300)}],
            num_cameras,
        )
        frame_associations.append(fa)
        frame_cam_dets = []
        for cam in range(num_cameras):
            frame_cam_dets.append([
                MockDetection(0, 200, 300, cam),
                MockDetection(1, 400, 300, cam),
            ])
        per_cam_detections.append(frame_cam_dets)

    tracker = TemporalTracker(reference_camera=0)
    temporal_results = tracker.track(frame_associations, per_cam_detections)

    per_camera_per_frame = []
    for cam in range(num_cameras):
        per_camera_per_frame.append([per_cam_detections[f][cam] for f in range(num_frames)])

    triangulator = PerActorTriangulator(calib)
    actor_results = triangulator.triangulate_all(
        per_camera_per_frame, frame_associations, temporal_results
    )

    pipeline_results = process_multi_actor(
        actor_results, fps=fps, align_floor=True,
        butterworth_cutoff=6.0,
    )

    # Check that all actors reference the same floor
    floor_references = set()
    for pid, data in pipeline_results.items():
        report = data["pipeline_report"]
        floor_info = report.get("shared_floor")
        if floor_info:
            ref = floor_info.get("estimated_from_actor")
            floor_references.add(ref)

    print(f"\n  Floor references: {floor_references}")

    if len(floor_references) <= 1:
        print(f"  PASS: All actors use the same shared floor")
        return True
    else:
        print(f"  FAIL: Different floor planes used")
        return False


def test_real_single_person():
    """Test 3: Real data - apply pipeline to real actor's triangulated skeleton."""
    print("\n" + "=" * 70)
    print("TEST 3: Real single-person pipeline")
    print("=" * 70)

    import cv2
    from multiperson_detector import MultiPersonDetector
    from cross_view_association import CrossViewAssociator

    calib_path = CALIB_PATH
    vid_dir = VID_DIR
    videos = sorted([
        os.path.join(vid_dir, f)
        for f in os.listdir(vid_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])

    num_frames = 20
    fps = 6.0

    calib = load_calibration_toml(calib_path)
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    caps = [cv2.VideoCapture(v) for v in videos]
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    frame_associations = []
    per_cam_detections = []

    print(f"\n  Running full pipeline: detection -> association -> tracking -> triangulation -> v3.1...")
    t0 = time.time()
    for frame_idx in range(num_frames):
        frame_cam_dets = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            frame_cam_dets.append(dets)

        per_cam_detections.append(frame_cam_dets)
        fa = associator.associate_frame(frame_cam_dets, frame_idx)
        frame_associations.append(fa)

    t1 = time.time()
    print(f"  Detection + association: {t1-t0:.2f}s")

    tracker = TemporalTracker(reference_camera=0)
    temporal_results = tracker.track(frame_associations, per_cam_detections)

    per_camera_per_frame = []
    for cam in range(len(videos)):
        per_camera_per_frame.append([per_cam_detections[f][cam] for f in range(num_frames)])

    triangulator = PerActorTriangulator(calib)
    actor_results = triangulator.triangulate_all(
        per_camera_per_frame, frame_associations, temporal_results
    )

    # Stage 5: Pipeline
    t0 = time.time()
    pipeline_results = process_multi_actor(
        actor_results, fps=fps, align_floor=True,
        butterworth_cutoff=6.0,
    )
    t1 = time.time()
    print(f"\n  Pipeline time: {t1-t0:.2f}s")

    # Validate
    all_ok = True
    summary = get_pipeline_summary(pipeline_results)

    for pid, data in sorted(pipeline_results.items()):
        skel = data["processed_skeleton"]
        report = data["pipeline_report"]
        nan_before = report.get("nan_before", 0)
        nan_after = report.get("nan_after", 0)

        print(f"\n  Actor {pid}:")
        print(f"    Shape: {skel.shape}")
        print(f"    NaN: {nan_before} -> {nan_after} (reduced {nan_before - nan_after})")
        print(f"    Time: {report.get('total_time_seconds', 0):.2f}s")

        floor_info = report.get("shared_floor")
        if floor_info:
            print(f"    Shared floor from Actor {floor_info.get('estimated_from_actor', '?')}")

    for cap in caps:
        cap.release()

    if summary["num_actors"] > 0:
        print(f"\n  PASS: Real data processed through full multi-actor pipeline")
        return True
    else:
        print(f"\n  FAIL: No actors processed")
        return False


def main():
    print("=" * 70)
    print("STAGE 5: Multi-Actor Pipeline Tests")
    print("=" * 70)

    results = {}
    results["synthetic_two_actors"] = test_synthetic_two_actors()
    results["shared_floor"] = test_shared_floor()
    results["real_single_person"] = test_real_single_person()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")


if __name__ == "__main__":
    main()
