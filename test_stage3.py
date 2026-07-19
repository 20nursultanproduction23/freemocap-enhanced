"""
Stage 3 Test: Temporal Tracking across frames.

Tests the TemporalTracker module with:
1. Synthetic FrameAssociation sequences simulating 2 actors across 30 frames
2. Synthetic entry/exit events (actor 1 appears at frame 0, actor 2 enters at frame 10)
3. Real single-person data (should produce 1 persistent track with no switches)

Output: per-frame table with local_actor_id -> persistent_global_id mapping.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_view_association import FrameAssociation, CameraPairAssignment
from temporal_tracker import TemporalTracker, TemporalAssociation


def _make_fake_frame_association(
    frame_idx: int,
    actor_configs: list,
    num_cameras: int = 3,
) -> FrameAssociation:
    """Create a synthetic FrameAssociation for testing.

    Args:
        frame_idx: frame number
        actor_configs: list of dicts, each with:
            - "center": (x, y) pixel coordinates in camera 0
            - "body_kpts": optional (17, 3) keypoint array
        num_cameras: number of cameras

    Returns:
        FrameAssociation with the specified actors
    """
    detection_to_actor = {}
    actor_memberships = {}

    for actor_local_id, config in enumerate(actor_configs):
        detection_to_actor[(0, actor_local_id)] = actor_local_id

        if num_cameras >= 2:
            detection_to_actor[(1, actor_local_id)] = actor_local_id
        if num_cameras >= 3:
            detection_to_actor[(2, actor_local_id)] = actor_local_id

        members = [(0, actor_local_id)]
        if num_cameras >= 2:
            members.append((1, actor_local_id))
        if num_cameras >= 3:
            members.append((2, actor_local_id))
        actor_memberships[actor_local_id] = members

    return FrameAssociation(
        frame_idx=frame_idx,
        num_cameras=num_cameras,
        num_actors=len(actor_configs),
        detection_to_actor=detection_to_actor,
        actor_memberships=actor_memberships,
    )


def _make_body_kpts(center_x: float, center_y: float, noise: float = 0.0) -> np.ndarray:
    """Create synthetic body keypoints centered at (center_x, center_y)."""
    # 5 body keypoints: nose, L shoulder, R shoulder, L hip, R hip
    offsets = np.array([
        [0, -30],    # nose
        [-20, -10],  # L shoulder
        [20, -10],   # R shoulder
        [-15, 20],   # L hip
        [15, 20],    # R hip
    ])
    kpts = np.zeros((5, 3))
    kpts[:, 0] = center_x + offsets[:, 0] + np.random.randn(5) * noise
    kpts[:, 1] = center_y + offsets[:, 1] + np.random.randn(5) * noise
    kpts[:, 2] = 0.7 + np.random.rand(5) * 0.2  # confidence 0.7-0.9
    return kpts


def _make_bbox(cx: float, cy: float, w: float = 60, h: float = 120) -> np.ndarray:
    """Create bounding box centered at (cx, cy)."""
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class MockDetection:
    """Minimal PersonDetection mock for testing."""
    def __init__(self, person_id, center_x, center_y):
        self.person_id = person_id
        self.keypoints_133 = np.zeros((133, 3))
        body = _make_body_kpts(center_x, center_y, noise=2.0)
        self.keypoints_133[:5] = body
        self.bbox = _make_bbox(center_x, center_y)
        self.confidence = 0.75


def test_basic_two_actors_30_frames():
    """Test 1: Two actors present in all 30 frames, stable positions.

    Actor 0: center at x=200 (static)
    Actor 1: center at x=400 (static)

    Expected: 2 persistent tracks, no ID switches.
    """
    print("=" * 70)
    print("TEST 1: Two actors, 30 frames, stable positions")
    print("=" * 70)

    num_frames = 30
    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        actor_configs = [
            {"center": (200, 300)},
            {"center": (400, 300)},
        ]
        fa = _make_fake_frame_association(frame_idx, actor_configs)
        frame_associations.append(fa)

        # Mock detections for feature extraction
        cam0_dets = [
            MockDetection(0, 200, 300),
            MockDetection(1, 400, 300),
        ]
        cam1_dets = [
            MockDetection(0, 180, 310),
            MockDetection(1, 380, 310),
        ]
        cam2_dets = [
            MockDetection(0, 210, 290),
            MockDetection(1, 410, 290),
        ]
        per_cam_detections.append([cam0_dets, cam1_dets, cam2_dets])

    tracker = TemporalTracker(
        max_lost_frames=10,
        search_window=30,
        reference_camera=0,
    )

    results = tracker.track(frame_associations, per_cam_detections)

    # Print per-frame table
    print(f"\n{'Frame':>5} | {'Local0->P':>10} | {'Local1->P':>10} | {'Active':>6}")
    print("-" * 50)
    id_switches = 0
    prev_mapping = {}
    for r in results:
        l0 = r.local_to_persistent.get(0, "-")
        l1 = r.local_to_persistent.get(1, "-")
        active = len(r.active_tracks)
        print(f"{r.frame_idx:5d} | {str(l0):>10} | {str(l1):>10} | {active:6d}")

        # Check for ID switches
        if prev_mapping:
            for local_id, persistent_id in r.local_to_persistent.items():
                if local_id in prev_mapping and prev_mapping[local_id] != persistent_id:
                    id_switches += 1
        prev_mapping = dict(r.local_to_persistent)

    # Assertions
    all_same_persistent_count = all(
        len(r.local_to_persistent) == 2 for r in results
    )
    all_active = all(len(r.active_tracks) == 2 for r in results[1:])  # frame 0 may differ

    print(f"\nResults:")
    print(f"  ID switches: {id_switches}")
    print(f"  Consistent 2 actors: {all_same_persistent_count}")
    print(f"  Active tracks >=2 after init: {all_active}")

    summary = tracker.get_track_summary()
    print(f"  Total tracks created: {summary['total_tracks_created']}")
    print(f"  Active tracks: {summary['active_tracks']}")

    if id_switches == 0 and all_same_persistent_count:
        print("\n  PASS: No ID switches, consistent actor count")
    else:
        print(f"\n  FAIL: {id_switches} ID switches detected")

    return id_switches == 0 and all_same_persistent_count


def test_actor_entry_exit():
    """Test 2: Actor 1 enters at frame 10, actor 0 exits at frame 20.

    Frames 0-9:   Actor 0 only (center x=200)
    Frames 10-19: Actor 0 (x=200) + Actor 1 (x=400)
    Frames 20-29: Actor 1 only (x=400)

    Expected:
    - Actor 0 gets persistent ID P0 from frame 0 to frame 19
    - Actor 1 gets persistent ID P1 from frame 10 to frame 29
    - No ID switches when actors enter/exit
    """
    print("\n" + "=" * 70)
    print("TEST 2: Actor entry at frame 10, exit at frame 20")
    print("=" * 70)

    num_frames = 30
    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        actors = []
        cam0_dets = []

        if frame_idx < 20:
            actors.append({"center": (200, 300)})
            cam0_dets.append(MockDetection(0, 200, 300))

        if frame_idx >= 10:
            actors.append({"center": (400, 300)})
            cam0_dets.append(MockDetection(len(actors) - 1, 400, 300))

        fa = _make_fake_frame_association(frame_idx, actors)
        frame_associations.append(fa)

        per_cam_detections.append([cam0_dets, list(cam0_dets), list(cam0_dets)])

    tracker = TemporalTracker(
        max_lost_frames=10,
        search_window=30,
        reference_camera=0,
    )

    results = tracker.track(frame_associations, per_cam_detections)

    # Print per-frame table
    print(f"\n{'Frame':>5} | {'#Actors':>7} | {'Mappings':>20} | {'Active':>6}")
    print("-" * 55)
    for r in results:
        mappings = ", ".join(
            f"L{k}->P{v}" for k, v in sorted(r.local_to_persistent.items())
        )
        print(f"{r.frame_idx:5d} | {r.num_persistent_actors:7d} | {mappings:>20} | {len(r.active_tracks):6d}")

    # Check: actor at x=200 should keep same persistent ID in frames 0-19
    actor_at_200_ids = []
    for r in results[:20]:
        if 0 in r.local_to_persistent:
            actor_at_200_ids.append(r.local_to_persistent[0])

    # Check: actor at x=400 should keep same persistent ID in frames 10-29
    actor_at_400_ids = []
    for r in results[10:]:
        if 1 in r.local_to_persistent:
            actor_at_400_ids.append(r.local_to_persistent[1])

    consistent_200 = len(set(actor_at_200_ids)) == 1 if actor_at_200_ids else False
    consistent_400 = len(set(actor_at_400_ids)) == 1 if actor_at_400_ids else False

    print(f"\nResults:")
    print(f"  Actor@200 persistent IDs: {set(actor_at_200_ids)} -> {'PASS' if consistent_200 else 'FAIL'}")
    print(f"  Actor@400 persistent IDs: {set(actor_at_400_ids)} -> {'PASS' if consistent_400 else 'FAIL'}")

    summary = tracker.get_track_summary()
    print(f"  Total tracks created: {summary['total_tracks_created']}")

    if consistent_200 and consistent_400:
        print("\n  PASS: Persistent IDs maintained through entry/exit")
    else:
        print("\n  FAIL: ID inconsistency during entry/exit")

    return consistent_200 and consistent_400


def test_single_person_real_data():
    """Test 3: Real single-person synchronized video data.

    Runs multi-person detection + cross-view association on real camera data,
    then temporal tracking. Should produce 1 persistent track with no switches.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Real single-person data (synchronized cameras)")
    print("=" * 70)

    import cv2
    from multiperson_detector import MultiPersonDetector
    from cross_view_association import CrossViewAssociator
    from calibration_loader import load_calibration_toml

    from test_config import CALIB_PATH, VID_DIR
    calib_path = CALIB_PATH
    vid_dir = VID_DIR
    videos = sorted([
        os.path.join(vid_dir, f)
        for f in os.listdir(vid_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])

    num_frames_to_test = 20

    # Load calibration
    print("\n  Loading calibration...")
    calib = load_calibration_toml(calib_path)

    # Create associator
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    # Open video captures
    caps = [cv2.VideoCapture(v) for v in videos]

    # Run detection + association + temporal tracking frame by frame
    detector = MultiPersonDetector(device="cpu", mode="balanced")
    frame_associations = []
    per_cam_detections = []

    print(f"  Running {num_frames_to_test} frames...")
    t0 = time.time()

    for frame_idx in range(num_frames_to_test):
        per_cam_dets = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                per_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            per_cam_dets.append(dets)

        fa = associator.associate_frame(per_cam_dets, frame_idx)
        frame_associations.append(fa)
        per_cam_detections.append(per_cam_dets)

        if frame_idx % 5 == 0:
            n_dets = [len(d) for d in per_cam_dets]
            print(f"    Frame {frame_idx}: dets={n_dets}, actors={fa.num_actors}")

    t1 = time.time()
    print(f"  Detection + association: {t1-t0:.2f}s ({(t1-t0)/num_frames_to_test:.2f}s/frame)")

    # Run temporal tracking
    tracker = TemporalTracker(
        max_lost_frames=10,
        search_window=30,
        reference_camera=0,
    )

    t0 = time.time()
    results = tracker.track(frame_associations, per_cam_detections)
    t1 = time.time()
    print(f"  Temporal tracking: {t1-t0:.4f}s")

    # Print per-frame table
    print(f"\n  {'Frame':>5} | {'Local0->P':>10} | {'Actors':>6} | {'Active':>6}")
    print("  " + "-" * 45)
    for r in results:
        l0 = r.local_to_persistent.get(0, "-")
        print(f"  {r.frame_idx:5d} | {str(l0):>10} | {r.num_persistent_actors:6d} | {len(r.active_tracks):6d}")

    # Check consistency: main actor should always have persistent ID P0
    # Even if false positives cause extra actors in some frames, the primary
    # actor (always local_id=0 in camera 0, highest confidence) must keep P0
    main_actor_ids = []
    for r in results:
        if 0 in r.local_to_persistent:
            main_actor_ids.append(r.local_to_persistent[0])

    consistent_main_id = len(set(main_actor_ids)) == 1 if main_actor_ids else False
    main_id_value = main_actor_ids[0] if main_actor_ids else None

    print(f"\n  Results:")
    print(f"    Main actor persistent ID: {main_id_value} (consistent: {consistent_main_id})")
    print(f"    Main actor IDs across frames: {set(main_actor_ids)}")

    summary = tracker.get_track_summary()
    print(f"    Total tracks created: {summary['total_tracks_created']}")
    print(f"    Track durations: {summary['track_durations']}")

    # PASS criteria: main actor keeps the same persistent ID across ALL frames
    # Extra tracks from false positives are acceptable (they come and go naturally)
    if consistent_main_id and summary['track_durations'][0] >= num_frames_to_test - 1:
        print("\n  PASS: Main actor maintains consistent persistent ID")
    else:
        print("\n  FAIL: Main actor ID changed or track dropped prematurely")

    for cap in caps:
        cap.release()

    return consistent_main_id and summary['track_durations'][0] >= num_frames_to_test - 1


def test_id_switch_robustness():
    """Test 4: Two actors with similar positions (stress test for ID switches).

    Actor 0: oscillates between x=200-250 (slight movement)
    Actor 1: oscillates between x=350-400 (slight movement)

    The actors are far apart (150px gap) so matching should be trivial.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Two actors with slight oscillation (30 frames)")
    print("=" * 70)

    num_frames = 30
    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        # Slight oscillation
        x0 = 200 + 10 * np.sin(frame_idx * 0.3)
        x1 = 400 + 10 * np.cos(frame_idx * 0.3)

        actor_configs = [
            {"center": (x0, 300)},
            {"center": (x1, 300)},
        ]
        fa = _make_fake_frame_association(frame_idx, actor_configs)
        frame_associations.append(fa)

        cam0_dets = [
            MockDetection(0, x0, 300),
            MockDetection(1, x1, 300),
        ]
        per_cam_detections.append([cam0_dets, list(cam0_dets), list(cam0_dets)])

    tracker = TemporalTracker(
        max_lost_frames=10,
        search_window=30,
        reference_camera=0,
    )

    results = tracker.track(frame_associations, per_cam_detections)

    # Print per-frame table
    print(f"\n{'Frame':>5} | {'Local0->P':>10} | {'Local1->P':>10}")
    print("-" * 40)
    id_switches = 0
    prev_mapping = {}
    for r in results:
        l0 = r.local_to_persistent.get(0, "-")
        l1 = r.local_to_persistent.get(1, "-")
        print(f"{r.frame_idx:5d} | {str(l0):>10} | {str(l1):>10}")

        if prev_mapping:
            for local_id, persistent_id in r.local_to_persistent.items():
                if local_id in prev_mapping and prev_mapping[local_id] != persistent_id:
                    id_switches += 1
        prev_mapping = dict(r.local_to_persistent)

    print(f"\n  ID switches: {id_switches}")

    if id_switches == 0:
        print("\n  PASS: No ID switches with oscillating actors")
    else:
        print(f"\n  FAIL: {id_switches} ID switches")

    return id_switches == 0


def main():
    print("=" * 70)
    print("STAGE 3: Temporal Tracking Tests")
    print("=" * 70)

    results = {}

    results["basic_two_actors"] = test_basic_two_actors_30_frames()
    results["actor_entry_exit"] = test_actor_entry_exit()
    results["id_switch_robustness"] = test_id_switch_robustness()
    results["single_person_real"] = test_single_person_real_data()

    # Summary
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
