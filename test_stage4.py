"""
Stage 4 Test: Per-Actor Triangulation.

Tests:
1. Synthetic: 2 actors, 30 frames, verify 3D output shapes and reprojection
2. Real data: single-person synchronized cameras → 1 actor, validate 3D coords

Output: per-frame tables with 3D coordinate stats and reprojection errors.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_loader import load_calibration_toml
from cross_view_association import FrameAssociation
from temporal_tracker import TemporalTracker, TemporalAssociation
from per_actor_triangulation import (
    PerActorTriangulator,
    ActorTriangulationResult,
    triangulate_multi_actor,
    NUM_KEYPOINTS,
)
from multiperson_detector import PersonDetection
from test_config import CALIB_PATH, VID_DIR


# ── Mock data helpers ──────────────────────────────────────────────────

class MockDetection:
    """Minimal PersonDetection mock for testing."""
    def __init__(self, person_id, center_x, center_y, cam_idx=0):
        self.person_id = person_id
        self.keypoints_133 = np.zeros((133, 3))
        # 17 body keypoints with realistic offsets from center
        body_offsets = np.array([
            [0, -40],    # 0: nose
            [-5, -35],   # 1: left_eye_inner
            [-10, -35],  # 2: left_eye
            [-15, -35],  # 3: left_eye_outer
            [5, -35],    # 4: right_eye_inner
            [10, -35],   # 5: right_eye
            [15, -35],   # 6: right_eye_outer
            [-20, -30],  # 7: left_ear
            [20, -30],   # 8: right_ear
            [-10, -25],  # 9: mouth_left
            [10, -25],   # 10: mouth_right
            [-25, -10],  # 11: left_shoulder
            [25, -10],   # 12: right_shoulder
            [-35, 10],   # 13: left_elbow
            [35, 10],    # 14: right_elbow
            [-40, 30],   # 15: left_wrist
            [40, 30],    # 16: right_wrist
        ])
        noise = np.random.randn(17, 2) * 1.5
        # Add slight camera-dependent offset (simulates different viewpoints)
        cam_offset = np.array([cam_idx * 3.0, cam_idx * 1.5])
        for i in range(17):
            self.keypoints_133[i, 0] = center_x + body_offsets[i, 0] + noise[i, 0] + cam_offset[0]
            self.keypoints_133[i, 1] = center_y + body_offsets[i, 1] + noise[i, 1] + cam_offset[1]
            self.keypoints_133[i, 2] = 0.7 + np.random.rand() * 0.2  # confidence
        # Remaining 116 keypoints: low confidence (not visible)
        self.keypoints_133[17:, 2] = 0.0
        self.bbox = np.array([center_x - 50, center_y - 60, center_x + 50, center_y + 60])
        self.confidence = 0.75


def _make_frame_association(frame_idx, actor_configs, num_cameras=3):
    """Create synthetic FrameAssociation."""
    detection_to_actor = {}
    actor_memberships = {}
    for actor_local_id, config in enumerate(actor_configs):
        members = []
        for cam in range(num_cameras):
            detection_to_actor[(cam, actor_local_id)] = actor_local_id
            members.append((cam, actor_local_id))
        actor_memberships[actor_local_id] = members
    return FrameAssociation(
        frame_idx=frame_idx,
        num_cameras=num_cameras,
        num_actors=len(actor_configs),
        detection_to_actor=detection_to_actor,
        actor_memberships=actor_memberships,
    )


# ── Tests ──────────────────────────────────────────────────────────────

def test_synthetic_two_actors():
    """Test 1: Two synthetic actors, 30 frames.

    Actor 0: center at (200, 300)
    Actor 1: center at (400, 300)

    Both detected in all 3 cameras for all frames.
    Expected: 2 ActorTriangulationResult objects with valid 3D skeletons.
    """
    print("=" * 70)
    print("TEST 1: Synthetic 2 actors, 30 frames, 3 cameras")
    print("=" * 70)

    calib_path = CALIB_PATH
    calib = load_calibration_toml(calib_path)

    num_frames = 30
    num_cameras = 3

    # Build data structures
    frame_associations = []
    temporal_associations_gen = TemporalTracker(reference_camera=0)
    per_cam_detections = []  # [frame][camera] format

    for frame_idx in range(num_frames):
        actor_configs = [
            {"center": (200, 300)},
            {"center": (400, 300)},
        ]
        fa = _make_frame_association(frame_idx, actor_configs, num_cameras)
        frame_associations.append(fa)

        # Build per-camera detections
        frame_cam_dets = []
        for cam in range(num_cameras):
            cam_dets = [
                MockDetection(0, 200, 300, cam),
                MockDetection(1, 400, 300, cam),
            ]
            frame_cam_dets.append(cam_dets)
        per_cam_detections.append(frame_cam_dets)

    # Run Stage 3 (temporal tracking)
    temporal_results = temporal_associations_gen.track(frame_associations, per_cam_detections)

    # Build [camera][frame] format for triangulator
    per_camera_per_frame = []
    for cam in range(num_cameras):
        cam_frames = [per_cam_detections[f][cam] for f in range(num_frames)]
        per_camera_per_frame.append(cam_frames)

    # Run Stage 4 (triangulation)
    print("\n  Running triangulation...")
    t0 = time.time()
    results = triangulate_multi_actor(
        per_camera_per_frame, frame_associations, temporal_results, calib
    )
    t1 = time.time()
    print(f"  Time: {t1-t0:.4f}s")

    # Validate results
    print(f"\n  Results summary:")
    all_ok = True
    for pid, res in sorted(results.items()):
        skel = res.skeleton_3d
        visible_count = int(np.sum(res.visible_frames))
        valid_3d = np.sum(np.isfinite(skel[:, :, 0]))
        total_pts = skel.shape[0] * skel.shape[1]

        print(f"    Actor {pid}:")
        print(f"      Frames: {res.num_frames}")
        print(f"      Visible: {visible_count}/{res.num_frames}")
        print(f"      Valid 3D points: {valid_3d}/{total_pts}")
        print(f"      Mean reproj error: {res.mean_reprojection_error:.2f}px")
        print(f"      Mean visible kpts/frame: {res.mean_visible_kpts_per_frame:.0f}")

        # Check 3D coords are reasonable (within 1000 units of origin)
        valid_coords = skel[res.visible_frames]
        if valid_coords.size > 0:
            valid_xyz = valid_coords[np.isfinite(valid_coords[:, :, 0])]
            if valid_xyz.size > 0:
                abs_max = np.max(np.abs(valid_xyz))
                print(f"      Max abs coord: {abs_max:.1f}")
                if abs_max > 10000:
                    print(f"      WARNING: coordinates seem too large")
                    all_ok = False

        if visible_count != num_frames:
            print(f"      WARNING: not all frames visible")
            all_ok = False

        if res.mean_reprojection_error > 50:
            print(f"      WARNING: high reprojection error")
            all_ok = False

    # Check that actors are separated in 3D
    if len(results) == 2:
        pid0, pid1 = sorted(results.keys())
        skel0 = results[pid0].skeleton_3d
        skel1 = results[pid1].skeleton_3d

        # Use frame 15 (middle) for comparison
        frame_mid = 15
        valid0 = np.isfinite(skel0[frame_mid, :, 0])
        valid1 = np.isfinite(skel1[frame_mid, :, 0])
        if np.sum(valid0) > 0 and np.sum(valid1) > 0:
            mean0 = np.nanmean(skel0[frame_mid], axis=0)
            mean1 = np.nanmean(skel1[frame_mid], axis=0)
            dist = np.linalg.norm(mean0 - mean1)
            print(f"\n  Actor separation (frame {frame_mid}): {dist:.1f} units")
            if dist < 1.0:
                print(f"  WARNING: actors too close in 3D")
                all_ok = False
            else:
                print(f"  Good: actors are well-separated in 3D space")

    if all_ok:
        print(f"\n  PASS: Both actors triangulated correctly")
    else:
        print(f"\n  PARTIAL: Some issues detected (see warnings above)")

    return all_ok


def test_real_single_person():
    """Test 2: Real single-person synchronized cameras.

    Runs full pipeline: detection → cross-view association → temporal tracking
    → per-actor triangulation on real camera data.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Real single-person data (3 cameras, 20 frames)")
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

    # Load calibration
    calib = load_calibration_toml(calib_path)
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    # Open videos
    caps = [cv2.VideoCapture(v) for v in videos]

    # Run detection + association + temporal tracking
    detector = MultiPersonDetector(device="cpu", mode="balanced")
    frame_associations = []
    per_cam_detections = []  # [frame][camera]

    print(f"\n  Running detection + association on {num_frames} frames...")
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

        if frame_idx % 5 == 0:
            n_dets = [len(d) for d in frame_cam_dets]
            print(f"    Frame {frame_idx}: dets={n_dets}, actors={fa.num_actors}")

    t1 = time.time()
    print(f"  Detection + association: {t1-t0:.2f}s")

    # Temporal tracking
    tracker = TemporalTracker(reference_camera=0)
    temporal_results = tracker.track(frame_associations, per_cam_detections)

    # Build [camera][frame] format
    num_cameras = len(videos)
    per_camera_per_frame = []
    for cam in range(num_cameras):
        cam_frames = [per_cam_detections[f][cam] for f in range(num_frames)]
        per_camera_per_frame.append(cam_frames)

    # Triangulation
    print(f"\n  Running triangulation...")
    t0 = time.time()
    results = triangulate_multi_actor(
        per_camera_per_frame, frame_associations, temporal_results, calib
    )
    t1 = time.time()
    print(f"  Triangulation time: {t1-t0:.4f}s")

    # Validate
    all_ok = True
    for pid, res in sorted(results.items()):
        visible_count = int(np.sum(res.visible_frames))
        valid_3d = np.sum(np.isfinite(res.skeleton_3d[:, :, 0]))
        total_pts = res.num_frames * NUM_KEYPOINTS

        print(f"\n  Actor {pid}:")
        print(f"    Visible: {visible_count}/{res.num_frames} frames")
        print(f"    Valid 3D points: {valid_3d}/{total_pts}")
        print(f"    Mean reproj error: {res.mean_reprojection_error:.2f}px")
        print(f"    Mean visible kpts/frame: {res.mean_visible_kpts_per_frame:.0f}")

        if visible_count == 0:
            print(f"    FAIL: No visible frames")
            all_ok = False

        if np.isfinite(res.mean_reprojection_error) and res.mean_reprojection_error > 100:
            print(f"    WARNING: High reprojection error")
            all_ok = False

    for cap in caps:
        cap.release()

    if all_ok:
        print(f"\n  PASS: Real data triangulation successful")
    else:
        print(f"\n  FAIL: Issues detected")

    return all_ok


def test_reprojection_consistency():
    """Test 3: Reprojection consistency check.

    Triangulates 3D points, then reprojects back to 2D and checks
    that the reprojection error is within acceptable bounds.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Reprojection consistency (synthetic, 10 frames)")
    print("=" * 70)

    calib_path = CALIB_PATH
    calib = load_calibration_toml(calib_path)

    num_frames = 10
    num_cameras = 3

    # Build data with a single actor
    frame_associations = []
    per_cam_detections = []

    for frame_idx in range(num_frames):
        fa = _make_frame_association(frame_idx, [{"center": (300, 250)}], num_cameras)
        frame_associations.append(fa)

        frame_cam_dets = []
        for cam in range(num_cameras):
            frame_cam_dets.append([MockDetection(0, 300, 250, cam)])
        per_cam_detections.append(frame_cam_dets)

    # Temporal tracking
    tracker = TemporalTracker(reference_camera=0)
    temporal_results = tracker.track(frame_associations, per_cam_detections)

    # Build [camera][frame]
    per_camera_per_frame = []
    for cam in range(num_cameras):
        per_camera_per_frame.append([per_cam_detections[f][cam] for f in range(num_frames)])

    # Triangulate
    results = triangulate_multi_actor(
        per_camera_per_frame, frame_associations, temporal_results, calib
    )

    # Check reprojection consistency
    from rtmpose_triangulation import reproject_to_2d

    pid = list(results.keys())[0]
    res = results[pid]
    skel = res.skeleton_3d  # (F, 133, 3)

    print(f"\n  Reprojection analysis for Actor {pid}:")
    all_errors = []
    for frame_idx in range(num_frames):
        if not res.visible_frames[frame_idx]:
            continue

        frame_3d = skel[frame_idx]  # (133, 3)
        valid_mask = np.isfinite(frame_3d[:, 0])
        valid_3d = frame_3d[valid_mask]

        if len(valid_3d) == 0:
            continue

        # Reproject to camera 0
        reproj = reproject_to_2d(valid_3d, PerActorTriangulator(calib).projection_matrices[0])

        # Compare with observed 2D
        cam_dets = per_cam_detections[frame_idx][0]
        if len(cam_dets) > 0:
            observed_2d = cam_dets[0].keypoints_133[valid_mask, :2]
            errors = np.sqrt(np.sum((reproj - observed_2d) ** 2, axis=1))
            all_errors.extend(errors.tolist())

            if frame_idx % 3 == 0:
                print(f"    Frame {frame_idx}: {len(valid_3d)} kpts, "
                      f"mean err={np.mean(errors):.2f}px, max={np.max(errors):.2f}px")

    if all_errors:
        mean_err = np.mean(all_errors)
        max_err = np.max(all_errors)
        p95_err = np.percentile(all_errors, 95)
        print(f"\n  Overall: mean={mean_err:.2f}px, p95={p95_err:.2f}px, max={max_err:.2f}px")

        if mean_err < 50:
            print(f"  PASS: Reprojection errors within acceptable bounds")
            return True
        else:
            print(f"  WARNING: High mean reprojection error")
            return False
    else:
        print(f"  No valid points to analyze")
        return False


def main():
    print("=" * 70)
    print("STAGE 4: Per-Actor Triangulation Tests")
    print("=" * 70)

    results = {}
    results["synthetic_two_actors"] = test_synthetic_two_actors()
    results["reprojection_consistency"] = test_reprojection_consistency()
    results["real_single_person"] = test_real_single_person()

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
