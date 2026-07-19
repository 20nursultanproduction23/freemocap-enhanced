"""
Stage 2 Test: Cross-View Association on synchronized cameras.

Tests the full pipeline: calibration loading → multi-person detection →
cross-view association with epipolar geometry + Hungarian algorithm.

First tests on single-person data (trivial: 1 detection = 1 global actor),
then will test on multi-person data when available.
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_loader import load_calibration_toml
from multiperson_detector import MultiPersonDetector
from cross_view_association import CrossViewAssociator


def load_first_frame(video_path: str) -> np.ndarray:
    """Load first frame from video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from {video_path}")
    return frame


def main():
    # Paths
    from test_config import CALIB_PATH, VID_DIR
    calib_path = CALIB_PATH
    vid_dir = VID_DIR
    videos = sorted([
        os.path.join(vid_dir, f)
        for f in os.listdir(vid_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])

    print("=" * 60)
    print("Stage 2: Cross-View Association Test")
    print("=" * 60)

    # Step 1: Load calibration
    print("\n[1] Loading calibration...")
    calib = load_calibration_toml(calib_path)
    print(f"  Cameras: {len(calib['camera_matrices'])}")
    for i, name in enumerate(calib['names']):
        K = calib['camera_matrices'][i]
        E = calib['extrinsic_matrices'][i]
        sz = calib['image_sizes'][i]
        print(f"  Cam{i}: {name}")
        print(f"    K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print(f"    Image size: {sz}")
        print(f"    Extrinsic R trace: {np.trace(E[:3,:3]):.3f}")
        print(f"    Extrinsic t: [{E[0,3]:.1f}, {E[1,3]:.1f}, {E[2,3]:.1f}]")

    # Step 2: Create associator
    print("\n[2] Creating CrossViewAssociator...")
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )
    print(f"  Precomputed {len(associator._fundamental_matrices)} fundamental matrices")

    # Show fundamental matrices
    for (i, j), F in associator._fundamental_matrices.items():
        if i < j:
            print(f"  F_{i}{j}: norm={np.linalg.norm(F):.6f}, max={np.abs(F).max():.6f}")

    # Step 3: Run multi-person detection on first frame of each camera
    print("\n[3] Running multi-person detection on first frame...")
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    per_camera_detections = []
    for i, vid_path in enumerate(videos):
        frame = load_first_frame(vid_path)
        print(f"  Frame shape: {frame.shape}")
        t0 = time.time()
        dets = detector.detect_frame(frame)
        t1 = time.time()
        print(f"  Cam{i}: {len(dets)} person(s) detected in {t1-t0:.2f}s")
        for d in dets:
            print(f"    Person {d.person_id}: conf={d.confidence:.3f}, bbox={d.bbox.astype(int).tolist()}")
        per_camera_detections.append(dets)

    # Step 4: Run cross-view association
    print("\n[4] Running cross-view association...")
    t0 = time.time()
    result = associator.associate_frame(per_camera_detections, frame_idx=0)
    t1 = time.time()

    print(f"  Time: {t1-t0:.4f}s")
    print(f"  Global actors detected: {result.num_actors}")
    print(f"  Detection-to-actor mapping:")
    for (cam_idx, person_id), actor_id in sorted(result.detection_to_actor.items()):
        print(f"    Cam{cam_idx} Person{person_id} -> Actor {actor_id}")

    print(f"  Actor memberships:")
    for actor_id, members in sorted(result.actor_memberships.items()):
        member_str = ", ".join(f"Cam{c}P{p}" for c, p in members)
        print(f"    Actor {actor_id}: [{member_str}]")

    # Step 5: Show pair-wise assignment details
    print(f"\n  Pair-wise assignments:")
    for pa in result.pair_assignments:
        matched = sum(1 for v in pa.assignments.values() if v != -1)
        total = len(pa.assignments)
        print(f"    Cam{pa.camera_a_idx} <-> Cam{pa.camera_b_idx}: "
              f"{matched}/{total} matched, cost={pa.total_cost:.2f}, "
              f"matched={pa.matched}")
        if pa.matched:
            for pid_a, pid_b in pa.assignments.items():
                if pid_b != -1:
                    print(f"      P{pid_a} -> P{pid_b} (cost={pa.cost_matrix[pid_a, pid_b]:.2f})")

    # Step 6: Test on more frames (first 10)
    print("\n[5] Testing on first 10 frames...")
    vid_caps = [cv2.VideoCapture(v) for v in videos]

    frame_results = []
    for frame_idx in range(10):
        per_cam_dets = []
        for cap in vid_caps:
            ret, frame = cap.read()
            if not ret:
                per_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            per_cam_dets.append(dets)

        result = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)
        frame_results.append(result)

        n_dets = [len(d) for d in per_cam_dets]
        print(f"  Frame {frame_idx}: dets={n_dets}, actors={result.num_actors}, "
              f"mapping={dict(result.detection_to_actor)}")

    for cap in vid_caps:
        cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    actor_counts = [r.num_actors for r in frame_results]
    print(f"Frames tested: {len(frame_results)}")
    print(f"Actors per frame: {actor_counts}")
    print(f"All same count: {len(set(actor_counts)) == 1}")

    if len(set(actor_counts)) == 1:
        print(f"\nPASS: Consistent {actor_counts[0]} actor(s) across all frames")
    else:
        print(f"\nWARNING: Inconsistent actor counts: {actor_counts}")


if __name__ == "__main__":
    main()
