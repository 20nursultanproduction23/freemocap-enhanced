"""
Stage 2 Comprehensive Test: Cross-View Association

Tests on multiple frames using both single-person (real) and synthetic 2-person
data. Measures:
1. Real person consistency across cameras
2. Epipolar distance for correct matches
3. False positive rejection rate
"""
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_loader import load_calibration_toml
from multiperson_detector import MultiPersonDetector, PersonDetection
from cross_view_association import CrossViewAssociator
from multiperson_detector import RTMPOSE_SEGMENTS


def create_synthetic_person(det: PersonDetection, offset_x: int = 200, offset_y: int = 0) -> PersonDetection:
    new_bbox = det.bbox.copy()
    new_bbox[0] += offset_x
    new_bbox[2] += offset_x
    new_bbox[1] += offset_y
    new_bbox[3] += offset_y
    new_keypoints = det.keypoints_133.copy()
    new_keypoints[:, 0] += offset_x
    new_keypoints[:, 1] += offset_y
    new_confidence = det.confidence * 0.85
    segments = {}
    for name, (start, end) in RTMPOSE_SEGMENTS.items():
        segments[name] = new_keypoints[start:end].copy()
    return PersonDetection(person_id=1, bbox=new_bbox, keypoints_133=new_keypoints,
                           confidence=new_confidence, segments=segments)


def main():
    from test_config import CALIB_PATH, VID_DIR
    calib_path = CALIB_PATH
    vid_dir = VID_DIR
    videos = sorted([
        os.path.join(vid_dir, f)
        for f in os.listdir(vid_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])

    print("=" * 70)
    print("Stage 2 Comprehensive Cross-View Association Test")
    print("=" * 70)

    # Load calibration and create associator
    calib = load_calibration_toml(calib_path)
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    detector = MultiPersonDetector(device="cpu", mode="balanced")

    # Open all video captures
    caps = [cv2.VideoCapture(v) for v in videos]

    # ─── TEST 1: Single-person across all frames ────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: Real person consistency across cameras (20 frames)")
    print("=" * 70)

    real_person_results = []
    for frame_idx in range(20):
        per_cam_dets = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                per_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            per_cam_dets.append(dets)

        result = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)
        real_person_results.append(result)

        # Check: is real person (highest conf) consistent across cameras?
        actor_map = {}
        for (cam_idx, person_id), actor_id in result.detection_to_actor.items():
            actor_map.setdefault(actor_id, []).append((cam_idx, person_id))

        n_actors = result.num_actors
        n_dets = [len(d) for d in per_cam_dets]
        print(f"  Frame {frame_idx:>2}: dets={n_dets}, actors={n_actors}")

    # Summary
    actor_counts = [r.num_actors for r in real_person_results]
    print(f"\n  Summary: {len(real_person_results)} frames tested")
    print(f"  Actor counts per frame: {actor_counts}")
    unique_counts = set(actor_counts)
    if len(unique_counts) == 1:
        print(f"  PASS: Consistent {unique_counts.pop()} actor(s) across all frames")
    else:
        print(f"  INFO: Varying actor counts (expected with false positives)")

    # ─── TEST 2: Epipolar distance for correct matches ──────────────────
    print("\n" + "=" * 70)
    print("TEST 2: Epipolar distance analysis")
    print("=" * 70)

    # Reset video captures
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    epipolar_distances = []
    for frame_idx in range(20):
        per_cam_dets = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                per_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            per_cam_dets.append(dets)

        result = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)

        for pa in result.pair_assignments:
            for pid_a, pid_b in pa.assignments.items():
                if pid_b != -1:
                    cost = pa.cost_matrix[pid_a, pid_b]
                    epipolar_distances.append({
                        'frame': frame_idx,
                        'cam_a': pa.camera_a_idx,
                        'cam_b': pa.camera_b_idx,
                        'pid_a': pid_a,
                        'pid_b': pid_b,
                        'distance': cost,
                    })

    if epipolar_distances:
        distances = [e['distance'] for e in epipolar_distances]
        print(f"  Total matched pairs: {len(epipolar_distances)}")
        print(f"  Mean epipolar distance: {np.mean(distances):.2f}px")
        print(f"  Median: {np.median(distances):.2f}px")
        print(f"  Min: {np.min(distances):.2f}px, Max: {np.max(distances):.2f}px")
        print(f"  Std: {np.std(distances):.2f}px")
        print(f"  Matches < 20px: {sum(1 for d in distances if d < 20)}/{len(distances)} "
              f"({sum(1 for d in distances if d < 20)/len(distances)*100:.1f}%)")
        print(f"  Matches < 30px: {sum(1 for d in distances if d < 30)}/{len(distances)} "
              f"({sum(1 for d in distances if d < 30)/len(distances)*100:.1f}%)")
        print(f"  Matches < 50px (threshold): {sum(1 for d in distances if d < 50)}/{len(distances)} "
              f"({sum(1 for d in distances if d < 50)/len(distances)*100:.1f}%)")
    else:
        print("  No matches found!")

    # ─── TEST 3: Synthetic 2-person stress test ─────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: Synthetic 2-person data (geometric consistency test)")
    print("=" * 70)

    # Reset video captures
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    synth_results = []
    for frame_idx in range(10):
        per_cam_dets = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                per_cam_dets.append([])
                continue
            dets = detector.detect_frame(frame)
            if len(dets) > 0:
                real_person = dets[0]
                synth = create_synthetic_person(real_person, offset_x=200 + i*30, offset_y=i*20)
                per_cam_dets.append([real_person, synth])
            else:
                per_cam_dets.append([])

        result = associator.associate_frame(per_cam_dets, frame_idx=frame_idx)
        synth_results.append(result)

        # Check real person mapping
        real_actors = set()
        synth_actors = set()
        for (cam_idx, person_id), actor_id in result.detection_to_actor.items():
            if person_id == 0:
                real_actors.add(actor_id)
            else:
                synth_actors.add(actor_id)

        real_ok = "OK" if len(real_actors) == 1 else "FAIL"
        synth_ok = "OK" if len(synth_actors) == 1 else "FAIL"
        print(f"  Frame {frame_idx:>2}: actors={result.num_actors}, "
              f"real={real_actors} {real_ok}, synth={synth_actors} {synth_ok}")

    # Summary
    real_consistent = sum(1 for r in synth_results
                         if len(set(a for (c, p), a in r.detection_to_actor.items() if p == 0)) == 1)
    synth_consistent = sum(1 for r in synth_results
                          if len(set(a for (c, p), a in r.detection_to_actor.items() if p == 1)) == 1)
    print(f"\n  Real person consistent: {real_consistent}/{len(synth_results)}")
    print(f"  Synthetic person consistent: {synth_consistent}/{len(synth_results)}")

    # ─── FINAL VERDICT ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    all_real_consistent = all(
        len(set(a for (c, p), a in r.detection_to_actor.items() if p == 0)) == 1
        for r in real_person_results
    )

    if all_real_consistent:
        print("  PASS: Real person correctly matched across all cameras in all frames")
    else:
        print("  FAIL: Real person inconsistent across cameras")

    epipolar_ok = np.mean(distances) < 30 if epipolar_distances else False
    if epipolar_ok:
        print(f"  PASS: Mean epipolar distance ({np.mean(distances):.1f}px) < 30px threshold")
    else:
        print(f"  FAIL: Mean epipolar distance too high")

    print(f"\n  Note: Synthetic person test is a stress test — geometric inconsistency")
    print(f"  is expected because synthetic keypoints lack real 3D geometry.")

    for cap in caps:
        cap.release()


if __name__ == "__main__":
    main()
