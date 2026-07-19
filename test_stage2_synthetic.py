"""
Create synthetic 2-person multi-camera data for Stage 2 testing.

Takes the existing single-person synchronized videos and creates a synthetic
second person by shifting one camera's detections. This provides ground truth
for validating cross-view association.

The synthetic "second person" is created by:
1. Running detection on the real person in all 3 cameras
2. Creating a shifted version of the detection (offset bbox + offset keypoints)
3. Combining both into a 2-person detection list per camera

This gives us known ground truth: we know exactly which detection maps to
which person across cameras, so we can validate the association.
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


def create_synthetic_person(det: PersonDetection, offset_x: int = 200, offset_y: int = 0) -> PersonDetection:
    """Create a synthetic second person by offsetting a real detection.

    This shifts the bbox and keypoints to simulate a second person standing
    to the right of the real person.
    """
    new_bbox = det.bbox.copy()
    new_bbox[0] += offset_x  # x1
    new_bbox[2] += offset_x  # x2
    new_bbox[1] += offset_y  # y1
    new_bbox[3] += offset_y  # y2

    new_keypoints = det.keypoints_133.copy()
    new_keypoints[:, 0] += offset_x  # x coordinates
    new_keypoints[:, 1] += offset_y  # y coordinates

    # Lower confidence slightly for the synthetic person
    new_confidence = det.confidence * 0.85

    # Rebuild segments
    segments = {}
    from multiperson_detector import RTMPOSE_SEGMENTS
    for name, (start, end) in RTMPOSE_SEGMENTS.items():
        segments[name] = new_keypoints[start:end].copy()

    return PersonDetection(
        person_id=1,
        bbox=new_bbox,
        keypoints_133=new_keypoints,
        confidence=new_confidence,
        segments=segments,
    )


def main():
    from test_config import CALIB_PATH, VID_DIR
    calib_path = CALIB_PATH
    vid_dir = VID_DIR
    videos = sorted([
        os.path.join(vid_dir, f)
        for f in os.listdir(vid_dir)
        if f.endswith(".mp4") and "mediapipe" not in f
    ])

    print("=" * 60)
    print("Stage 2 Test: Synthetic 2-Person Multi-Camera Data")
    print("=" * 60)

    # Load calibration
    print("\n[1] Loading calibration...")
    calib = load_calibration_toml(calib_path)
    print(f"  {len(calib['camera_matrices'])} cameras loaded")

    # Create associator
    associator = CrossViewAssociator(
        camera_matrices=calib['camera_matrices'],
        extrinsic_matrices=calib['extrinsic_matrices'],
        image_sizes=calib['image_sizes'],
        max_epipolar_distance=50.0,
    )

    # Initialize detector
    print("\n[2] Running detection on first frame of each camera...")
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    real_detections = []
    for i, vid_path in enumerate(videos):
        cap = cv2.VideoCapture(vid_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read from {vid_path}")

        dets = detector.detect_frame(frame)
        print(f"  Cam{i}: {len(dets)} real detection(s)")
        for d in dets:
            print(f"    Person {d.person_id}: conf={d.confidence:.3f}, "
                  f"bbox=[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}]")
        real_detections.append(dets)

    # Create synthetic 2-person data
    print("\n[3] Creating synthetic 2-person data...")
    per_camera_detections = []
    for i, dets in enumerate(real_detections):
        if len(dets) == 0:
            per_camera_detections.append([])
            continue

        # Take the first (highest confidence) real detection
        real_person = dets[0]

        # Create synthetic second person with different offsets per camera
        # This simulates a real person at a different 3D position
        if i == 0:
            synth = create_synthetic_person(real_person, offset_x=250, offset_y=0)
        elif i == 1:
            synth = create_synthetic_person(real_person, offset_x=200, offset_y=50)
        else:
            synth = create_synthetic_person(real_person, offset_x=220, offset_y=-30)

        two_persons = [real_person, synth]
        per_camera_detections.append(two_persons)
        print(f"  Cam{i}: 1 real + 1 synthetic = 2 detections")
        print(f"    Real: conf={real_person.confidence:.3f}")
        print(f"    Synthetic: conf={synth.confidence:.3f}")

    # Run cross-view association
    print("\n[4] Running cross-view association...")
    t0 = time.time()
    result = associator.associate_frame(per_camera_detections, frame_idx=0)
    t1 = time.time()

    print(f"  Time: {t1-t0:.4f}s")
    print(f"  Global actors detected: {result.num_actors}")
    print(f"\n  Detection-to-actor mapping:")
    for (cam_idx, person_id), actor_id in sorted(result.detection_to_actor.items()):
        person_type = "REAL " if person_id == 0 else "SYNTH"
        print(f"    Cam{cam_idx} {person_type}P{person_id} -> Actor {actor_id}")

    print(f"\n  Actor memberships:")
    for actor_id, members in sorted(result.actor_memberships.items()):
        member_str = ", ".join(f"Cam{c}P{p}" for c, p in members)
        print(f"    Actor {actor_id}: [{member_str}]")

    # Show pair-wise details
    print(f"\n  Pair-wise assignments:")
    for pa in result.pair_assignments:
        matched = sum(1 for v in pa.assignments.values() if v != -1)
        total = len(pa.assignments)
        print(f"    Cam{pa.camera_a_idx} <-> Cam{pa.camera_b_idx}: {matched}/{total} matched")
        for pid_a, pid_b in pa.assignments.items():
            if pid_b != -1:
                cost = pa.cost_matrix[pid_a, pid_b]
                print(f"      P{pid_a} -> P{pid_b} (epipolar dist: {cost:.2f}px)")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Check: real person (P0) should be Actor 0 across all cameras
    real_actor_ids = set()
    for (cam_idx, person_id), actor_id in result.detection_to_actor.items():
        if person_id == 0:  # real person
            real_actor_ids.add(actor_id)

    # Check: synthetic person (P1) should be Actor 1 across all cameras
    synth_actor_ids = set()
    for (cam_idx, person_id), actor_id in result.detection_to_actor.items():
        if person_id == 1:  # synthetic person
            synth_actor_ids.add(actor_id)

    print(f"  Real person (P0) mapped to actors: {real_actor_ids}")
    print(f"  Synthetic person (P1) mapped to actors: {synth_actor_ids}")

    real_consistent = len(real_actor_ids) == 1
    synth_consistent = len(synth_actor_ids) == 1
    different_actors = real_actor_ids != synth_actor_ids

    if real_consistent and synth_consistent and different_actors:
        print(f"\n  PASS: Real person -> Actor {real_actor_ids.pop()}, "
              f"Synthetic -> Actor {synth_actor_ids.pop()}")
    else:
        print(f"\n  FAIL: Real consistent={real_consistent}, "
              f"Synth consistent={synth_consistent}, Different={different_actors}")


if __name__ == "__main__":
    main()
