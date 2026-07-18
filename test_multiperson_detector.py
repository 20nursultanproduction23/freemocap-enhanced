"""
Tests for Stage 1: Multi-Person Detector

Tests:
  1. Module imports and dataclass creation
  2. Output format validation
  3. Single-person detection on real test data
  4. Detection rate ≥90% on real data (single person expected)
  5. Multi-person parsing with synthetic data
  6. aggregate_detections summary function
  7. Edge cases: empty frames, no detection
"""
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiperson_detector import (
    MultiPersonDetector,
    PersonDetection,
    aggregate_detections,
    RTMPOSE_SEGMENTS,
    NUM_KEYPOINTS,
)


TEST_DATA_DIR = r"D:\freemocap_test_data\freemocap_test_data"
VIDEO_DIR = os.path.join(TEST_DATA_DIR, "synchronized_videos")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results_stage1.json")


def test_01_imports_and_dataclass():
    """Test 1: Module imports and PersonDetection dataclass."""
    # Test imports
    from multiperson_detector import MultiPersonDetector, PersonDetection
    from multiperson_detector import RTMPOSE_SEGMENTS, NUM_KEYPOINTS
    from multiperson_detector import aggregate_detections

    # Test dataclass creation with synthetic data
    bbox = np.array([100, 100, 300, 400], dtype=np.float64)
    kpts = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float64)
    kpts[:, 2] = 0.8  # scores

    det = PersonDetection(
        person_id=0,
        bbox=bbox,
        keypoints_133=kpts,
        confidence=0.85,
        segments={},
    )

    assert det.person_id == 0
    assert det.confidence == 0.85
    assert det.bbox.shape == (4,)
    assert det.keypoints_133.shape == (133, 3)

    # Test to_dict
    d = det.to_dict()
    assert isinstance(d, dict)
    assert d["person_id"] == 0
    assert d["confidence"] == 0.85
    assert d["num_keypoints"] == 133

    print("  PASS: imports and dataclass")
    return True


def test_02_output_format():
    """Test 2: Output format validation — segments split correctly."""
    bbox = np.array([50, 50, 200, 350], dtype=np.float64)
    kpts = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float64)
    kpts[:, 2] = np.linspace(0.5, 1.0, NUM_KEYPOINTS)

    det = PersonDetection(
        person_id=0,
        bbox=bbox,
        keypoints_133=kpts,
        confidence=0.9,
        segments={},
    )

    # Verify segment boundaries
    for name, (start, end) in RTMPOSE_SEGMENTS.items():
        assert end - start > 0, f"Segment {name} has zero length"
        assert start >= 0, f"Segment {name} starts at {start}"
        assert end <= NUM_KEYPOINTS, f"Segment {name} ends at {end} > {NUM_KEYPOINTS}"

    # Verify total coverage
    total_kpts = sum(end - start for start, end in RTMPOSE_SEGMENTS.values())
    assert total_kpts == NUM_KEYPOINTS, f"Segments cover {total_kpts} keypoints, expected {NUM_KEYPOINTS}"

    print("  PASS: output format")
    return True


def test_03_real_data_detection():
    """Test 3: Single-person detection on real test data (1 frame per camera)."""
    # Find first video
    videos = []
    for f in os.listdir(VIDEO_DIR):
        if f.endswith(".mp4"):
            videos.append(os.path.join(VIDEO_DIR, f))

    if not videos:
        print("  SKIP: no video files found")
        return None

    detector = MultiPersonDetector(device="cpu", mode="balanced")

    import cv2

    results = {}
    for video_path in sorted(videos)[:1]:  # Just Cam1
        cam_name = os.path.basename(video_path).split("_")[-1].replace(".mp4", "")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"  SKIP: cannot read frame from {cam_name}")
            continue

        t0 = time.time()
        detections = detector.detect_frame(frame)
        t1 = time.time()

        results[cam_name] = {
            "num_detections": len(detections),
            "time_seconds": t1 - t0,
            "persons": [d.to_dict() for d in detections],
        }

        print(f"  {cam_name}: {len(detections)} person(s) detected in {t1-t0:.2f}s")

    assert len(results) > 0, "No cameras processed"
    assert all(r["num_detections"] >= 1 for r in results.values()), \
        f"Not all cameras detected at least 1 person: {results}"

    print("  PASS: real data detection")
    return results


def test_04_detection_rate():
    """Test 4: Detection rate ≥90% across multiple frames (Cam1, first 30 frames)."""
    videos = sorted([
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")
    ])
    if not videos:
        print("  SKIP: no videos")
        return None

    video_path = videos[0]  # Cam1
    detector = MultiPersonDetector(device="cpu", mode="balanced")

    all_dets = detector.detect_video(video_path, frame_range=(0, 29))
    stats = aggregate_detections(all_dets, min_confidence=0.3)

    print(f"  Frames: {stats['total_frames']}, "
          f"With detections: {stats['frames_with_detections']}, "
          f"Rate: {stats['detection_rate']:.1%}")

    assert stats["detection_rate"] >= 0.9, \
        f"Detection rate {stats['detection_rate']:.1%} < 90%"

    print("  PASS: detection rate >=90%")
    return stats


def test_05_synthetic_multi_person():
    """Test 5: Multi-person parsing with synthetic data."""
    # Simulate what _parse_multi_person_results does with mock output
    from multiperson_detector import MultiPersonDetector

    det = MultiPersonDetector(pose_conf_threshold=0.3)

    # Mock: 2 persons detected
    num_persons = 2
    keypoints = np.random.rand(num_persons, NUM_KEYPOINTS, 2).astype(np.float64) * 500
    scores = np.random.rand(num_persons, NUM_KEYPOINTS).astype(np.float64) * 0.5 + 0.5
    bboxes = np.array([
        [100, 100, 300, 400],
        [400, 150, 600, 450],
    ], dtype=np.float64)

    detections = det._parse_multi_person_results(keypoints, scores, bboxes)

    assert len(detections) == 2, f"Expected 2 detections, got {len(detections)}"
    assert detections[0].person_id == 0
    assert detections[1].person_id == 1
    assert detections[0].bbox.shape == (4,)
    assert detections[1].bbox.shape == (4,)
    assert all(d.keypoints_133.shape == (NUM_KEYPOINTS, 3) for d in detections)

    # Verify segments are populated
    for d in detections:
        assert len(d.segments) == 5, f"Expected 5 segments, got {len(d.segments)}"
        for name, seg in d.segments.items():
            expected_len = RTMPOSE_SEGMENTS[name][1] - RTMPOSE_SEGMENTS[name][0]
            assert seg.shape == (expected_len, 3), \
                f"Segment {name}: expected ({expected_len}, 3), got {seg.shape}"

    print("  PASS: synthetic multi-person parsing")
    return True


def test_06_no_detection_edge_case():
    """Test 6: Edge case — no persons detected (all below threshold)."""
    from multiperson_detector import MultiPersonDetector

    det = MultiPersonDetector(pose_conf_threshold=0.99)  # Very high threshold

    # Mock: 1 person but with low scores
    keypoints = np.random.rand(1, NUM_KEYPOINTS, 2).astype(np.float64) * 300
    scores = np.full((1, NUM_KEYPOINTS), 0.1, dtype=np.float64)  # All below 0.99
    bboxes = np.array([[100, 100, 300, 400]], dtype=np.float64)

    detections = det._parse_multi_person_results(keypoints, scores, bboxes)
    assert len(detections) == 0, f"Expected 0 detections, got {len(detections)}"

    # Test with None inputs
    detections = det._parse_multi_person_results(None, None, None)
    assert len(detections) == 0

    # Test with empty arrays
    detections = det._parse_multi_person_results(
        np.array([]).reshape(0, NUM_KEYPOINTS, 2),
        np.array([]).reshape(0, NUM_KEYPOINTS),
        np.array([]).reshape(0, 4),
    )
    assert len(detections) == 0

    print("  PASS: no detection edge cases")
    return True


def test_07_aggregate_detections():
    """Test 7: aggregate_detections summary function."""
    dets_frame0 = [
        PersonDetection(0, np.zeros(4), np.zeros((133, 3)), 0.9),
        PersonDetection(1, np.zeros(4), np.zeros((133, 3)), 0.7),
    ]
    dets_frame1 = [
        PersonDetection(0, np.zeros(4), np.zeros((133, 3)), 0.85),
    ]
    dets_frame2 = []  # No detection

    all_dets = [dets_frame0, dets_frame1, dets_frame2]
    stats = aggregate_detections(all_dets, min_confidence=0.5)

    assert stats["total_frames"] == 3
    assert stats["frames_with_detections"] == 2
    assert abs(stats["detection_rate"] - 2/3) < 1e-6
    assert stats["persons_per_frame"] == [2, 1, 0]
    assert stats["mean_persons"] == 1.0
    assert stats["min_persons"] == 0
    assert stats["max_persons"] == 2
    assert stats["confidence_stats"]["mean"] > 0

    # Test with min_confidence filtering
    stats_filtered = aggregate_detections(all_dets, min_confidence=0.8)
    assert stats_filtered["persons_per_frame"] == [1, 1, 0]

    print("  PASS: aggregate_detections")
    return True


def run_all_tests():
    """Run all Stage 1 tests."""
    print("=" * 60)
    print("  Stage 1 Tests: Multi-Person Detector")
    print("=" * 60)

    tests = [
        ("01 Imports & Dataclass", test_01_imports_and_dataclass),
        ("02 Output Format", test_02_output_format),
        ("03 Real Data Detection", test_03_real_data_detection),
        ("04 Detection Rate", test_04_detection_rate),
        ("05 Synthetic Multi-Person", test_05_synthetic_multi_person),
        ("06 No Detection Edge Case", test_06_no_detection_edge_case),
        ("07 Aggregate Detections", test_07_aggregate_detections),
    ]

    results = {}
    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n--- Test {name} ---")
        try:
            result = test_fn()
            if result is None:
                results[name] = "SKIP"
                skipped += 1
            else:
                results[name] = "PASS"
                passed += 1
        except Exception as e:
            results[name] = f"FAIL: {e}"
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
    print("=" * 60)

    for name, status in results.items():
        marker = "[PASS]" if status == "PASS" else ("[SKIP]" if status == "SKIP" else "[FAIL]")
        print(f"  {marker} {name}: {status}")

    # Save results
    save_data = {
        "stage": 1,
        "module": "multiperson_detector.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": results,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "overall": "PASS" if failed == 0 else "FAIL",
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_FILE}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
