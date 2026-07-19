# Stage 1: Multi-Person 2D Detection — Validation Results

## Status: PASS

## Test Configuration
- **Video**: two_people_talking.mp4 (2-person test video)
- **Resolution**: 640×360, 25 FPS, 277 frames (11.1s)
- **Model**: RTMDet (detection) + RTMPose (pose), balanced mode, CPU
- **Thresholds**: det_conf=0.3, pose_conf=0.3

## Results

| Metric | Value |
|--------|-------|
| Total frames | 277 |
| Exactly 2 detections | 277 (100.0%) |
| False positives (3+) | 0 |
| Missed detections (0-1) | 0 |
| Mean persons/frame | 2.00 |
| Confidence range (person 1) | 0.746–0.803 |
| Confidence range (person 2) | 0.613–0.758 |
| Processing time | 235.0s (0.85s/frame on CPU) |

## Per-Frame Results

All 277 frames show exactly 2 detections. No false positives or missed detections.

Sample frames:
- Frame 0: conf 0.757, 0.709
- Frame 100: conf 0.771, 0.677
- Frame 200: conf 0.790, 0.670
- Frame 276: conf 0.773, 0.681

## Key Findings

1. **RTMDet correctly identifies 2 persons** in every frame when both are visible
2. **No false positives** — detector doesn't create ghost persons
3. **Confidence is consistent** — person 1: 0.75-0.80, person 2: 0.61-0.76
4. **Performance**: ~0.85s/frame on CPU (suitable for offline processing)

## Initial Test Failure (v1 Video)

First attempt used `two_dancers.mp4` (Mixkit dance clip):
- Result: 82% (41/50 frames) — **FAIL**
- Root cause: Video had single-person segments (dancer exits frame)
- Lesson: Test video must have both persons visible throughout

## Conclusion

Stage 1 is validated. The multi-person detector (RTMDet + RTMPose) correctly identifies 2 persons in 100% of frames when both are visible in the video. The detector is ready for integration with Stage 2 (cross-view association).

## Files
- `multiperson_detector.py` — 366 lines, RTMDet + RTMPose via rtmlib
- `joint_definitions.py` — 106 lines, keypoint definitions
- `test_multiperson_detector.py` — 7/7 tests (single-person)
- `run_two_person_test_v2.py` — 2-person validation script
- `threshold_sweep.py` — threshold analysis script
