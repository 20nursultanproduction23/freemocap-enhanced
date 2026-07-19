# Stage 2: Cross-View Association — Validation Results

## Status: PASS (with limitations)

## Test Configuration
- **Calibration**: `freemocap_test_data_camera_calibration.toml` (3 cameras)
- **Videos**: 3 synchronized cameras, 720x1280, 6 FPS, 222 frames
- **Association method**: Epipolar geometry + Hungarian algorithm + Union-Find
- **Max epipolar distance**: 50px threshold

## Results

### Test 1: Real Person Consistency (20 frames)
- **Result**: PASS — Real person (P0) correctly matched across all 3 cameras in all frames
- Frames 0-8: 2 detections (real + false positive) → 2 actors
- Frames 9-19: 1 detection per camera → 1 actor
- Real person consistently mapped to Actor 0

### Test 2: Epipolar Distance Analysis (60 matched pairs)
| Metric | Value |
|--------|-------|
| Mean distance | 10.29px |
| Median | 9.49px |
| Min | 1.86px |
| Max | 20.36px |
| Std | 4.92px |
| Matches < 20px | 58/60 (96.7%) |
| Matches < 30px | 60/60 (100.0%) |
| Matches < 50px | 60/60 (100.0%) |

### Test 3: Synthetic 2-Person (10 frames)
- Real person consistent: 10/10 (100%)
- Synthetic person merged with real person (expected — keypoints too similar)
- Both mapped to Actor 0 (synthetic lacks real 3D geometry)

## Key Findings

1. **Calibration loads correctly** — 3 cameras with K, R, t matrices from TOML
2. **Fundamental matrices well-conditioned** — norm=1.0, max=0.9999
3. **Real person matched across all cameras** — consistent Actor 0 assignment
4. **Epipolar distances very low** — 10.3px mean indicates correct geometry
5. **False positive handled gracefully** — isolated to Cam0, doesn't affect real person matching

## Files Created
- `calibration_loader.py` — Loads FreeMoCap TOML calibration, converts Rodrigues to R matrix
- `cross_view_association.py` — Existing 535-line module (epipolar + Hungarian + Union-Find)
- `test_stage2.py` — Basic pipeline test
- `test_stage2_synthetic.py` — Synthetic 2-person test
- `test_stage2_full.py` — Comprehensive 3-test suite

## Limitations
- Synthetic 2-person test doesn't prove real multi-person association (keypoints are just shifted copies)
- Need real multi-person multi-camera data for full validation
- Only tested on 3 cameras (not 6)

## Conclusion
Stage 2 pipeline works correctly on available data. The association algorithm correctly matches the real person across cameras using epipolar geometry. Full validation requires real multi-person multi-camera data.
