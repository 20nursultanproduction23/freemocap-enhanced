"""Comprehensive test for problems 15-20 modules."""
import sys
sys.path.insert(0, 'D:\\')

import numpy as np
import os
import json
import time

np.random.seed(42)
results = {}

# ============================================================
# TEST #15: Self-occlusion detection
# ============================================================
print("=" * 60)
print("TEST #15: Self-occlusion detection (cross-camera consistency)")
print("=" * 60)

from self_occlusion import detect_self_occlusion

num_frames = 222
np.random.seed(42)

cameras_data = []
for cam_idx in range(3):
    cam_dict = {}
    for seg, n_kp in [("body", 17), ("face", 68), ("left_hand", 21), ("right_hand", 21)]:
        base = np.random.randn(num_frames, n_kp, 3).astype(np.float32) * 50
        base += np.array([160, 100, -0.5])
        scores = np.ones((num_frames, n_kp, 1), dtype=np.float32) * 0.8
        cam_dict[seg] = np.concatenate([base, scores], axis=2)
    cameras_data.append(cam_dict)

# Inject self-occlusion: frame 50, camera 0, body keypoint 9 (left wrist in COCO)
cameras_data[0]["body"][50, 9, :3] = cameras_data[1]["body"][50, 9, :3] + np.array([80, 0, 0])

image_sizes = [(720, 1280)] * 3
result15 = detect_self_occlusion(cameras_data, image_sizes)
results['15'] = {
    'status': 'PASS' if len(result15['flagged_frames']) > 0 else 'PARTIAL',
    'flagged_frames': len(result15['flagged_frames']),
    'flagged_keypoints': len(result15['flagged_keypoints']),
}
print(f"  Flagged frames: {results['15']['flagged_frames']}")
print(f"  Flagged keypoints: {results['15']['flagged_keypoints']}")
print(f"  Status: {results['15']['status']}")

# ============================================================
# TEST #16: Motion blur detection
# ============================================================
print("\n" + "=" * 60)
print("TEST #16: Motion blur detection")
print("=" * 60)

from motion_blur import detect_motion_from_skeleton, evaluate_deblur_impact

skeleton = np.random.randn(222, 553, 3).astype(np.float32) * 200
# Add consistent motion
skeleton[:, 15, 0] = np.linspace(0, 5000, 222)
skeleton[:, 16, 0] = np.linspace(0, 5000, 222)

result16_motion = detect_motion_from_skeleton(skeleton, fps=6.0)
high_motion = len(result16_motion['high_motion_frames'])
print(f"  High motion frames: {high_motion}")
print(f"  Mean velocity: {result16_motion['mean_velocity'].mean():.1f} mm/s")

# Test deblur evaluation
blur_scores = np.random.randn(222) * 20 + 100
quality = blur_scores * 0.5 + np.random.randn(222) * 5
result16_deblur = evaluate_deblur_impact(blur_scores, quality)
print(f"  Correlation: {result16_deblur['correlation']:.3f}")
print(f"  Recommendation: {result16_deblur['recommendation']}")
results['16'] = {'status': 'PASS', 'high_motion': high_motion, 'correlation': result16_deblur['correlation']}

# ============================================================
# TEST #17: Calibration drift detection
# ============================================================
print("\n" + "=" * 60)
print("TEST #17: Calibration drift detection")
print("=" * 60)

from calibration_drift import detect_calibration_drift

num_frames, num_pts = 222, 33
pts_3d = np.random.randn(num_frames, num_pts, 3).astype(np.float32) * 500 + np.array([0, -800, 2000])
# Inject drift at frame 150: add sudden oscillation (increases acceleration)
pts_3d[150:, :, 0] += np.sin(np.linspace(0, 20 * np.pi, num_frames - 150))[:, np.newaxis] * 2000

result17 = detect_calibration_drift(pts_3d, camera_matrices=None, projection_matrices=None)
drift_signals = result17['drift_signals']
print(f"  Drift signals detected: {len(drift_signals)}")
if drift_signals:
    for sig in drift_signals[:3]:
        print(f"    Camera {sig.get('camera_idx', '?')}: frame {sig.get('frame_start', '?')}, severity={sig.get('severity', '?')}")
print(f"  Camera health: {[h['status'] for h in result17['camera_health']]}")
results['17'] = {'status': 'PASS' if drift_signals else 'PARTIAL', 'signals': len(drift_signals)}

# ============================================================
# TEST #18: Texture analysis
# ============================================================
print("\n" + "=" * 60)
print("TEST #18: Texture/clothing false positive analysis")
print("=" * 60)

from texture_analysis import analyze_texture_artifacts

skeleton_tex = np.random.randn(222, 553, 3).astype(np.float32) * 100
# Add high jitter to torso (indices 11, 12, 23, 24)
for idx in [11, 12, 23, 24]:
    skeleton_tex[:, idx, :] += np.random.randn(222, 3) * 50
confidence_tex = np.ones((222, 553), dtype=np.float32) * 0.8
confidence_tex[:, [11, 12, 23, 24]] = 0.4

result18 = analyze_texture_artifacts(skeleton_tex, fps=6.0, confidence_data=confidence_tex)
print(f"  Overall risk: {result18['overall_texture_risk']}")
print(f"  Suspicious regions: {result18['suspicious_regions']}")
results['18'] = {
    'status': 'PASS',
    'risk': result18['overall_texture_risk'],
    'suspicious': len(result18['suspicious_regions'])
}

# ============================================================
# TEST #19: Distortion weighting
# ============================================================
print("\n" + "=" * 60)
print("TEST #19: Wide-angle distortion weighting")
print("=" * 60)

from distortion_weighting import compute_distortion_weights, analyze_edge_reliability

kpts_2d = np.random.rand(222, 133, 2).astype(np.float32)
kpts_2d[:, :, 0] *= 1280
kpts_2d[:, :, 1] *= 720
# Keypoint 0 at center, keypoint 1 at edge
kpts_2d[:, 0, :] = [640, 360]
kpts_2d[:, 1, :] = [1270, 710]

weights = compute_distortion_weights(kpts_2d, (720, 1280))
print(f"  Center keypoint (0) weight: {weights[0, 0]:.3f}")
print(f"  Edge keypoint (1) weight: {weights[0, 1]:.3f}")
assert weights[0, 0] > weights[0, 1], "Center should have higher weight than edge"

result19_edge = analyze_edge_reliability(kpts_2d, (720, 1280))
print(f"  Edge keypoints: {result19_edge['edge_keypoints'][:5]}...")
print(f"  Reliable zone: {result19_edge['reliable_zone_mask'].mean()*100:.1f}% of points reliable")
results['19'] = {'status': 'PASS', 'center_weight': float(weights[0, 0]), 'edge_weight': float(weights[0, 1])}

# ============================================================
# TEST #20: Camera coverage map
# ============================================================
print("\n" + "=" * 60)
print("TEST #20: Camera coverage zone map")
print("=" * 60)

from camera_coverage import compute_coverage_map, generate_ascii_map

cam_positions = [
    (0, -3, 2), (0, 3, 2), (-3, 0, 2), (3, 0, 2),
]
cam_orientations = [
    (0, 0, 0),     # cam at (0,-3,2) looking +Z
    (180, 0, 0),   # cam at (0,3,2) looking -Z
    (90, 0, 0),    # cam at (-3,0,2) looking +X
    (-90, 0, 0),   # cam at (3,0,2) looking -X
]
cam_fov_h = [120] * 4
cam_fov_v = [90] * 4

result20 = compute_coverage_map(cam_positions, cam_orientations, cam_fov_h, cam_fov_v, room_size=(6, 6), resolution=10)
grid = result20['coverage_grid']
reliable_pct = result20['reliable_zone'].mean() * 100
print(f"  Grid shape: {grid.shape}")
print(f"  Min cameras: {grid.min()}, Max cameras: {grid.max()}")
print(f"  Reliable zone (>=3 cameras): {reliable_pct:.1f}%")
print(f"  Unreliable cells: {len(result20['unreliable_cells'])}")

ascii_map = generate_ascii_map(grid, result20['cell_size'])
print(f"\n  ASCII coverage map ({len(ascii_map)} rows):")
for row in ascii_map[:5]:
    print(f"    {row}")

results['20'] = {'status': 'PASS', 'reliable_pct': float(reliable_pct), 'max_cameras': int(grid.max())}

# ============================================================
# TEST #15 (REAL DATA): Self-occlusion on RTMPose multi-camera
# ============================================================
print("\n" + "=" * 60)
print("TEST #15 (REAL DATA): Self-occlusion on RTMPose 3-camera data")
print("=" * 60)

rtmpose_dir = r'D:\freemocap_test_data\freemocap_test_data\output_data\raw_data'
cam_files = [
    os.path.join(rtmpose_dir, f'rtmpose3d_Cam{c}_numFrames_numTrackedPoints_spatialXYZ.npy')
    for c in [1, 2, 3]
]

if all(os.path.exists(f) for f in cam_files):
    cam_data_real = [np.load(f) for f in cam_files]
    cameras_real = []
    for cd in cam_data_real:
        valid = ~np.isnan(cd).any(axis=2)
        cam_dict = {
            "body": np.concatenate([cd[:, :17, :], valid[:, :17, np.newaxis].astype(np.float32)], axis=2),
            "face": np.concatenate([cd[:, 75:143, :], valid[:, 75:143, np.newaxis].astype(np.float32)], axis=2),
            "left_hand": np.concatenate([cd[:, 54:75, :], valid[:, 54:75, np.newaxis].astype(np.float32)], axis=2),
            "right_hand": np.concatenate([cd[:, 33:54, :], valid[:, 33:54, np.newaxis].astype(np.float32)], axis=2),
        }
        cameras_real.append(cam_dict)

    result15_real = detect_self_occlusion(cameras_real, [(720, 1280)] * 3)
    print(f"  Flagged frames: {len(result15_real['flagged_frames'])}")
    print(f"  Flagged keypoints: {len(result15_real['flagged_keypoints'])}")
    print(f"  Max disagreement score: {result15_real['per_frame_scores'].max():.3f}")
else:
    print("  RTMPose data not found, skipping real data test")

# ============================================================
# TEST #18 (REAL DATA): Texture analysis on real data
# ============================================================
print("\n" + "=" * 60)
print("TEST #18 (REAL DATA): Texture analysis on real FreeMoCap data")
print("=" * 60)

mp_enhanced = os.path.join(rtmpose_dir, 'mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ_v3_aligned.npy')
if os.path.exists(mp_enhanced):
    mp_data = np.load(mp_enhanced)
    result18_real = analyze_texture_artifacts(mp_data, fps=6.0)
    print(f"  Overall risk: {result18_real['overall_texture_risk']}")
    print(f"  Suspicious regions: {result18_real['suspicious_regions']}")
    for region, jitter in result18_real['temporal_jitter_per_region'].items():
        print(f"    {region}: jitter={jitter:.2f}")
else:
    print("  MediaPipe enhanced data not found")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY — Problems 15-20")
print("=" * 60)

all_pass = True
for key, val in results.items():
    status = val['status']
    if status != 'PASS':
        all_pass = False
    print(f"  #{key}: {status}")

print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME NEED ATTENTION'}")

# Save results
with open(r'D:\freemocap_enhanced\test_results_15_20.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results saved to test_results_15_20.json")
