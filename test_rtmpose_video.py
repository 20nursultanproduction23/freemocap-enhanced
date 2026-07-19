"""Test RTMPose tracker on real video and run pipeline."""
import sys
sys.path.insert(0, r'D:')

import numpy as np
import os
import time
from alternative_tracker import AlternativeTracker

CAM1_PATH = r'D:\freemocap_test_data\freemocap_test_data\synchronized_videos\sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4'
MEDIAPIPE_PATH = r'D:\freemocap_test_data\freemocap_test_data\output_data\raw_data\mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy'
OUTPUT_DIR = r'D:\freemocap_test_data\freemocap_test_data\output_data\raw_data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'rtmpose3dData_numFrames_numTrackedPoints_spatialXYZ.npy')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Run RTMPose on Cam1 video")
print("=" * 60)

tracker = AlternativeTracker(mode="balanced", device="cpu", backend="onnxruntime")
t0 = time.time()
results = tracker.process_video(CAM1_PATH)
t1 = time.time()
print(f"\nRTMPose processing time: {t1 - t0:.1f}s")

# Print segment info
for key in ["body", "feet", "face", "left_hand", "right_hand"]:
    arr = results[key]
    nan_pct = 100 * np.isnan(arr).all(axis=-1).mean() if arr.size > 0 else 100
    print(f"  {key:15s}: shape={str(arr.shape):20s} nan_frames={nan_pct:.1f}%")

# Convert to FreeMoCap format
freemocap_data = tracker.convert_to_freemocap_format(results)
print(f"\nFreeMoCap format: {freemocap_data.shape}")
nan_pct = 100 * np.isnan(freemocap_data).all(axis=-1).mean()
print(f"  NaN frames (all-NaN points): {nan_pct:.1f}%")

# Load MediaPipe for comparison
mediapipe_data = np.load(MEDIAPIPE_PATH)
print(f"\nMediaPipe format: {mediapipe_data.shape}")
mp_nan = 100 * np.isnan(mediapipe_data).all(axis=-1).mean()
print(f"  NaN frames: {mp_nan:.1f}%")

# Compare key body points
print("\n" + "=" * 60)
print("STEP 2: Compare RTMPose vs MediaPipe on same frames")
print("=" * 60)

# Compare nose position (index 0)
for frame_idx in [0, 50, 100, 150, 200]:
    if frame_idx >= min(freemocap_data.shape[0], mediapipe_data.shape[0]):
        continue
    rt_nose = freemocap_data[frame_idx, 0, :]  # nose in RTMPose
    mp_nose = mediapipe_data[frame_idx, 0, :]   # nose in MediaPipe
    print(f"  Frame {frame_idx:3d}: RTMPose nose=({rt_nose[0]:.1f},{rt_nose[1]:.1f},{rt_nose[2]:.1f})  "
          f"MediaPipe nose=({mp_nose[0]:.1f},{mp_nose[1]:.1f},{mp_nose[2]:.1f})")

# Compare wrists
print("\nWrist comparison (index 15=left, 16=right):")
for name, mp_idx in [("left_wrist", 15), ("right_wrist", 16)]:
    rt_vals = freemocap_data[:, mp_idx, :]
    mp_vals = mediapipe_data[:, mp_idx, :]
    rt_valid = rt_vals[~np.isnan(rt_vals).any(axis=1)]
    mp_valid = mp_vals[~np.isnan(mp_vals).any(axis=1)]
    if len(rt_valid) > 0 and len(mp_valid) > 0:
        print(f"  {name}:")
        print(f"    RTMPose:   mean=({rt_valid[:,0].mean():.1f},{rt_valid[:,1].mean():.1f},{rt_valid[:,2].mean():.1f})")
        print(f"    MediaPipe: mean=({mp_valid[:,0].mean():.1f},{mp_valid[:,1].mean():.1f},{mp_valid[:,2].mean():.1f})")

# Save RTMPose output
np.save(OUTPUT_FILE, freemocap_data)
print(f"\nSaved RTMPose output: {OUTPUT_FILE}")

# Step 3: Try running pipeline on RTMPose output
print("\n" + "=" * 60)
print("STEP 3: Run enhanced pipeline on RTMPose output")
print("=" * 60)

try:
    from freemocap_enhanced.pipeline import run_enhanced_pipeline
    t2 = time.time()
    result = run_enhanced_pipeline(
        raw_data=freemocap_data,
        align_floor=False,
        target_height=None,
    )
    t3 = time.time()
    print(f"Pipeline time: {t3 - t2:.1f}s")
    print(f"Output shape: {result.shape}")

    # Save pipeline output
    pipeline_output_file = os.path.join(OUTPUT_DIR, 'rtmpose3d_enhanced.npy')
    np.save(pipeline_output_file, result)
    print(f"Saved pipeline output: {pipeline_output_file}")

    # Quick bone length check
    from joint_definitions import MEDIAPIPE_BONE_CONNECTIONS
    for bone_name, (i, j) in list(MEDIAPIPE_BONE_CONNECTIONS.items())[:6]:
        diffs = np.diff(result[:, [i, j], :], axis=0)
        lengths = np.linalg.norm(diffs, axis=2).mean(axis=1)
        valid = lengths[~np.isnan(lengths)]
        if len(valid) > 0:
            print(f"  {bone_name}: mean_length={valid.mean():.2f}mm, std={valid.std():.2f}mm")

except Exception as e:
    print(f"Pipeline failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
