"""Run RTMPose on all 3 cameras and compare."""
import sys
sys.path.insert(0, 'D:\\')

import numpy as np
import os
import time
from alternative_tracker import AlternativeTracker

TEST_DIR = r'D:\freemocap_test_data\freemocap_test_data\synchronized_videos'
OUTPUT_DIR = r'D:\freemocap_test_data\freemocap_test_data\output_data\raw_data'

cameras = {
    "Cam1": os.path.join(TEST_DIR, "sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam1.mp4"),
    "Cam2": os.path.join(TEST_DIR, "sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam2.mp4"),
    "Cam3": os.path.join(TEST_DIR, "sesh_2022-09-19_16_16_50_in_class_jsm_synced_Cam3.mp4"),
}

tracker = AlternativeTracker(mode="balanced", device="cpu", backend="onnxruntime")

all_results = {}
for cam_name, video_path in cameras.items():
    print(f"\n{'='*60}")
    print(f"Processing {cam_name}: {video_path}")
    print(f"{'='*60}")

    t0 = time.time()
    results = tracker.process_video(video_path)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s")

    # Stats
    for key in ["body", "feet", "face", "left_hand", "right_hand"]:
        arr = results[key]
        nan_pct = 100 * np.isnan(arr).all(axis=-1).mean() if arr.size > 0 else 100
        # Score stats (4th column)
        scores = arr[:, :, 3]
        valid_scores = scores[~np.isnan(scores)]
        mean_score = valid_scores.mean() if len(valid_scores) > 0 else 0
        print(f"  {key:15s}: shape={str(arr.shape):20s} nan={nan_pct:.1f}% mean_score={mean_score:.3f}")

    # Convert to FreeMoCap format
    freemocap_data = tracker.convert_to_freemocap_format(results)
    print(f"  FreeMoCap: {freemocap_data.shape}, NaN={100*np.isnan(freemocap_data).all(axis=-1).mean():.1f}%")

    # Save
    out_file = os.path.join(OUTPUT_DIR, f'rtmpose3d_{cam_name}_numFrames_numTrackedPoints_spatialXYZ.npy')
    np.save(out_file, freemocap_data)
    print(f"  Saved: {out_file}")

    all_results[cam_name] = {
        "freemocap": freemocap_data,
        "segments": results,
        "time": t1 - t0,
    }

# Cross-camera comparison
print(f"\n{'='*60}")
print("CROSS-CAMERA COMPARISON")
print("="*60)

from freemocap_enhanced.joint_definitions import BODY_IDX

# Compare nose position across cameras
nose_idx = BODY_IDX["nose"]
print("\nNose position across cameras (frame 100):")
for cam_name in cameras:
    data = all_results[cam_name]["freemocap"]
    if data.shape[0] > 100:
        nose = data[100, nose_idx, :]
        print(f"  {cam_name}: ({nose[0]:.1f}, {nose[1]:.1f}, {nose[2]:.1f})")

# Compare mean detection scores
print("\nMean body detection score across cameras:")
for cam_name in cameras:
    body = all_results[cam_name]["segments"]["body"]
    scores = body[:, :, 3]
    valid = scores[~np.isnan(scores)]
    print(f"  {cam_name}: {valid.mean():.3f} (min={valid.min():.3f}, max={valid.max():.3f})")

print("\nDONE - All 3 cameras processed")
