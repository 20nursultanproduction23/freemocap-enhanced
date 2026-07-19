import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "freemocap_test_data", "output_data", "raw_data", "mediapipe3dData_numFrames_numTrackedPoints_spatialXYZ.npy")

print("Loading real FreeMoCap data...")
skeleton = np.load(data_path)
print(f"Shape: {skeleton.shape}")
print(f"Frames: {skeleton.shape[0]}, Tracked points: {skeleton.shape[1]}, Dims: {skeleton.shape[2]}")

nan_count = np.sum(np.isnan(skeleton))
total = skeleton.size
print(f"NaN: {nan_count}/{total} ({nan_count/total*100:.1f}%)")

print("\nRunning Enhanced Pipeline...")
from freemocap_enhanced.pipeline import run_full_pipeline

processed, report = run_full_pipeline(
    skeleton,
    fps=30.0,
    enforce_bones=True,
    fix_foot_sliding=True,
)

print(f"\n=== FINAL RESULTS ===")
print(f"Input shape:  {skeleton.shape}")
print(f"Output shape: {processed.shape}")
nan_before = np.sum(np.isnan(skeleton))
nan_after = np.sum(np.isnan(processed))
print(f"NaN: {nan_before} -> {nan_after}")
print(f"Total time: {report['total_time_seconds']:.2f}s")

from freemocap_enhanced.joint_definitions import BONE_CONNECTIONS, BODY_IDX, NUM_BODY

# Fair comparison: only compare frames where BOTH raw and processed have valid data
print("\nBone length stability (std) - FAIR COMPARISON (matched frames):")
for p, c in BONE_CONNECTIONS:
    pi, ci = BODY_IDX[p], BODY_IDX[c]
    bl_before = np.sqrt(np.sum((skeleton[:, pi, :] - skeleton[:, ci, :]) ** 2, axis=1))
    bl_after = np.sqrt(np.sum((processed[:, pi, :] - processed[:, ci, :]) ** 2, axis=1))
    
    # Only compare frames where raw data was valid
    raw_valid = ~np.isnan(bl_before) & (bl_before > 1e-8)
    proc_valid = ~np.isnan(bl_after) & (bl_after > 1e-8)
    both_valid = raw_valid & proc_valid
    
    if np.any(both_valid):
        std_b = np.std(bl_before[both_valid])
        std_a = np.std(bl_after[both_valid])
        if std_b > 0:
            print(f"  {p:20s} -> {c:20s}: {std_b:.4f} -> {std_a:.4f} ({(1-std_a/std_b)*100:+.1f}%)")

# Also show overall improvement summary
print("\nSummary of improvements:")
improved = 0
worsened = 0
for p, c in BONE_CONNECTIONS:
    pi, ci = BODY_IDX[p], BODY_IDX[c]
    bl_before = np.sqrt(np.sum((skeleton[:, pi, :] - skeleton[:, ci, :]) ** 2, axis=1))
    bl_after = np.sqrt(np.sum((processed[:, pi, :] - processed[:, ci, :]) ** 2, axis=1))
    raw_valid = ~np.isnan(bl_before) & (bl_before > 1e-8)
    both_valid = raw_valid & ~np.isnan(bl_after)
    if np.any(both_valid):
        std_b = np.std(bl_before[both_valid])
        std_a = np.std(bl_after[both_valid])
        if std_b > 0:
            pct = (1 - std_a/std_b) * 100
            if pct > 0:
                improved += 1
            else:
                worsened += 1
print(f"  {improved}/{improved+worsened} bones improved")

output_path = data_path.replace("_spatialXYZ.npy", "_spatialXYZ_enhanced.npy")
np.save(output_path, processed)
print(f"\nSaved: {output_path}")
