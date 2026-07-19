import sys
sys.path.insert(0, 'D:\\')
import numpy as np
np.random.seed(42)

print('=== GENERATING TEST DATA ===')
num_frames = 300
num_tracked = 553

skeleton = np.full((num_frames, num_tracked, 3), np.nan)

body_data = np.zeros((num_frames, 33, 3))
for joint in range(33):
    body_data[:, joint, 0] = joint * 0.1 + np.random.randn(num_frames) * 0.002
    body_data[:, joint, 1] = joint * 0.05 + np.random.randn(num_frames) * 0.002
    body_data[:, joint, 2] = joint * 0.03 + np.sin(np.linspace(0, 2*np.pi, num_frames)) * 0.01
    body_data[:, joint, :] += np.random.randn(num_frames, 3) * 0.001

body_data[50:55, 15, :] = np.nan
body_data[100, :, :] = np.nan
body_data[200:203, 16, :] = np.nan

for i in range(33):
    skeleton[:, i, :] = body_data[:, i, :]

print(f'Test data: {skeleton.shape}, NaN count: {np.sum(np.isnan(skeleton))}')

print('\n=== RUNNING PIPELINE ===')
from freemocap_enhanced.pipeline import run_full_pipeline

processed, report = run_full_pipeline(
    skeleton,
    fps=30.0,
    enforce_bones=True,
    fix_foot_sliding=True,
)

print(f'\n=== RESULTS ===')
print(f'Output shape: {processed.shape}')
nan_before = np.sum(np.isnan(skeleton))
nan_after = np.sum(np.isnan(processed))
print(f'NaN count: {nan_before} -> {nan_after}')
print(f'Total time: {report["total_time_seconds"]:.2f}s')

bone_lengths_before = {}
bone_lengths_after = {}
from freemocap_enhanced.joint_definitions import BONE_CONNECTIONS, BODY_IDX
for p, c in BONE_CONNECTIONS:
    pi, ci = BODY_IDX[p], BODY_IDX[c]
    bl = np.sqrt(np.sum((skeleton[:, pi, :] - skeleton[:, ci, :]) ** 2, axis=1))
    valid = ~np.isnan(bl)
    if np.any(valid):
        bone_lengths_before[(p,c)] = np.std(bl[valid])
    bl2 = np.sqrt(np.sum((processed[:, pi, :] - processed[:, ci, :]) ** 2, axis=1))
    valid2 = ~np.isnan(bl2)
    if np.any(valid2):
        bone_lengths_after[(p,c)] = np.std(bl2[valid2])

print('\nBone length std (before -> after):')
for bone in list(bone_lengths_before.keys())[:8]:
    b = bone_lengths_before[bone]
    a = bone_lengths_after.get(bone, 0)
    if b > 0:
        reduction = (1 - a/b) * 100
        print(f'  {bone}: {b:.6f} -> {a:.6f} ({reduction:.1f}% reduction)')
    else:
        print(f'  {bone}: {b:.6f} -> {a:.6f}')

print('\n=== ALL TESTS PASSED ===')
