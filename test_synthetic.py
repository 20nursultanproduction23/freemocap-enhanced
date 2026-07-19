"""
Test cross-view association on synthetic separated/close/crossing scenarios.
"""
import sys
sys.path.insert(0, r'D:\freemocap_enhanced')
from gui.screens.cross_view_association import (
    run_crossview_association, load_campus_calibration, load_campus_2d_annotations
)
import json
import numpy as np
from scipy.spatial.distance import cdist

def load_synthetic_calibration(calib_path):
    """Load synthetic calibration format."""
    with open(calib_path) as f:
        data = json.load(f)
    cams = {}
    for cam_name, params in data['cameras'].items():
        K = np.array(params['K'], dtype=np.float32)
        Tw = np.array(params['Tw'], dtype=np.float32)
        R = Tw[:3, :3]
        t = Tw[:3, 3]
        dist = np.array(params['dist_coeffs'], dtype=np.float32)
        cams[cam_name] = {'K': K, 'R': R, 't': t, 'dist': dist}
    return cams


def load_synthetic_annotations(ann_path):
    """Load synthetic annotations format."""
    with open(ann_path) as f:
        data = json.load(f)
    cams = ['Camera0', 'Camera1']
    result = {cam: {} for cam in cams}
    for frame_key, frame_data in data['frames'].items():
        cam = frame_data['camera']
        ts = frame_data['timestamp']
        poses = frame_data['poses']
        if poses:
            kpts = np.array([p['points_2d'] for p in poses], dtype=np.float32)
            scores = np.array([p['scores'] for p in poses], dtype=np.float32)
            ids = np.array([p['id'] for p in poses], dtype=np.int32)
            result[cam][ts] = {'kpts': kpts, 'scores': scores, 'ids': ids}
    return result


def evaluate_scenario(scenario_name, calib_path, ann_path):
    """Run association and evaluate."""
    from gui.screens.cross_view_association import (
        CameraCalibration, PoseDetection, associate_poses_epipolar,
        triangulate_point, epipolar_distance, compute_fundamental_matrix,
        undistort_points
    )
    
    cams = load_synthetic_calibration(calib_path)
    cam0_obj = CameraCalibration(K=cams['Camera0']['K'], R=cams['Camera0']['R'], 
                                  t=cams['Camera0']['t'], dist_coeffs=cams['Camera0']['dist'])
    cam1_obj = CameraCalibration(K=cams['Camera1']['K'], R=cams['Camera1']['R'],
                                  t=cams['Camera1']['t'], dist_coeffs=cams['Camera1']['dist'])
    
    ann = load_synthetic_annotations(ann_path)
    
    timestamps = sorted(set(ann['Camera0'].keys()) & set(ann['Camera1'].keys()))
    
    confident = 0
    ambiguous = 0
    correct_id = 0
    total = 0
    
    for ts in timestamps:
        data0 = ann['Camera0'].get(ts)
        data1 = ann['Camera1'].get(ts)
        if not data0 or not data1:
            continue
        
        dets0 = [PoseDetection(keypoints=undistort_points(data0['kpts'][i], cam0_obj), 
                               scores=data0['scores'][i], camera_id='Camera0', 
                               timestamp=ts, detection_id=i) 
                 for i in range(len(data0['kpts']))]
        dets1 = [PoseDetection(keypoints=undistort_points(data1['kpts'][i], cam1_obj),
                               scores=data1['scores'][i], camera_id='Camera1',
                               timestamp=ts, detection_id=i)
                 for i in range(len(data1['kpts']))]
        
        m0, m1, confs = associate_poses_epipolar(dets0, dets1, cam0_obj, cam1_obj, 5.0)
        
        for mi, mj, conf in zip(m0, m1, confs):
            total += 1
            if conf >= 0.6:
                confident += 1
            else:
                ambiguous += 1
            
            # Check if IDs match
            gt_id0 = data0['ids'][mi]
            gt_id1 = data1['ids'][mj]
            if gt_id0 == gt_id1:
                correct_id += 1
    
    return {
        'scenario': scenario_name,
        'total': total,
        'confident': confident,
        'ambiguous': ambiguous,
        'confident_pct': 100 * confident / max(1, total),
        'id_accuracy': 100 * correct_id / max(1, total)
    }


if __name__ == '__main__':
    base = 'D:/crossview_data/synthetic_test'
    calib = base + '/calibration.json'
    
    for scenario in ['separated', 'close', 'crossing']:
        ann = base + f'/annotation_2d_{scenario}.json'
        print(f"\n=== {scenario.upper()} ===")
        result = evaluate_scenario(scenario, calib, ann)
        print(f"  Total associations: {result['total']}")
        print(f"  Confident (>=0.6): {result['confident']} ({result['confident_pct']:.1f}%)")
        print(f"  Ambiguous (<0.6): {result['ambiguous']} ({100-result['confident_pct']:.1f}%)")
        print(f"  ID Accuracy: {result['id_accuracy']:.1f}%")