"""
Stage 2: Cross-View Association for Multi-Person 3D Pose Estimation

Adapts epipolar geometry + appearance matching for FreeMoCap 2D keypoints format.
Uses Campus_Seq1 dataset (3 cameras, 3 actors) as testbed.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class CameraCalibration:
    K: np.ndarray          # 3x3 intrinsic matrix
    R: np.ndarray          # 3x3 rotation (world to camera)
    t: np.ndarray          # 3x1 translation
    dist_coeffs: np.ndarray # distortion coefficients
    
    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix"""
        RT = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ RT
    
    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates"""
        return -self.R.T @ self.t


@dataclass
class PoseDetection:
    keypoints: np.ndarray    # (K, 2) keypoints
    scores: np.ndarray       # (K,) confidence scores
    camera_id: str
    timestamp: float
    detection_id: int = -1


@dataclass
class AssociationResult:
    timestamp: float
    associations: List[Dict]  # [{actor_id, cam0_det_idx, cam1_det_idx, confidence, ambiguous, triangulated_3d}]
    unmatched_cam0: List[int]
    unmatched_cam1: List[int]


def load_campus_calibration(calib_path: str) -> Dict[str, CameraCalibration]:
    """Load Campus dataset calibration into CameraCalibration objects."""
    with open(calib_path) as f:
        data = json.load(f)
    
    cams = {}
    for cam_name, params in data['cameras'].items():
        Tw = np.array(params['Tw'], dtype=np.float32)  # 4x4 [R|t; 0|1]
        K = np.array(params['K'], dtype=np.float32)
        dist = np.array(params['dist_coeffs'], dtype=np.float32)
        
        R = Tw[:3, :3]
        t = Tw[:3, 3]
        
        cams[cam_name] = CameraCalibration(K=K, R=R, t=t, dist_coeffs=dist)
    return cams


def load_campus_2d_annotations(ann_path: str, cameras: List[str]) -> Dict[str, Dict]:
    """Load 2D annotations organized by camera and frame."""
    with open(ann_path) as f:
        data = json.load(f)
    
    result = {cam: {} for cam in cameras}
    for frame_key, frame_data in data['frames'].items():
        cam = frame_data['camera']
        if cam not in cameras:
            continue
        ts = frame_data['timestamp']
        poses = frame_data['poses']
        if poses:
            kpts = np.array([p['points_2d'] for p in poses], dtype=np.float32)  # (N, 14, 2)
            scores = np.array([p['scores'] for p in poses], dtype=np.float32)
            ids = np.array([p.get('id', -1) for p in poses], dtype=np.int32)
            result[cam][ts] = {'kpts': kpts, 'scores': scores, 'ids': ids}
    return result


def undistort_points(points: np.ndarray, cam: CameraCalibration) -> np.ndarray:
    """Undistort 2D points using camera calibration."""
    if len(points) == 0:
        return points
    try:
        import cv2
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        K = cam.K.astype(np.float64)
        dist = cam.dist_coeffs.astype(np.float64)
        undist = cv2.undistortPoints(pts, K, dist, P=K)
        return undist.reshape(-1, 2).astype(np.float32)
    except ImportError:
        return points


def triangulate_point(pt1: np.ndarray, pt2: np.ndarray,
                      cam1: CameraCalibration, cam2: CameraCalibration) -> np.ndarray:
    """Linear triangulation (DLT) for corresponding 2D points."""
    P1 = cam1.P
    P2 = cam2.P
    
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1]
    ], dtype=np.float32)
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) > 1e-6:
        X = X / X[3]
    return X[:3]


def compute_fundamental_matrix(cam1: CameraCalibration, cam2: CameraCalibration) -> np.ndarray:
    """Compute fundamental matrix from two camera calibrations."""
    P1 = cam1.P
    P2 = cam2.P
    
    C1 = -cam1.R.T @ cam1.t
    C2 = -cam2.R.T @ cam2.t
    
    R_rel = cam2.R @ cam1.R.T
    t_rel = cam2.t - R_rel @ cam1.t
    
    t_cross = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ], dtype=np.float32)
    
    E = t_cross @ R_rel
    F = np.linalg.inv(cam2.K).T @ E @ np.linalg.inv(cam1.K)
    return F.astype(np.float32)


def epipolar_distance(F: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Compute Sampson epipolar distance."""
    x1 = np.array([pt1[0], pt1[1], 1.0])
    x2 = np.array([pt2[0], pt2[1], 1.0])
    x2Fx1 = x2 @ F @ x1
    Fx1 = F @ x1
    Ft_x2 = F.T @ x2
    denom = Fx1[0]**2 + Fx1[1]**2 + Ft_x2[0]**2 + Ft_x2[1]**2
    if denom < 1e-8:
        return float('inf')
    return abs(x2Fx1) / np.sqrt(denom)


def pose_similarity(kpts1: np.ndarray, kpts2: np.ndarray,
                    scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Appearance similarity between two pose detections."""
    valid = (scores1 > 0.3) & (scores2 > 0.3)
    if not valid.any():
        return 0.0
    diff = np.linalg.norm(kpts1[valid] - kpts2[valid], axis=1)
    return np.exp(-diff.mean() / 50.0)


def associate_poses_epipolar(dets0: List[PoseDetection], dets1: List[PoseDetection],
                              cam0: CameraCalibration, cam1: CameraCalibration,
                              epipolar_thresh: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate full pose detections between two cameras using epipolar geometry.
    Returns (matched_idx0, matched_idx1, confidences).
    """
    n0, n1 = len(dets0), len(dets1)
    if n0 == 0 or n1 == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    
    F = compute_fundamental_matrix(cam0, cam1)
    
    cost_matrix = np.full((n0, n1), 1000.0, dtype=np.float32)
    
    for i in range(n0):
        for j in range(n1):
            # Check epipolar constraint on multiple keypoints
            epi_dists = []
            for k in range(min(dets0[i].keypoints.shape[0], dets1[j].keypoints.shape[0])):
                s0 = dets0[i].scores[k]
                s1 = dets1[j].scores[k]
                if s0 > 0.3 and s1 > 0.3:
                    d = epipolar_distance(F, dets0[i].keypoints[k], dets1[j].keypoints[k])
                    epi_dists.append(d)
            
            if not epi_dists:
                continue
            
            mean_epi = np.mean(epi_dists)
            if mean_epi > epipolar_thresh:
                continue
            
            # Appearance similarity
            app_sim = pose_similarity(
                dets0[i].keypoints, dets1[j].keypoints,
                dets0[i].scores, dets1[j].scores
            )
            
            cost_matrix[i, j] = mean_epi * (2.0 - app_sim)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_0, matched_1, confs = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < epipolar_thresh:
            matched_0.append(r)
            matched_1.append(c)
            
            # Compute confidence
            epi_dists = []
            for k in range(min(dets0[r].keypoints.shape[0], dets1[c].keypoints.shape[0])):
                s0 = dets0[r].scores[k]
                s1 = dets1[c].scores[k]
                if s0 > 0.3 and s1 > 0.3:
                    d = epipolar_distance(F, dets0[r].keypoints[k], dets1[c].keypoints[k])
                    epi_dists.append(d)
            
            mean_epi = np.mean(epi_dists) if epi_dists else epipolar_thresh
            app_sim = pose_similarity(
                dets0[r].keypoints, dets1[c].keypoints,
                dets0[r].scores, dets1[c].scores
            )
            det_conf = (dets0[r].scores.mean() + dets1[c].scores.mean()) / 2.0
            
            epi_score = max(0, 1.0 - mean_epi / epipolar_thresh)
            conf = 0.3 * det_conf + 0.4 * epi_score + 0.3 * app_sim
            confs.append(conf)
    
    return np.array(matched_0), np.array(matched_1), np.array(confs)


def run_crossview_association(cam_names: Tuple[str, str],
                               calib_path: str,
                               ann_2d_path: str,
                               epipolar_thresh: float = 5.0) -> Dict:
    """Main entry point for cross-view association."""
    cams = load_campus_calibration(calib_path)
    cam0, cam1 = cams[cam_names[0]], cams[cam_names[1]]
    
    ann = load_campus_2d_annotations(ann_2d_path, list(cam_names))
    
    timestamps = sorted(set(ann[cam_names[0]].keys()) & set(ann[cam_names[1]].keys()))
    
    results = []
    for ts in timestamps:
        data0 = ann[cam_names[0]].get(ts)
        data1 = ann[cam_names[1]].get(ts)
        if not data0 or not data1:
            continue
        
        # Create pose detections
        dets0 = []
        for i in range(len(data0['kpts'])):
            kpts = undistort_points(data0['kpts'][i], cam0)
            dets0.append(PoseDetection(
                keypoints=kpts, scores=data0['scores'][i],
                camera_id=cam_names[0], timestamp=ts, detection_id=i
            ))
        
        dets1 = []
        for i in range(len(data1['kpts'])):
            kpts = undistort_points(data1['kpts'][i], cam1)
            dets1.append(PoseDetection(
                keypoints=kpts, scores=data1['scores'][i],
                camera_id=cam_names[1], timestamp=ts, detection_id=i
            ))
        
        m0, m1, confs = associate_poses_epipolar(dets0, dets1, cam0, cam1, epipolar_thresh)
        
        associations = []
        for idx, (mi, mj, conf) in enumerate(zip(m0, m1, confs)):
            # Triangulate all keypoints for this association
            kpts3d = []
            for k in range(min(dets0[mi].keypoints.shape[0], dets1[mj].keypoints.shape[0])):
                s0 = dets0[mi].scores[k]
                s1 = dets1[mj].scores[k]
                if s0 > 0.3 and s1 > 0.3:
                    pt3d = triangulate_point(
                        dets0[mi].keypoints[k], dets1[mj].keypoints[k], cam0, cam1
                    )
                    kpts3d.append(pt3d)
            
            associations.append({
                'actor_id': idx,
                'cam0_det_idx': int(mi),
                'cam1_det_idx': int(mj),
                'confidence': float(conf),
                'ambiguous': conf < 0.6,
                'triangulated_3d': np.array(kpts3d) if kpts3d else None
            })
        
        results.append(AssociationResult(
            timestamp=ts,
            associations=associations,
            unmatched_cam0=[int(i) for i in range(len(dets0)) if i not in m0],
            unmatched_cam1=[int(j) for j in range(len(dets1)) if j not in m1]
        ))
    
    stats = {
        'total_timestamps': len(results),
        'total_associations': sum(len(r.associations) for r in results),
        'avg_per_frame': np.mean([len(r.associations) for r in results]) if results else 0,
        'confident': sum(1 for r in results for a in r.associations if a['confidence'] >= 0.6),
        'ambiguous': sum(1 for r in results for a in r.associations if a['confidence'] < 0.6),
        'timestamps': [r.timestamp for r in results]
    }
    
    return {'results': results, 'stats': stats, 'cameras': cam_names}


def evaluate_separated_vs_crossing(results: List[AssociationResult],
                                    gt_3d_path: str) -> Dict:
    """
    Evaluate association quality on separated vs crossing actors.
    Uses 3D ground truth to determine if actors are close (crossing) or far (separated).
    """
    with open(gt_3d_path) as f:
        gt_data = json.load(f)
    
    gt_by_ts = {e['timestamp']: e for e in gt_data if e['poses']}
    
    separated_confident = 0
    separated_ambiguous = 0
    crossing_confident = 0
    crossing_ambiguous = 0
    separated_total = 0
    crossing_total = 0
    
    for res in results:
        ts = res.timestamp
        if ts not in gt_by_ts:
            continue
        
        gt = gt_by_ts[ts]
        if len(gt['poses']) < 2:
            continue
        
        # Compute pairwise distances between actors in 3D (use first 3 joints as root)
        positions = []
        for p in gt['poses']:
            pts3d = np.array(p['points_3d'])
            if len(pts3d) > 0:
                positions.append(pts3d[0])  # Use first keypoint (root/hip)
            else:
                positions.append([0, 0, 0])
        positions = np.array(positions)  # (N, 3)
        
        if len(positions) < 2:
            continue
            
        dists = cdist(positions, positions)
        min_dist = np.min(dists[np.triu_indices_from(dists, 1)])
        
        is_crossing = min_dist < 1000.0  # 1 meter threshold
        
        for assoc in res.associations:
            if is_crossing:
                crossing_total += 1
                if assoc['confidence'] >= 0.6:
                    crossing_confident += 1
                else:
                    crossing_ambiguous += 1
            else:
                separated_total += 1
                if assoc['confidence'] >= 0.6:
                    separated_confident += 1
                else:
                    separated_ambiguous += 1
    
    return {
        'separated': {
            'total': separated_total,
            'confident': separated_confident,
            'ambiguous': separated_ambiguous,
            'confident_pct': 100 * separated_confident / max(1, separated_total)
        },
        'crossing': {
            'total': crossing_total,
            'confident': crossing_confident,
            'ambiguous': crossing_ambiguous,
            'confident_pct': 100 * crossing_confident / max(1, crossing_total)
        }
    }


if __name__ == '__main__':
    calib_path = r'D:\crossview_data\Campus_Seq1\Campus_Seq1\calibration.json'
    ann_2d_path = r'D:\crossview_data\Campus_Seq1\Campus_Seq1\annotation_2d.json'
    gt_3d_path = r'D:\crossview_data\Campus_Seq1\Campus_Seq1\annotation_3d.json'
    
    # Test Camera0 + Camera1
    result = run_crossview_association(('Camera0', 'Camera1'), calib_path, ann_2d_path)
    stats = result['stats']
    print("=== Camera0 + Camera1 ===")
    print(f"Timestamps processed: {stats['total_timestamps']}")
    print(f"Total associations: {stats['total_associations']}")
    print(f"Avg per frame: {stats['avg_per_frame']:.2f}")
    print(f"Confident (>=0.6): {stats['confident']}")
    print(f"Ambiguous (<0.6): {stats['ambiguous']}")
    
    # Evaluate separated vs crossing
    eval_result = evaluate_separated_vs_crossing(result['results'], gt_3d_path)
    print("\n=== Separated vs Crossing Evaluation ===")
    for scenario, data in eval_result.items():
        print(f"{scenario.capitalize()}:")
        print(f"  Total associations: {data['total']}")
        print(f"  Confident: {data['confident']} ({data['confident_pct']:.1f}%)")
        print(f"  Ambiguous: {data['ambiguous']} ({100-data['confident_pct']:.1f}%)")
    
    # Also test Camera0 + Camera2
    result2 = run_crossview_association(('Camera0', 'Camera2'), calib_path, ann_2d_path)
    stats2 = result2['stats']
    print("\n=== Camera0 + Camera2 ===")
    print(f"Timestamps processed: {stats2['total_timestamps']}")
    print(f"Total associations: {stats2['total_associations']}")
    print(f"Avg per frame: {stats2['avg_per_frame']:.2f}")
    print(f"Confident (>=0.6): {stats2['confident']}")
    print(f"Ambiguous (<0.6): {stats2['ambiguous']}")
    
    eval2 = evaluate_separated_vs_crossing(result2['results'], gt_3d_path)
    print("\n=== Separated vs Crossing (Cam0+Cam2) ===")
    for scenario, data in eval2.items():
        print(f"{scenario.capitalize()}:")
        print(f"  Total: {data['total']}, Confident: {data['confident']} ({data['confident_pct']:.1f}%)")