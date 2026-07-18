"""
#10 FLOOR PLANE — Estimate ground plane and ground skeleton

MediaPipe outputs coordinates in an arbitrary coordinate system. The ground
plane is not aligned to Y=0 (or any consistent axis), causing feet to
float above or penetrate through the floor.

Approach:
1. Estimate ground plane from foot contact frames (when ankle velocity is minimal)
2. Project skeleton so feet rest on the estimated ground
3. Optionally scale to real-world dimensions if reference measurements are provided
"""
import numpy as np
from joint_definitions import BODY_IDX


def estimate_ground_plane(skeleton_data, fps=30.0, contact_percentile=10):
    """Estimate the ground plane from foot contact moments.

    Uses a combined criterion for contact detection:
    1. Low velocity (foot is stationary)
    2. Low Y position (foot is near ground level)

    Both criteria must be met for a frame to count as contact.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        fps: video frame rate
        contact_percentile: use the lowest N% of combined score as contacts

    Returns:
        plane_normal: (3,) unit normal vector of the ground plane
        plane_point: (3,) a point on the plane (mean of contact points)
        contact_info: dict with contact frame indices and diagnostics
    """
    contact_info = {"left_frames": [], "right_frames": []}

    foot_data = {}
    for side in ["left", "right"]:
        ankle_idx = BODY_IDX[f"{side}_ankle"]
        heel_idx = BODY_IDX[f"{side}_heel"]
        toe_idx = BODY_IDX[f"{side}_foot_index"]

        ankle = skeleton_data[:, ankle_idx, :]
        heel = skeleton_data[:, heel_idx, :]
        toe = skeleton_data[:, toe_idx, :]

        valid_ankle = ~np.isnan(ankle).any(axis=1)
        valid_heel = ~np.isnan(heel).any(axis=1)
        valid_toe = ~np.isnan(toe).any(axis=1)
        valid = valid_ankle | valid_heel | valid_toe

        if np.sum(valid) < 10:
            continue

        stacked = np.stack([ankle, heel, toe], axis=1)
        foot_center = np.zeros_like(ankle)
        for dim in range(3):
            for f in range(stacked.shape[0]):
                vals = stacked[f, :, dim]
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    foot_center[f, dim] = np.mean(valid_vals)
                else:
                    foot_center[f, dim] = np.nan

        foot_valid = ~np.isnan(foot_center).any(axis=1)

        velocity = np.full(skeleton_data.shape[0], np.inf)
        valid_vel_indices = np.where(foot_valid)[0]
        if len(valid_vel_indices) > 1:
            for i in range(len(valid_vel_indices) - 1):
                fi = valid_vel_indices[i]
                fi_next = valid_vel_indices[i + 1]
                dt = (fi_next - fi) / fps
                if dt > 0:
                    velocity[fi_next] = np.linalg.norm(
                        foot_center[fi_next] - foot_center[fi]
                    ) / dt

        y_values = np.full(skeleton_data.shape[0], np.inf)
        y_values[foot_valid] = foot_center[foot_valid, 1]

        combined_valid = foot_valid & (velocity < np.inf) & (y_values < np.inf)

        if np.sum(combined_valid) < 5:
            continue

        vel_at_valid = velocity[combined_valid]
        vel_scores = np.zeros_like(velocity)
        if np.max(vel_at_valid) > np.min(vel_at_valid):
            vel_scores[combined_valid] = (
                (vel_at_valid - np.min(vel_at_valid))
                / (np.max(vel_at_valid) - np.min(vel_at_valid))
            )

        y_at_valid = y_values[combined_valid]
        y_scores = np.zeros_like(y_values)
        if np.max(y_at_valid) > np.min(y_at_valid):
            y_scores[combined_valid] = (
                (y_at_valid - np.min(y_at_valid))
                / (np.max(y_at_valid) - np.min(y_at_valid))
            )

        combined_score = np.full(skeleton_data.shape[0], np.inf)
        combined_score[combined_valid] = (
            0.5 * vel_scores[combined_valid] + 0.5 * y_scores[combined_valid]
        )

        threshold = np.percentile(combined_score[combined_valid], contact_percentile)
        contact_mask = (combined_score <= threshold) & combined_valid

        contact_frames = np.where(contact_mask)[0]
        contact_info[f"{side}_frames"] = contact_frames.tolist()
        foot_data[side] = foot_center[contact_mask]

    all_contact_points = []
    for side in foot_data:
        if len(foot_data[side]) > 0:
            all_contact_points.append(foot_data[side])

    if not all_contact_points:
        return np.array([0, 1, 0]), np.zeros(3), contact_info

    all_points = np.vstack(all_contact_points)

    centroid = np.mean(all_points, axis=0)

    if len(all_points) >= 3:
        cov = np.cov(all_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        pca_normal = eigenvectors[:, 0]

        if abs(pca_normal[1]) >= 0.3 * np.max(np.abs(pca_normal)):
            normal = pca_normal
        else:
            normal = np.array([0.0, -1.0, 0.0])

        if normal[1] < 0:
            normal = -normal
    else:
        normal = np.array([0.0, 1.0, 0.0])

    contact_info["num_contact_points"] = len(all_points)
    contact_info["plane_normal"] = normal.tolist()
    contact_info["plane_point"] = centroid.tolist()

    return normal, centroid, contact_info


def align_to_ground(
    skeleton_data,
    ground_normal=None,
    ground_point=None,
    target_height=None,
):
    """Align skeleton so feet rest on the ground plane.

    Rotates the skeleton so the estimated ground normal aligns with (0,1,0),
    then translates so the lowest foot point touches Y=0.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        ground_normal: (3,) ground plane normal (estimated if None)
        ground_point: (3,) a point on the ground plane (estimated if None)
        target_height: if provided, scale skeleton so total height matches
            this value in mm (e.g., 1750 for 1.75m person)

    Returns:
        aligned_data: (numFrames, numTrackedPoints, 3) array
        alignment_info: dict with transformation details
    """
    aligned = skeleton_data.copy()
    info = {}

    if ground_normal is None or ground_point is None:
        normal, point, contact_info = estimate_ground_plane(skeleton_data)
        if ground_normal is None:
            ground_normal = normal
        if ground_point is None:
            ground_point = point
        info["contact_info"] = contact_info

    up = np.array([0, 1, 0])
    cos_angle = np.dot(ground_normal, up)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    if abs(cos_angle - 1.0) > 1e-6:
        axis = np.cross(ground_normal, up)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            angle = np.arccos(cos_angle)

            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            for frame_idx in range(skeleton_data.shape[0]):
                aligned[frame_idx] = (R @ skeleton_data[frame_idx].T).T

            info["rotation_angle_deg"] = float(np.degrees(angle))
            info["rotation_axis"] = axis.tolist()

    foot_indices = [
        BODY_IDX["left_ankle"],
        BODY_IDX["left_heel"],
        BODY_IDX["left_foot_index"],
        BODY_IDX["right_ankle"],
        BODY_IDX["right_heel"],
        BODY_IDX["right_foot_index"],
    ]

    all_foot_y = []
    for fi in foot_indices:
        foot_y = aligned[:, fi, 1]
        valid = ~np.isnan(foot_y)
        if np.any(valid):
            all_foot_y.append(foot_y[valid])

    if all_foot_y:
        min_foot_y = np.min(np.concatenate(all_foot_y))
        aligned[:, :, 1] -= min_foot_y
        info["ground_offset_y"] = float(min_foot_y)

    if target_height is not None:
        head_idx = BODY_IDX["nose"]
        foot_ys = []
        for fi in foot_indices:
            foot_ys.append(aligned[:, fi, 1])
        foot_ys_arr = np.array(foot_ys)
        valid_ys = ~np.isnan(foot_ys_arr)
        if np.any(valid_ys):
            mean_foot_ys = np.nanmean(foot_ys_arr, axis=0)
        else:
            mean_foot_ys = np.zeros(foot_ys_arr.shape[1])

        head_y = aligned[:, head_idx, 1]
        valid = ~np.isnan(head_y) & ~np.isnan(mean_foot_ys)
        if np.any(valid):
            heights = head_y[valid] - mean_foot_ys[valid]
            current_height = np.median(heights)
            if current_height > 1e-3:
                scale = target_height / current_height
                for fi in range(aligned.shape[1]):
                    aligned[:, fi, :] *= scale
                info["scale_factor"] = float(scale)
                info["original_height_mm"] = float(current_height)
                info["target_height_mm"] = target_height

    info["ground_normal"] = ground_normal.tolist()
    info["ground_point"] = ground_point.tolist()

    return aligned, info


def detect_floor_penetration(skeleton_data, penetration_threshold=10.0):
    """Detect frames where feet penetrate below the ground plane.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        penetration_threshold: mm below ground to count as penetration

    Returns:
        penetration_info: dict with per-side penetration frame lists
    """
    info = {"left_penetration_frames": [], "right_penetration_frames": []}

    for side in ["left", "right"]:
        ankle_idx = BODY_IDX[f"{side}_ankle"]
        ankle_y = skeleton_data[:, ankle_idx, 1]
        valid = ~np.isnan(ankle_y)

        if not np.any(valid):
            continue

        min_y = np.nanmin(ankle_y[valid])
        threshold_y = min_y - penetration_threshold

        penetration = np.where(valid & (ankle_y < threshold_y))[0]
        info[f"{side}_penetration_frames"] = penetration.tolist()
        info[f"{side}_penetration_count"] = len(penetration)

    return info
