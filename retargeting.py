"""
#12 RETARGETING — Scale and adapt skeleton to different body proportions

FreeMoCap captures motion in an arbitrary coordinate system with proportions
determined by the tracked person. For use in animation, game engines, or
comparison with SMPL/SMPL-X models, the motion needs to be retargeted to
a skeleton with different proportions.

This module provides:
1. Compute reference proportions from captured data
2. Apply per-bone scaling to match target proportions
3. Preserve motion dynamics while adjusting proportions
4. Support both uniform scaling and anatomical proportion adjustment
"""
import numpy as np
from joint_definitions import (
    BODY_IDX, BONE_CONNECTIONS,
    NUM_BODY, NUM_RIGHT_HAND, NUM_LEFT_HAND, NUM_FACE, NUM_TOTAL,
)


def compute_skeleton_proportions(skeleton_data):
    """Compute body proportions from captured skeleton data.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array

    Returns:
        proportions: dict with bone lengths, ratios, and body measurements
    """
    bone_lengths = {}
    for parent_name, child_name in BONE_CONNECTIONS:
        i = BODY_IDX[parent_name]
        j = BODY_IDX[child_name]
        dists = np.linalg.norm(
            skeleton_data[:, i, :] - skeleton_data[:, j, :], axis=1
        )
        valid = ~np.isnan(dists)
        if np.any(valid):
            bone_lengths[f"{parent_name}->{child_name}"] = float(
                np.median(dists[valid])
            )

    shoulder_l = BODY_IDX["left_shoulder"]
    shoulder_r = BODY_IDX["right_shoulder"]
    shoulder_width = np.nanmedian(np.linalg.norm(
        skeleton_data[:, shoulder_l, :] - skeleton_data[:, shoulder_r, :],
        axis=1
    ))

    hip_l = BODY_IDX["left_hip"]
    hip_r = BODY_IDX["right_hip"]
    hip_width = np.nanmedian(np.linalg.norm(
        skeleton_data[:, hip_l, :] - skeleton_data[:, hip_r, :],
        axis=1
    ))

    nose_y = np.nanmedian(skeleton_data[:, BODY_IDX["nose"], 1])
    left_ankle_y = skeleton_data[:, BODY_IDX["left_ankle"], 1]
    right_ankle_y = skeleton_data[:, BODY_IDX["right_ankle"], 1]
    stacked_y = np.stack([left_ankle_y, right_ankle_y], axis=0)
    with np.errstate(invalid='ignore'):
        ankle_y = np.nanmedian(np.nanmean(stacked_y, axis=0))
    total_height = abs(ankle_y - nose_y)

    trunk_length = bone_lengths.get("left_hip->right_hip", 0)
    torso_l = bone_lengths.get("left_shoulder->left_elbow", 0)
    forearm_l = bone_lengths.get("left_elbow->left_wrist", 0)
    thigh_l = bone_lengths.get("left_hip->left_knee", 0)
    shin_l = bone_lengths.get("left_knee->left_ankle", 0)

    return {
        "bone_lengths": bone_lengths,
        "shoulder_width": float(shoulder_width),
        "hip_width": float(hip_width),
        "shoulder_hip_ratio": float(shoulder_width / hip_width) if hip_width > 0 else 1.0,
        "total_height": float(total_height),
        "torso_length": trunk_length,
        "upper_arm_length": torso_l,
        "forearm_length": forearm_l,
        "thigh_length": thigh_l,
        "shin_length": shin_l,
        "limb_ratio": float((thigh_l + shin_l) / (torso_l + forearm_l))
            if (torso_l + forearm_l) > 0 else 1.0,
    }


def retarget_to_proportions(skeleton_data, target_proportions=None,
                              target_height=None, preserve_motion=True):
    """Retarget skeleton motion to different body proportions.

    Two modes:
    1. Uniform scaling: scale entire skeleton to match target height
    2. Proportional retarget: adjust individual bone lengths to match
       target proportions while preserving motion trajectory

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        target_proportions: dict from compute_skeleton_proportions() of
            target skeleton. If None, only uniform scaling is applied.
        target_height: desired total height in mm. If None, uses source height.
        preserve_motion: if True, preserve root trajectory and overall motion

    Returns:
        retargeted: (numFrames, numTrackedPoints, 3) array
        info: dict with transformation details
    """
    retargeted = skeleton_data.copy()
    info = {"mode": "none"}

    source_props = compute_skeleton_proportions(skeleton_data)

    if target_height is None:
        target_height = source_props["total_height"]

    if target_height > 0 and source_props["total_height"] > 0:
        uniform_scale = target_height / source_props["total_height"]
    else:
        uniform_scale = 1.0

    if target_proportions is None or not target_proportions.get("bone_lengths"):
        root_idx = BODY_IDX["hips_center"] if "hips_center" in BODY_IDX else BODY_IDX["left_hip"]
        root_pos = retargeted[:, BODY_IDX["left_hip"], :].copy()

        for f in range(retargeted.shape[0]):
            for p in range(retargeted.shape[1]):
                offset = retargeted[f, p, :] - root_pos[f]
                retargeted[f, p, :] = root_pos[f] + offset * uniform_scale

        info["mode"] = "uniform_scaling"
        info["scale_factor"] = float(uniform_scale)
        info["source_height"] = source_props["total_height"]
        info["target_height"] = target_height
        return retargeted, info

    target_bones = target_proportions.get("bone_lengths", {})
    scale_factors = {}
    for bone_name, source_len in source_props["bone_lengths"].items():
        target_len = target_bones.get(bone_name, source_len)
        if source_len > 1e-6:
            scale_factors[bone_name] = target_len / source_len
        else:
            scale_factors[bone_name] = 1.0

    bfs_order = [
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_foot_index"),
    ]

    root_idx = BODY_IDX["left_hip"]
    root_pos = retargeted[:, root_idx, :].copy()

    target_height_ratio = (
        target_proportions.get("total_height", target_height)
        / source_props["total_height"]
        if source_props["total_height"] > 0 else 1.0
    )

    for f in range(retargeted.shape[0]):
        offset = retargeted[f, root_idx, :] - root_pos[f]
        retargeted[f, root_idx, :] = root_pos[f] + offset * target_height_ratio

    stored_directions = {}
    stored_lengths = {}
    for parent_name, child_name in bfs_order:
        bone_key = f"{parent_name}->{child_name}"
        parent_idx = BODY_IDX[parent_name]
        child_idx = BODY_IDX[child_name]

        dirs = skeleton_data[:, child_idx, :] - skeleton_data[:, parent_idx, :]
        lens = np.linalg.norm(dirs, axis=1)
        stored_directions[bone_key] = dirs
        stored_lengths[bone_key] = lens

    adjusted_skeleton = skeleton_data.copy()

    for parent_name, child_name in bfs_order:
        bone_key = f"{parent_name}->{child_name}"
        parent_idx = BODY_IDX[parent_name]
        child_idx = BODY_IDX[child_name]

        sf = scale_factors.get(bone_key, 1.0)
        sf = float(np.clip(sf, 0.3, 3.0))

        for f in range(adjusted_skeleton.shape[0]):
            parent_pos = adjusted_skeleton[f, parent_idx, :]
            orig_dir = stored_directions[bone_key][f]
            orig_len = stored_lengths[bone_key][f]

            if np.isnan(parent_pos).any() or np.isnan(orig_dir).any():
                continue
            if orig_len < 1e-6:
                adjusted_skeleton[f, child_idx, :] = parent_pos
                continue

            unit_dir = orig_dir / orig_len
            new_len = orig_len * sf
            adjusted_skeleton[f, child_idx, :] = parent_pos + unit_dir * new_len

    retargeted = adjusted_skeleton

    for parent_name, child_name in bfs_order:
        parent_idx = BODY_IDX[parent_name]
        child_idx = BODY_IDX[child_name]
        bone_key = f"{parent_name}->{child_name}"

        sf = scale_factors.get(bone_key, 1.0)
        sf = float(np.clip(sf, 0.3, 3.0))

        for f in range(retargeted.shape[0]):
            parent_pos = retargeted[f, parent_idx, :]
            orig_dir = stored_directions[bone_key][f]
            orig_len = stored_lengths[bone_key][f]

            if np.isnan(parent_pos).any() or np.isnan(orig_dir).any():
                continue
            if orig_len < 1e-6:
                retargeted[f, child_idx, :] = parent_pos
                continue

            unit_dir = orig_dir / orig_len
            new_len = orig_len * sf
            retargeted[f, child_idx, :] = parent_pos + unit_dir * new_len

    retarget_props = compute_skeleton_proportions(retargeted)
    mismatch = {}
    for bone_key, target_len in target_bones.items():
        actual_len = retarget_props["bone_lengths"].get(bone_key, 0.0)
        if target_len > 1e-6:
            mismatch[bone_key] = float(abs(actual_len - target_len) / target_len)
        else:
            mismatch[bone_key] = 0.0

    overall_error = float(np.mean(list(mismatch.values()))) if mismatch else 0.0

    info["mode"] = "proportional_retarget"
    info["scale_factors"] = {k: float(v) for k, v in scale_factors.items()}
    info["target_height_ratio"] = float(target_height_ratio)
    info["source_height"] = source_props["total_height"]
    info["target_height"] = target_proportions.get("total_height", target_height)
    info["proportion_mismatch"] = mismatch
    info["overall_proportion_error"] = overall_error

    return retargeted, info


def get_smpl_compatible_proportions(skeleton_data):
    """Convert proportions to SMPL-compatible format.

    SMPL has 24 body joints. This maps FreeMoCap's 33 body landmarks
    to approximate SMPL joint positions for retargeting.

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array

    Returns:
        smpl_joints: dict mapping SMPL joint names to mean positions
        proportions: dict with SMPL-compatible measurements
    """
    mean_data = np.nanmean(skeleton_data, axis=0)

    smpl_mapping = {
        "pelvis": "hips_center",
        "spine1": "left_shoulder",
        "spine2": "left_shoulder",
        "neck": "left_shoulder",
        "head": "nose",
        "left_shoulder": "left_shoulder",
        "left_elbow": "left_elbow",
        "left_wrist": "left_wrist",
        "right_shoulder": "right_shoulder",
        "right_elbow": "right_elbow",
        "right_wrist": "right_wrist",
        "left_hip": "left_hip",
        "left_knee": "left_knee",
        "left_ankle": "left_ankle",
        "right_hip": "right_hip",
        "right_knee": "right_knee",
        "right_ankle": "right_ankle",
    }

    smpl_joints = {}
    for smpl_name, freemocap_name in smpl_mapping.items():
        if freemocap_name in BODY_IDX:
            idx = BODY_IDX[freemocap_name]
            smpl_joints[smpl_name] = mean_data[idx].tolist()

    props = compute_skeleton_proportions(skeleton_data)
    props["smpl_joint_positions"] = smpl_joints

    return smpl_joints, props


def validate_retarget(source_data, retargeted_data, target_proportions=None):
    """Validate retargeted skeleton against source or target proportions.

    Args:
        source_data: (numFrames, numTrackedPoints, 3) original skeleton
        retargeted_data: (numFrames, numTrackedPoints, 3) retargeted skeleton
        target_proportions: dict from compute_skeleton_proportions() of the
            intended target. If None, compares retargeted against source.

    Returns:
        result: dict with:
            - source_proportions: proportions of source skeleton
            - retargeted_proportions: proportions of retargeted skeleton
            - per_bone_mismatch: dict mapping bone_key to absolute ratio error
            - overall_error: mean proportion error across all bones
    """
    source_props = compute_skeleton_proportions(source_data)
    retarget_props = compute_skeleton_proportions(retargeted_data)

    if target_proportions is not None:
        reference_bones = target_proportions.get("bone_lengths", {})
    else:
        reference_bones = source_props.get("bone_lengths", {})

    per_bone = {}
    for bone_key, ref_len in reference_bones.items():
        actual_len = retarget_props["bone_lengths"].get(bone_key, 0.0)
        if ref_len > 1e-6:
            per_bone[bone_key] = float(abs(actual_len - ref_len) / ref_len)
        else:
            per_bone[bone_key] = 0.0

    overall = float(np.mean(list(per_bone.values()))) if per_bone else 0.0

    return {
        "source_proportions": source_props,
        "retargeted_proportions": retarget_props,
        "per_bone_mismatch": per_bone,
        "overall_error": overall,
    }


def retarget_to_smpl(skeleton_data):
    """Retarget skeleton to SMPL-typical body proportions.

    Uses standard SMPL measurements for a 1700mm tall adult:
    - shoulder width: 415mm
    - hip width: 290mm
    - upper arm: 280mm
    - forearm: 250mm
    - thigh: 420mm
    - shin: 400mm
    - trunk (hip-to-hip): 290mm

    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array

    Returns:
        retargeted: (numFrames, numTrackedPoints, 3) array
        info: dict with transformation details
    """
    smpl_bone_lengths = {
        "left_hip->right_hip": 290.0,
        "left_shoulder->right_shoulder": 415.0,
        "left_shoulder->left_elbow": 280.0,
        "left_elbow->left_wrist": 250.0,
        "right_shoulder->right_elbow": 280.0,
        "right_elbow->right_wrist": 250.0,
        "left_hip->left_knee": 420.0,
        "left_knee->left_ankle": 400.0,
        "right_hip->right_knee": 420.0,
        "right_knee->right_ankle": 400.0,
        "left_ankle->left_heel": 60.0,
        "left_ankle->left_foot_index": 180.0,
        "right_ankle->right_heel": 60.0,
        "right_ankle->right_foot_index": 180.0,
    }

    target_proportions = {
        "total_height": 1700.0,
        "shoulder_width": 415.0,
        "hip_width": 290.0,
        "shoulder_hip_ratio": 415.0 / 290.0,
        "torso_length": 290.0,
        "upper_arm_length": 280.0,
        "forearm_length": 250.0,
        "thigh_length": 420.0,
        "shin_length": 400.0,
        "limb_ratio": (420.0 + 400.0) / (280.0 + 250.0),
        "bone_lengths": smpl_bone_lengths,
    }

    return retarget_to_proportions(
        skeleton_data,
        target_proportions=target_proportions,
        target_height=1700.0,
    )


def interpolate_retarget(source_data, target_data, alpha=0.5):
    """Blend between two skeletons by linear interpolation.

    Useful for morphing between body types or smoothing retargeting results.
    Both skeletons must have the same number of frames and tracked points.

    Args:
        source_data: (numFrames, numTrackedPoints, 3) array
        target_data: (numFrames, numTrackedPoints, 3) array
        alpha: blend weight, 0.0 = fully source, 1.0 = fully target

    Returns:
        blended: (numFrames, numTrackedPoints, 3) array
        info: dict with blend details
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    source_nan = np.isnan(source_data)
    target_nan = np.isnan(target_data)

    source_clean = np.where(source_nan, 0.0, source_data)
    target_clean = np.where(target_nan, 0.0, target_data)

    blended = source_clean * (1.0 - alpha) + target_clean * alpha

    both_valid = ~source_nan & ~target_nan
    source_only = ~source_nan & target_nan
    target_only = source_nan & ~target_nan
    neither_valid = source_nan & target_nan

    blended[source_only] = source_clean[source_only]
    blended[target_only] = target_clean[target_only]
    blended[neither_valid] = np.nan

    source_props = compute_skeleton_proportions(source_data)
    target_props = compute_skeleton_proportions(target_data)
    blended_props = compute_skeleton_proportions(blended)

    return blended, {
        "alpha": alpha,
        "source_height": source_props["total_height"],
        "target_height": target_props["total_height"],
        "blended_height": blended_props["total_height"],
        "num_frames": blended.shape[0],
        "num_points": blended.shape[1],
    }
