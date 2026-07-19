"""
Stage 6: Physical Interaction Validation

Detects interpenetration and contact events between two actors' 3D skeletons.
Outputs per-frame lists of InteractionEvent / InterpenetrationEvent.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class InteractionEvent:
    frame: int
    actor_a: int
    actor_b: int
    joint_a: int
    joint_b: int
    distance: float
    joint_a_label: str = ""
    joint_b_label: str = ""


@dataclass
class InterpenetrationEvent:
    frame: int
    actor_a: int
    actor_b: int
    overlap_fraction: float


BODY_PART_LABELS = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}


def _get_valid_joints(skeleton: np.ndarray) -> np.ndarray:
    """Return indices of joints that are not NaN. Expects (num_joints, 3)."""
    if skeleton.ndim != 2 or skeleton.shape[1] != 3:
        return np.array([], dtype=int)
    valid = ~np.isnan(skeleton).any(axis=1)
    return np.where(valid)[0]


def interpenetration_detection(
    actor_data: List[np.ndarray],
    threshold: float = 0.15,
    overlap_ratio_threshold: float = 0.3,
) -> List[InterpenetrationEvent]:
    """
    For each frame, check if a significant fraction of joints from actor A
    are closer than `threshold` to the nearest joint in actor B.

    Parameters
    ----------
    actor_data : list of np.ndarray
        Each element is (num_frames, num_joints, 3).
    threshold : float
        Distance threshold in meters.
    overlap_ratio_threshold : float
        Fraction of joints that must be within threshold to flag interpenetration.

    Returns
    -------
    list of InterpenetrationEvent
    """
    num_actors = len(actor_data)
    if num_actors < 2:
        return []

    num_frames = actor_data[0].shape[0]
    events = []

    for actor_i in range(num_actors):
        for actor_j in range(actor_i + 1, num_actors):
            for frame_idx in range(num_frames):
                skel_a = actor_data[actor_i][frame_idx]
                skel_b = actor_data[actor_j][frame_idx]

                valid_a = _get_valid_joints(skel_a)
                valid_b = _get_valid_joints(skel_b)

                if len(valid_a) == 0 or len(valid_b) == 0:
                    continue

                pts_a = skel_a[valid_a]
                pts_b = skel_b[valid_b]

                # For each joint in A, find nearest joint in B
                distances = np.linalg.norm(
                    pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :], axis=2
                )
                min_distances = distances.min(axis=1)

                overlap_fraction = np.mean(min_distances < threshold)

                if overlap_fraction >= overlap_ratio_threshold:
                    events.append(
                        InterpenetrationEvent(
                            frame=frame_idx,
                            actor_a=actor_i,
                            actor_b=actor_j,
                            overlap_fraction=float(overlap_fraction),
                        )
                    )

    return events


def contact_detection(
    actor_data: List[np.ndarray],
    contact_threshold: float = 0.05,
    joint_labels: Optional[dict] = None,
) -> List[InteractionEvent]:
    """
    For each frame, find all joint pairs (one from actor A, one from actor B)
    that are within `contact_threshold` meters.

    Parameters
    ----------
    actor_data : list of np.ndarray
        Each element is (num_frames, num_joints, 3).
    contact_threshold : float
        Distance threshold in meters for contact.
    joint_labels : dict, optional
        Mapping from joint index to label string.

    Returns
    -------
    list of InteractionEvent
    """
    if joint_labels is None:
        joint_labels = BODY_PART_LABELS

    num_actors = len(actor_data)
    if num_actors < 2:
        return []

    num_frames = actor_data[0].shape[0]
    events = []

    for actor_i in range(num_actors):
        for actor_j in range(actor_i + 1, num_actors):
            for frame_idx in range(num_frames):
                skel_a = actor_data[actor_i][frame_idx]
                skel_b = actor_data[actor_j][frame_idx]

                valid_a = _get_valid_joints(skel_a)
                valid_b = _get_valid_joints(skel_b)

                if len(valid_a) == 0 or len(valid_b) == 0:
                    continue

                for ja in valid_a:
                    for jb in valid_b:
                        dist = float(np.linalg.norm(skel_a[ja] - skel_b[jb]))
                        if dist < contact_threshold:
                            events.append(
                                InteractionEvent(
                                    frame=frame_idx,
                                    actor_a=actor_i,
                                    actor_b=actor_j,
                                    joint_a=int(ja),
                                    joint_b=int(jb),
                                    distance=dist,
                                    joint_a_label=joint_labels.get(int(ja), ""),
                                    joint_b_label=joint_labels.get(int(jb), ""),
                                )
                            )

    return events
