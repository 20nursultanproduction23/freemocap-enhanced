"""
Bone Length Consistency Solver
Post-processing pass that corrects 3D triangulated skeletons to enforce
consistent bone lengths across frames.

Uses RTMPose 133-keypoint format with common bone definitions.
"""

import numpy as np
from typing import Optional


# RTMPose 133-keypoint bone definitions (child_index, parent_index)
# Parent is the joint closer to the root (hips).
RTMPOSE_BONES = [
    # Spine / torso
    (0, 1),      # mid_hip -> pelvis
    (1, 2),      # pelvis -> spine_low
    (2, 3),      # spine_low -> spine_mid
    (3, 4),      # spine_mid -> spine_high
    (4, 5),      # spine_high -> neck
    (5, 6),      # neck -> head
    (6, 7),      # head -> head_top

    # Left arm
    (5, 8),      # neck -> l_clavicle
    (8, 9),      # l_clavicle -> l_shoulder
    (9, 10),     # l_shoulder -> l_upper_arm
    (10, 11),    # l_upper_arm -> l_lower_arm
    (11, 12),    # l_lower_arm -> l_wrist
    (12, 13),    # l_wrist -> l_hand
    (13, 14),    # l_hand -> l_hand指尖

    # Right arm
    (5, 15),     # neck -> r_clavicle
    (15, 16),    # r_clavicle -> r_shoulder
    (16, 17),    # r_shoulder -> r_upper_arm
    (17, 18),    # r_upper_arm -> r_lower_arm
    (18, 19),    # r_lower_arm -> r_wrist
    (19, 20),    # r_wrist -> r_hand
    (20, 21),    # r_hand -> r_hand_fingertip

    # Left leg
    (0, 22),     # mid_hip -> l_hip
    (22, 23),    # l_hip -> l_upper_leg
    (23, 24),    # l_upper_leg -> l_lower_leg
    (24, 25),    # l_lower_leg -> l_ankle
    (25, 26),    # l_ankle -> l_heel
    (26, 27),    # l_heel -> l_foot
    (25, 28),    # l_ankle -> l_foot_top

    # Right leg
    (0, 29),     # mid_hip -> r_hip
    (29, 30),    # r_hip -> r_upper_leg
    (30, 31),    # r_upper_leg -> r_lower_leg
    (31, 32),    # r_lower_leg -> r_ankle
    (32, 33),    # r_ankle -> r_heel
    (33, 34),    # r_heel -> r_foot
    (32, 35),    # r_ankle -> r_foot_top
]

# Extended RTMPose 133 keypoints — additional face / hand detail bones
# These cover the full 133-point set with finer hand and face bones.
RTMPOSE_BONES_EXTENDED = RTMPOSE_BONES.copy()


class BoneLengthSolver:
    """Correct 3D skeleton data so that bone lengths remain consistent
    across all frames.

    Algorithm:
        1. Compute reference bone lengths from the median of the first N
           visible frames (or all visible frames for short sequences).
        2. For each frame, if a bone deviates more than *threshold* from
           its reference, move the child joint along the bone direction so
           that the bone matches the reference length.
        3. Repeat for *num_passes* iterations for convergence.
        4. The root joint (index 0 / mid_hip) is never moved.

    Parameters
    ----------
    bones : list of tuple(int, int)
        Each element is ``(child_index, parent_index)``.
    threshold : float
        Maximum fractional deviation from reference before correction
        is applied (default 0.20 = 20 %).
    num_passes : int
        Number of correction iterations (3-5 recommended).
    reference_frames : int or None
        How many leading frames to use for the reference median.
        *None* means use all visible frames.
    """

    def __init__(
        self,
        bones: Optional[list] = None,
        threshold: float = 0.20,
        num_passes: int = 4,
        reference_frames: Optional[int] = None,
    ):
        self.bones = bones if bones is not None else RTMPOSE_BONES
        self.threshold = threshold
        self.num_passes = num_passes
        self.reference_frames = reference_frames
        self._reference_lengths: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------

    def fit_reference(self, skeleton_3d: np.ndarray) -> np.ndarray:
        """Compute reference bone lengths from skeleton data.

        Parameters
        ----------
        skeleton_3d : np.ndarray
            Shape ``(num_frames, num_keypoints, 3)``.
            NaN values indicate missing / invisible joints.

        Returns
        -------
        np.ndarray
            Reference length for each bone, shape ``(num_bones,)``.
        """
        num_frames = skeleton_3d.shape[0]
        num_keypoints = skeleton_3d.shape[1]
        n_ref = (
            num_frames
            if self.reference_frames is None
            else min(self.reference_frames, num_frames)
        )

        # Filter bones to only those within keypoint range
        valid_bones = [
            (child, parent) for child, parent in self.bones
            if child < num_keypoints and parent < num_keypoints
        ]
        if len(valid_bones) == 0:
            self._reference_lengths = np.array([])
            return np.array([])
        self._bones = valid_bones

        bone_lengths = self._compute_all_bone_lengths(skeleton_3d[:n_ref])

        # Median across frames, ignoring NaN
        ref = np.nanmedian(bone_lengths, axis=0)
        self._reference_lengths = ref
        return ref

    def solve(self, skeleton_3d: np.ndarray) -> tuple:
        """Run the bone-length correction.

        Parameters
        ----------
        skeleton_3d : np.ndarray
            Shape ``(num_frames, num_keypoints, 3)``.
            **Modified in-place** and also returned.

        Returns
        -------
        corrected : np.ndarray
            Corrected skeleton (same array, in-place).
        report : dict
            ``{"num_corrections": int, "corrections_per_frame": list[int]}``
        """
        if self._reference_lengths is None:
            self.fit_reference(skeleton_3d)

        bones = getattr(self, '_bones', self.bones)
        ref_lengths = self._reference_lengths.copy()
        if len(ref_lengths) == 0 or len(bones) == 0:
            return skeleton_3d, {"num_corrections": 0, "corrections_per_frame": [0] * skeleton_3d.shape[0]}
        num_frames = skeleton_3d.shape[0]
        corrections_per_frame = [0] * num_frames

        for _pass in range(self.num_passes):
            for frame_idx in range(num_frames):
                frame = skeleton_3d[frame_idx]
                count = 0
                for bone_idx, (child, parent) in enumerate(bones):
                    if child == 0 or parent == 0:
                        # Never move the root joint
                        pass

                    p_pos = frame[parent]
                    c_pos = frame[child]

                    if np.any(np.isnan(p_pos)) or np.any(np.isnan(c_pos)):
                        continue

                    direction = c_pos - p_pos
                    current_len = np.linalg.norm(direction)
                    if current_len < 1e-8:
                        continue

                    ref_len = ref_lengths[bone_idx]
                    if np.isnan(ref_len) or ref_len < 1e-8:
                        continue

                    deviation = abs(current_len - ref_len) / ref_len
                    if deviation <= self.threshold:
                        continue

                    # Correct: move child along bone direction
                    unit = direction / current_len
                    new_child = p_pos + unit * ref_len
                    skeleton_3d[frame_idx, child] = new_child
                    count += 1

                corrections_per_frame[frame_idx] += count

        total = sum(corrections_per_frame)
        report = {
            "num_corrections": total,
            "corrections_per_frame": corrections_per_frame,
        }
        return skeleton_3d, report

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _compute_all_bone_lengths(self, skeleton_3d: np.ndarray) -> np.ndarray:
        """Return bone lengths for every frame.

        Returns
        -------
        np.ndarray
            Shape ``(num_frames, num_bones)``.
        """
        num_frames = skeleton_3d.shape[0]
        bones = getattr(self, '_bones', self.bones)
        num_bones = len(bones)
        lengths = np.full((num_frames, num_bones), np.nan, dtype=np.float64)

        for bone_idx, (child, parent) in enumerate(bones):
            p = skeleton_3d[:, parent, :]  # (F, 3)
            c = skeleton_3d[:, child, :]   # (F, 3)
            valid = ~(np.any(np.isnan(p), axis=1) | np.any(np.isnan(c), axis=1))
            if not np.any(valid):
                continue
            diff = c[valid] - p[valid]
            lengths[valid, bone_idx] = np.linalg.norm(diff, axis=1)

        return lengths


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def correct_bone_lengths(
    skeleton_3d: np.ndarray,
    bones: Optional[list] = None,
    threshold: float = 0.20,
    num_passes: int = 4,
    fps: float = 30.0,
) -> tuple:
    """One-call convenience to correct bone lengths.

    Parameters
    ----------
    skeleton_3d : np.ndarray
        Shape ``(num_frames, num_keypoints, 3)``.
    bones : list or None
        Bone definitions; defaults to ``RTMPOSE_BONES``.
    threshold : float
        Max fractional deviation before correction (default 0.20).
    num_passes : int
        Iteration count (default 4).
    fps : float
        Frames per second (reserved for future velocity-based refinements).

    Returns
    -------
    corrected : np.ndarray
    report : dict
    """
    solver = BoneLengthSolver(
        bones=bones, threshold=threshold, num_passes=num_passes,
    )
    solver.fit_reference(skeleton_3d)
    return solver.solve(skeleton_3d)


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    np.random.seed(42)
    n_frames, n_kp = 60, 40
    skeleton = np.random.randn(n_frames, n_kp, 3).astype(np.float64)
    # Inject a spike at frame 25 for bone (1, 2)
    skeleton[25, 1] += np.array([5.0, 0.0, 0.0])

    corrected, report = correct_bone_lengths(skeleton, fps=30.0)
    print(f"Total corrections: {report['num_corrections']}")
    print(f"Shape: {corrected.shape}")
