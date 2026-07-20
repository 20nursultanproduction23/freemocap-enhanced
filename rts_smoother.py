"""
RTS (Rauch–Tung–Striebel) Smoother for 3D Skeleton Data

Post-processing Kalman smoother that reduces jitter in triangulated
skeleton data.  Per-keypoint independent filtering using a 6-state
constant-velocity model ``[x, y, z, vx, vy, vz]``.

Dependencies: only numpy.
"""

import numpy as np
from typing import Optional


class RTSSmoother:
    """Rauch–Tung–Striebel Kalman smoother for 3-D skeleton sequences.

    Each keypoint is filtered independently with a constant-velocity model.

    State
    -----
    ``[x, y, z, vx, vy, vz]``

    Transition (dt = 1/fps)
    -----------------------
    ``F = [[I, dt*I], [0, I]]``

    Observation
    -----------
    ``H = [[I, 0]]``  (observe position only)

    Parameters
    ----------
    fps : float
        Frames per second (determines dt).
    process_noise : float
        Standard deviation of acceleration noise (Q base scale).
        Higher values trust measurements more.
    observation_noise : float
        Standard deviation of measurement noise (R scale).
        Higher values smooth more aggressively.
    """

    def __init__(
        self,
        fps: float = 30.0,
        process_noise: float = 1.0,
        observation_noise: float = 10.0,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.q_scale = process_noise
        self.r_scale = observation_noise

        self._build_matrices()

    # ------------------------------------------------------------------
    # matrix construction
    # ------------------------------------------------------------------

    def _build_matrices(self) -> None:
        dt = self.dt

        # State transition  (6x6)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Observation matrix  (3x6)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise  (6x6)
        q = self.q_scale ** 2
        self.Q = np.diag([
            q * dt ** 4 / 4,
            q * dt ** 4 / 4,
            q * dt ** 4 / 4,
            q * dt ** 2,
            q * dt ** 2,
            q * dt ** 2,
        ])

        # Observation noise  (3x3)
        r = self.r_scale ** 2
        self.R = np.eye(3) * r

    # ------------------------------------------------------------------
    # core Kalman filter + RTS smoother
    # ------------------------------------------------------------------

    def _filter_single_keypoint(self, observations: np.ndarray) -> tuple:
        """Forward Kalman filter for one keypoint across all frames.

        Parameters
        ----------
        observations : np.ndarray
            Shape ``(num_frames, 3)``.  NaN rows indicate missing data.

        Returns
        -------
        means : np.ndarray
            Shape ``(num_frames, 6)`` — filtered state estimates.
        covariances : np.ndarray
            Shape ``(num_frames, 6, 6)`` — filtered covariances.
        valid : np.ndarray
            Boolean mask, True where observation was valid.
        """
        n = observations.shape[0]
        means = np.zeros((n, 6), dtype=np.float64)
        covs = np.zeros((n, 6, 6), dtype=np.float64)
        valid = ~np.any(np.isnan(observations), axis=1)

        # Initialise state from first valid observation
        first_valid = np.argmax(valid) if np.any(valid) else 0
        if valid[first_valid]:
            means[first_valid, :3] = observations[first_valid]
            covs[first_valid] = np.eye(6) * 1000.0  # large initial uncertainty
        else:
            covs[first_valid] = np.eye(6) * 1000.0

        # Forward pass
        x = means[first_valid].copy()
        P = covs[first_valid].copy()
        F = self.F
        H = self.H
        Q = self.Q
        R = self.R

        for i in range(first_valid, n):
            if i > first_valid:
                # Predict
                x = F @ x
                P = F @ P @ F.T + Q

            if valid[i]:
                # Update
                z = observations[i]
                y = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(6) - K @ H) @ P

            means[i] = x
            covs[i] = P

        # Back-fill frames before first valid
        for i in range(first_valid - 1, -1, -1):
            means[i] = means[first_valid]
            covs[i] = covs[first_valid]

        return means, covs, valid

    def _smooth_single_keypoint(
        self,
        filtered_means: np.ndarray,
        filtered_covs: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """RTS backward pass for one keypoint.

        Parameters
        ----------
        filtered_means : np.ndarray
            Shape ``(n, 6)``
        filtered_covs : np.ndarray
            Shape ``(n, 6, 6)``
        valid : np.ndarray
            Boolean mask.

        Returns
        -------
        smoothed : np.ndarray
            Shape ``(n, 6)`` — smoothed state estimates.
        """
        n = filtered_means.shape[0]
        F = self.F
        smoothed = filtered_means.copy()
        smooth_covs = filtered_covs.copy()

        # Find last valid frame to start smoothing from
        last_valid = n - 1
        while last_valid >= 0 and not valid[last_valid]:
            last_valid -= 1
        if last_valid < 0:
            return smoothed

        for i in range(last_valid - 1, -1, -1):
            # Predicted state and covariance from previous step
            x_pred = F @ filtered_means[i]
            P_pred = F @ filtered_covs[i] @ F.T + self.Q

            # Check for NaN / degenerate covariance
            det = np.linalg.det(P_pred)
            if det < 1e-12 or not np.isfinite(det):
                continue

            # RTS gain
            G = filtered_covs[i] @ F.T @ np.linalg.inv(P_pred)

            # Smoothed state
            smoothed[i] = filtered_means[i] + G @ (
                smoothed[i + 1] - x_pred
            )
            smooth_covs[i] = filtered_covs[i] + G @ (
                smooth_covs[i + 1] - P_pred
            ) @ G.T

        return smoothed[:, :3]  # return position components

    # ------------------------------------------------------------------
    # interpolation helpers for NaN gaps
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_gaps(
        data: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate NaN gaps in *data* using valid frames.

        Handles leading / trailing NaNs by constant extrapolation.
        """
        n = data.shape[0]
        out = data.copy()
        if np.all(valid):
            return out

        # Forward fill then backward fill for edges
        last_valid_val = None
        next_valid_idx = None

        # First pass: find next valid index for each position
        next_valid = np.full(n, -1, dtype=int)
        last_v = -1
        for i in range(n - 1, -1, -1):
            if valid[i]:
                last_v = i
            next_valid[i] = last_v

        # Second pass: find previous valid index
        prev_valid_idx = -1
        for i in range(n):
            if valid[i]:
                prev_valid_idx = i
            if not valid[i]:
                pv = prev_valid_idx
                nv = next_valid[i]
                if pv >= 0 and nv >= 0 and nv != pv:
                    # Linear interpolation
                    t = (i - pv) / (nv - pv)
                    out[i] = (1.0 - t) * data[pv] + t * data[nv]
                elif pv >= 0:
                    out[i] = data[pv]
                elif nv >= 0:
                    out[i] = data[nv]

        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def smooth(
        self,
        skeleton_3d: np.ndarray,
        interpolate_gaps: bool = True,
    ) -> dict:
        """Apply RTS smoothing to a skeleton sequence.

        Parameters
        ----------
        skeleton_3d : np.ndarray
            Shape ``(num_frames, num_keypoints, 3)``.
            NaN values indicate missing / invisible joints.
        interpolate_gaps : bool
            If True, linearly interpolate across NaN gaps after smoothing.

        Returns
        -------
        dict
            ``smoothed``  : np.ndarray — smoothed skeleton, same shape.
            ``original``  : np.ndarray — copy of the input for comparison.
            ``num_nan_frames`` : int — total frames that had NaN values.
        """
        original = skeleton_3d.copy()
        num_frames, num_kp, _ = skeleton_3d.shape
        smoothed_pos = np.zeros_like(skeleton_3d, dtype=np.float64)
        total_nan_frames = 0

        for kp in range(num_kp):
            obs = skeleton_3d[:, kp, :]  # (n, 3)
            has_nan = np.any(np.isnan(obs), axis=1)
            total_nan_frames += int(np.sum(has_nan))

            filt_mean, filt_cov, valid = self._filter_single_keypoint(obs)
            smoothed_state = self._smooth_single_keypoint(
                filt_mean, filt_cov, valid,
            )
            if interpolate_gaps:
                smoothed_state = self._interpolate_gaps(smoothed_state, valid)

            smoothed_pos[:, kp, :] = smoothed_state

        return {
            "smoothed": smoothed_pos,
            "original": original,
            "num_nan_frames": total_nan_frames,
        }


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------

def smooth_skeleton(
    skeleton_3d: np.ndarray,
    fps: float = 30.0,
    process_noise: float = 1.0,
    observation_noise: float = 10.0,
    interpolate_gaps: bool = True,
) -> dict:
    """One-call convenience for RTS smoothing.

    Parameters
    ----------
    skeleton_3d : np.ndarray
        Shape ``(num_frames, num_keypoints, 3)``.
    fps : float
        Frames per second.
    process_noise : float
        Process noise scale (higher = trust measurements more).
    observation_noise : float
        Observation noise scale (higher = smooth more).
    interpolate_gaps : bool
        Interpolate across NaN gaps.

    Returns
    -------
    dict with keys ``smoothed``, ``original``, ``num_nan_frames``.
    """
    smoother = RTSSmoother(
        fps=fps,
        process_noise=process_noise,
        observation_noise=observation_noise,
    )
    return smoother.smooth(skeleton_3d, interpolate_gaps=interpolate_gaps)


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    n_frames, n_kp = 120, 50
    ground_truth = np.cumsum(np.random.randn(n_frames, n_kp, 3) * 0.1, axis=0)
    noise = np.random.randn(n_frames, n_kp, 3) * 2.0
    noisy = ground_truth + noise
    # Inject some NaNs
    noisy[30:35, 10, :] = np.nan
    noisy[80:82, 20, :] = np.nan

    result = smooth_skeleton(noisy, fps=30.0)
    print(f"Shape in:  {noisy.shape}")
    print(f"Shape out: {result['smoothed'].shape}")
    print(f"NaN frames: {result['num_nan_frames']}")
    rmse_before = np.sqrt(np.nanmean((noisy - ground_truth) ** 2))
    rmse_after = np.sqrt(np.nanmean((result['smoothed'] - ground_truth) ** 2))
    print(f"RMSE before smoothing: {rmse_before:.4f}")
    print(f"RMSE after smoothing:  {rmse_after:.4f}")
