import numpy as np


def assess_hand_quality(skeleton_data, hand_start, hand_count):
    """Assess per-frame quality of a hand segment in FreeMoCap skeleton data.

    Returns a (numFrames,) array of quality scores in [0, 1]. Quality is
    computed as the fraction of non-NaN keypoints weighted by temporal
    smoothness (frame-to-frame consistency of keypoint positions).

    Parameters
    ----------
    skeleton_data : np.ndarray, shape (numFrames, 553, 3)
        FreeMoCap skeleton data (x, y, confidence per keypoint).
    hand_start : int
        Index of the first keypoint for this hand in the 553-keypoint layout.
    hand_count : int
        Number of keypoints belonging to this hand.

    Returns
    -------
    np.ndarray, shape (numFrames,)
        Per-frame quality scores in [0, 1].
    """
    hand_data = skeleton_data[:, hand_start:hand_start + hand_count, :]
    num_frames = hand_data.shape[0]

    xyz = hand_data[:, :, :3]
    nan_mask = np.isnan(xyz).any(axis=2)
    non_nan_fraction = 1.0 - nan_mask.sum(axis=1) / float(hand_count)

    temporal_smoothness = np.ones(num_frames)
    if num_frames > 1:
        diffs = np.diff(xyz, axis=0)
        finite_mask = np.isfinite(diffs)
        diffs_clean = np.where(finite_mask, diffs, 0.0)
        disp = np.sqrt(np.sum(diffs_clean ** 2, axis=2))
        max_keypoints = hand_count * 3
        frame_disp = disp.sum(axis=1)
        median_disp = np.median(frame_disp) if np.median(frame_disp) > 0 else 1.0
        temporal_smoothness[1:] = np.clip(1.0 - frame_disp / (median_disp * 5.0 + 1e-8), 0.0, 1.0)

    quality = non_nan_fraction * temporal_smoothness
    return np.clip(quality, 0.0, 1.0)


def ensemble_hands(freemocap_data, rtmpose_data, rtmpose_confidence,
                   quality_threshold=0.3, blend_width=5):
    """Blend RTMPose left-hand detections into MediaPipe output where quality is poor.

    Only modifies the left-hand region (keypoint indices 54:75) of the output.
    Body skeleton and right-hand data from MediaPipe are preserved unchanged.

    Parameters
    ----------
    freemocap_data : np.ndarray, shape (numFrames, 553, 3)
        MediaPipe skeleton output (possibly post-pipeline).
    rtmpose_data : np.ndarray, shape (numFrames, 553, 3)
        RTMPose skeleton output already converted to FreeMoCap format.
    rtmpose_confidence : np.ndarray, shape (numFrames, 21)
        RTMPose confidence scores for the left hand segment.
    quality_threshold : float
        Below this confidence value the frame is considered unreliable for
        the MediaPipe left hand, triggering substitution with RTMPose data.
    blend_width : int
        Number of frames over which to cross-fade at transition boundaries.

    Returns
    -------
    np.ndarray, shape (numFrames, 553, 3)
        Blended skeleton data with RTMPose left-hand substitution applied.
    """
    if freemocap_data.shape != rtmpose_data.shape:
        raise ValueError(
            f"Shape mismatch: freemocap_data {freemocap_data.shape} "
            f"!= rtmpose_data {rtmpose_data.shape}"
        )

    hand_start = 54
    hand_count = 21
    num_frames = freemocap_data.shape[0]

    output = freemocap_data.copy()

    mp_quality = assess_hand_quality(freemocap_data, hand_start, hand_count)
    rtmpose_hand_conf = np.mean(rtmpose_confidence, axis=1) if rtmpose_confidence.ndim > 1 else rtmpose_confidence.copy()

    use_rtmpose = np.zeros(num_frames, dtype=bool)
    for f in range(num_frames):
        if rtmpose_hand_conf[f] > quality_threshold:
            if mp_quality[f] < quality_threshold or np.any(np.isnan(freemocap_data[f, hand_start:hand_start + hand_count, :])):
                use_rtmpose[f] = True

    transition_frames = []
    for f in range(num_frames):
        if use_rtmpose[f]:
            prev_rtmpose = use_rtmpose[f - 1] if f > 0 else False
            if not prev_rtmpose:
                transition_frames.append(f)
        elif f > 0 and use_rtmpose[f - 1]:
            transition_frames.append(f)

    blend_weights = np.zeros(num_frames)
    blend_weights[use_rtmpose] = 1.0

    for tf in transition_frames:
        if tf > 0 and use_rtmpose[tf - 1] and not use_rtmpose[tf]:
            for w in range(1, blend_width + 1):
                idx = tf + w
                if idx >= num_frames:
                    break
                if use_rtmpose[idx]:
                    break
                weight = 1.0 - w / (blend_width + 1)
                blend_weights[idx] = max(blend_weights[idx], 1.0 - weight)
        elif use_rtmpose[tf]:
            for w in range(1, blend_width + 1):
                idx = tf - w
                if idx < 0:
                    break
                if not use_rtmpose[idx]:
                    break
                weight = w / (blend_width + 1)
                blend_weights[idx] = max(blend_weights[idx], weight)

    for f in range(num_frames):
        w = blend_weights[f]
        if w > 0.0:
            mp_hand = freemocap_data[f, hand_start:hand_start + hand_count, :]
            rt_hand = rtmpose_data[f, hand_start:hand_start + hand_count, :]
            mp_nan = np.isnan(mp_hand).any(axis=1)[:, None]
            rt_nan = np.isnan(rt_hand).any(axis=1)[:, None]

            mp_clean = np.where(mp_nan, 0.0, mp_hand)
            rt_clean = np.where(rt_nan, 0.0, rt_hand)

            blended = (1.0 - w) * mp_clean + w * rt_clean

            both_nan = mp_nan & rt_nan
            blended[both_nan[:, 0]] = np.nan

            output[f, hand_start:hand_start + hand_count, :] = blended

    return output


def compare_trackers(freemocap_data, rtmpose_data, hand="left"):
    """Compare MediaPipe and RTMPose tracking for a specific hand.

    Returns a dict with displacement statistics, correlation, and agreement
    counts that help decide whether ensemble blending is beneficial.

    Parameters
    ----------
    freemocap_data : np.ndarray, shape (numFrames, 553, 3)
        MediaPipe skeleton output.
    rtmpose_data : np.ndarray, shape (numFrames, 553, 3)
        RTMPose skeleton output in FreeMoCap format.
    hand : str
        ``"left"`` or ``"right"``.

    Returns
    -------
    dict
        ``mean_displacement``: mean Euclidean distance between corresponding
        non-NaN keypoints across all frames.
        ``correlation``: Pearson correlation of the mean-displacement
        time-series between the two trackers.
        ``agreement_frames``: count of frames where mean displacement < 20 mm.
        ``disagreement_frames``: count of frames where mean displacement >= 20 mm.
    """
    if hand.lower() == "left":
        hand_start = 54
    elif hand.lower() == "right":
        hand_start = 75
    else:
        raise ValueError(f"hand must be 'left' or 'right', got '{hand}'")

    hand_count = 21
    mp_hand = freemocap_data[:, hand_start:hand_start + hand_count, :]
    rt_hand = rtmpose_data[:, hand_start:hand_start + hand_count, :]
    num_frames = mp_hand.shape[0]

    valid = np.isfinite(mp_hand).all(axis=2) & np.isfinite(rt_hand).all(axis=2)
    diff = rt_hand - mp_hand
    diff_masked = np.where(valid[:, :, np.newaxis], diff, 0.0)
    dist = np.sqrt(np.sum(diff_masked ** 2, axis=2))
    per_frame_count = valid.sum(axis=1).astype(float)
    per_frame_count[per_frame_count == 0] = 1.0
    mean_disp_per_frame = dist.sum(axis=1) / per_frame_count
    mean_displacement = float(np.nanmean(mean_disp_per_frame))

    valid_frames = np.isfinite(mean_disp_per_frame)
    if valid_frames.sum() > 1:
        valid_3d = valid[:, :, np.newaxis]
        mp_center = np.nanmean(np.where(valid_3d, mp_hand[:, :, :2], np.nan), axis=1)
        rt_center = np.nanmean(np.where(valid_3d, rt_hand[:, :, :2], np.nan), axis=1)
        mp_center_flat = mp_center[valid_frames].flatten()
        rt_center_flat = rt_center[valid_frames].flatten()
        if np.std(mp_center_flat) > 1e-8 and np.std(rt_center_flat) > 1e-8:
            correlation = float(np.corrcoef(mp_center_flat, rt_center_flat)[0, 1])
        else:
            correlation = 0.0
    else:
        correlation = 0.0

    agreement_threshold = 20.0
    agreement_frames = int(np.sum(mean_disp_per_frame < agreement_threshold))
    disagreement_frames = int(np.sum(mean_disp_per_frame >= agreement_threshold))

    return {
        "mean_displacement": mean_displacement,
        "correlation": correlation,
        "agreement_frames": agreement_frames,
        "disagreement_frames": disagreement_frames,
    }
