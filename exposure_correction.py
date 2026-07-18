import cv2
import numpy as np
import os
import tempfile


def apply_clahe_to_frame(frame, clip_limit=2.0, grid_size=8, color_space="LAB"):
    """Apply CLAHE to a single frame to equalize brightness without affecting color.

    Args:
        frame: (H, W, 3) BGR uint8 image in OpenCV format.
        clip_limit: Threshold for contrast limiting. Higher values give more contrast.
        grid_size: Size of the grid for adaptive histogram equalization.
        color_space: Color space to use — "LAB" (L channel) or "HSV" (V channel).

    Returns:
        Corrected (H, W, 3) BGR uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    if color_space.upper() == "LAB":
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(converted)
        l = clahe.apply(l)
        corrected = cv2.merge([l, a, b])
        return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
    elif color_space.upper() == "HSV":
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(converted)
        v = clahe.apply(v)
        corrected = cv2.merge([h, s, v])
        return cv2.cvtColor(corrected, cv2.COLOR_HSV2BGR)
    else:
        raise ValueError(f"Unsupported color_space: {color_space}. Use 'LAB' or 'HSV'.")


def apply_clahe_to_video(video_path, output_path=None, clip_limit=2.0, grid_size=8, frame_range=None):
    """Process an entire video with CLAHE frame-by-frame.

    Args:
        video_path: Path to the input video file.
        output_path: Path for the output video. If None, frames are returned as numpy array.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of the grid for adaptive histogram equalization.
        frame_range: Optional (start, end) tuple (inclusive) to process only specific frames.

    Returns:
        Dict with keys:
            - 'output_path' or 'frames': output video path or numpy array of frames
            - 'frames_processed': number of frames processed
            - 'mean_brightness_before': average brightness of original frames
            - 'mean_brightness_after': average brightness of corrected frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_range is not None:
        start, end = frame_range
        start = max(0, start)
        end = min(total_frames - 1, end)
    else:
        start = 0
        end = total_frames - 1

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_before = []
    frames_after = []
    frames_processed = 0
    collected_frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_idx in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        brightness_before = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        frames_before.append(brightness_before)

        corrected = apply_clahe_to_frame(frame, clip_limit=clip_limit, grid_size=grid_size)

        brightness_after = float(np.mean(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)))
        frames_after.append(brightness_after)

        if writer is not None:
            writer.write(corrected)
        else:
            collected_frames.append(corrected)

        frames_processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    mean_before = float(np.mean(frames_before)) if frames_before else 0.0
    mean_after = float(np.mean(frames_after)) if frames_after else 0.0

    result = {
        "frames_processed": frames_processed,
        "mean_brightness_before": mean_before,
        "mean_brightness_after": mean_after,
    }

    if output_path is not None:
        result["output_path"] = output_path
    else:
        result["frames"] = np.array(collected_frames) if collected_frames else np.array([])

    return result


def selective_clahe(video_path, output_path, side="left", side_threshold=0.4, clip_limit=2.0):
    """Apply CLAHE only to the specified side of each frame.

    For frames where the person is positioned on the "bad" side, CLAHE is applied
    to the corresponding half of the frame to compensate for exposure asymmetry.

    Args:
        video_path: Path to the input video file.
        output_path: Path for the output video.
        side: Which side to apply CLAHE to — "left" or "right".
        side_threshold: Not used in the spatial version; retained for API consistency.
        clip_limit: Threshold for contrast limiting.

    Returns:
        Dict with keys:
            - 'output_path': path to the output video
            - 'frames_processed': number of frames processed
            - 'mean_brightness_before': average brightness of original frames
            - 'mean_brightness_after': average brightness of corrected frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    half_width = width // 2
    frames_processed = 0
    brightness_before_list = []
    brightness_after_list = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        brightness_before_list.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

        corrected = frame.copy()

        if side == "left":
            region = corrected[:, :half_width]
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab_corrected = cv2.merge([l, a, b])
            corrected[:, :half_width] = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        elif side == "right":
            region = corrected[:, half_width:]
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab_corrected = cv2.merge([l, a, b])
            corrected[:, half_width:] = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        else:
            raise ValueError(f"Invalid side: {side}. Use 'left' or 'right'.")

        brightness_after_list.append(float(np.mean(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))))

        writer.write(corrected)
        frames_processed += 1

    cap.release()
    writer.release()

    return {
        "output_path": output_path,
        "frames_processed": frames_processed,
        "mean_brightness_before": float(np.mean(brightness_before_list)) if brightness_before_list else 0.0,
        "mean_brightness_after": float(np.mean(brightness_after_list)) if brightness_after_list else 0.0,
    }


def analyze_brightness(video_path, num_samples=50):
    """Sample frames from a video and compute brightness statistics.

    Args:
        video_path: Path to the video file.
        num_samples: Number of frames to sample for analysis.

    Returns:
        Dict with keys:
            - 'mean_brightness': average brightness across sampled frames (0-255)
            - 'std_brightness': standard deviation of brightness
            - 'dark_frames_pct': percentage of frames below brightness 60
            - 'bright_frames_pct': percentage of frames above brightness 200
            - 'recommended_clip_limit': suggested CLAHE clip_limit based on distribution
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return {
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
            "dark_frames_pct": 0.0,
            "bright_frames_pct": 0.0,
            "recommended_clip_limit": 2.0,
        }

    sample_indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    brightness_values = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

    cap.release()

    if not brightness_values:
        return {
            "mean_brightness": 0.0,
            "std_brightness": 0.0,
            "dark_frames_pct": 0.0,
            "bright_frames_pct": 0.0,
            "recommended_clip_limit": 2.0,
        }

    brightness_array = np.array(brightness_values)
    mean_bright = float(np.mean(brightness_array))
    std_bright = float(np.std(brightness_array))
    dark_pct = float(np.sum(brightness_array < 60) / len(brightness_array) * 100)
    bright_pct = float(np.sum(brightness_array > 200) / len(brightness_array) * 100)

    if std_bright > 60 or dark_pct > 30:
        recommended_clip = 4.0
    elif std_bright > 30 or dark_pct > 15:
        recommended_clip = 3.0
    elif mean_bright < 80:
        recommended_clip = 2.5
    else:
        recommended_clip = 2.0

    return {
        "mean_brightness": mean_bright,
        "std_brightness": std_bright,
        "dark_frames_pct": dark_pct,
        "bright_frames_pct": bright_pct,
        "recommended_clip_limit": recommended_clip,
    }


def preprocess_for_detection(video_path, target_side=None, clip_limit=None):
    """Main entry point: analyze video and apply CLAHE preprocessing if needed.

    Determines whether CLAHE correction is necessary based on brightness analysis,
    then applies either full-frame or selective CLAHE accordingly.

    Args:
        video_path: Path to the input video file.
        target_side: If specified (e.g., "left"), applies selective CLAHE to that side.
        clip_limit: CLAHE clip_limit. If None, auto-detected from brightness analysis.

    Returns:
        Dict with keys:
            - 'output_path': path to the preprocessed video
            - 'clahe_applied': whether CLAHE was applied (bool)
            - 'analysis': brightness analysis stats dict
    """
    analysis = analyze_brightness(video_path)

    needs_clahe = (
        analysis["mean_brightness"] < 80
        or analysis["dark_frames_pct"] > 20
        or analysis["std_brightness"] > 40
    )

    if not needs_clahe:
        return {
            "output_path": video_path,
            "clahe_applied": False,
            "analysis": analysis,
        }

    effective_clip_limit = clip_limit if clip_limit is not None else analysis["recommended_clip_limit"]

    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_clahe{ext}"

    if target_side is not None:
        result = selective_clahe(
            video_path,
            output_path,
            side=target_side,
            clip_limit=effective_clip_limit,
        )
    else:
        result = apply_clahe_to_video(
            video_path,
            output_path,
            clip_limit=effective_clip_limit,
        )

    return {
        "output_path": output_path,
        "clahe_applied": True,
        "analysis": analysis,
        "processing_info": result,
    }
