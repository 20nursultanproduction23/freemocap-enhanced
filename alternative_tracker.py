"""
#4+#6 ALTERNATIVE TRACKER — RTMPose via rtmlib

Replaces MediaPipe Holistic with newer whole-body pose estimators:
- RTMPose Wholebody: 133 keypoints (body 17 + feet 6 + face 68 + hands 21x2),
  ONNX runtime, faster and more accurate than MediaPipe Holistic

These models solve:
- #4 Left/right confusion (trained on more diverse angles)
- #6 Body/hand seam issues (whole-body model, no stitching artifacts)

NOTE: RTMPose per single camera outputs pixel coordinates (x,y) + estimated depth (z),
NOT true 3D world coordinates. For proper 3D reconstruction, multi-camera RTMPose or
DLT calibration is needed. The z values are in camera-space depth (small range).

RTMPose Wholebody133 keypoint ordering:
  [0:17]   = body (COCO 17 keypoints)
  [17:23]  = feet (6 keypoints — ankles/toes)
  [23:91]  = face (68 keypoints)
  [91:112] = left hand (21 keypoints)
  [112:133]= right hand (21 keypoints)

Usage:
    tracker = AlternativeTracker(mode="balanced")
    results = tracker.process_video("input.mp4")
    freemocap_data = tracker.convert_to_freemocap_format(results)
"""
import numpy as np
import os


# RTMPose 133-keypoint segment boundaries
RTMPOSE_SEGMENTS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}

# COCO 17 body -> MediaPipe 33 body mapping
# RTMPose uses COCO body format (17 points)
# FreeMoCap expects MediaPipe body format (33 points)
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 2,    # left_eye (inner)
    2: 5,    # right_eye (inner)
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}


class AlternativeTracker:
    """Wrapper around rtmlib for alternative pose estimation."""

    def __init__(self, model="rtmpose-l", device="cpu", backend="onnxruntime", mode="balanced"):
        """Initialize the alternative tracker.

        Args:
            model: model size (rtmpose-s/m/l)
            device: "cpu" or "cuda"
            backend: "onnxruntime" or "openvino"
            mode: "balanced", "performance", or "lightweight"
        """
        self.model_name = model
        self.device = device
        self.backend = backend
        self.mode = mode
        self._model = None

    def _init_model(self):
        """Initialize rtmlib Wholebody3d model."""
        if self._model is not None:
            return

        print(f"Initializing rtmlib Wholebody3d ({self.model_name}, {self.device})...")

        from rtmlib import Wholebody3d

        self._model = Wholebody3d(
            mode=self.mode,
            backend=self.backend,
            device=self.device,
        )

        print(f"  Model initialized successfully")

    def _parse_result(self, result):
        """Parse RTMPose Wholebody3d output into structured keypoints.

        RTMPose returns a tuple of 4 arrays:
          [0]: keypoints_3d  (num_persons, 133, 3) — x,y in pixel coords, z is depth
          [1]: scores        (num_persons, 133)    — confidence scores
          [2]: keypoints_3d_refined (num_persons, 133, 3) — refined version
          [3]: keypoints_2d  (num_persons, 133, 2) — x,y only

        We take person 0 (highest confidence) and split by segment.

        Returns:
            dict with keys: body, feet, face, left_hand, right_hand
            Each value is (num_keypoints, 3) array with [x, y, score]
        """
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            return None

        kpts3d = result[0]    # (num_persons, 133, 3)
        scores = result[1]    # (num_persons, 133)

        if kpts3d is None or scores is None:
            return None

        num_persons = kpts3d.shape[0]
        if num_persons == 0:
            return None

        # Select person with highest mean confidence
        person_scores = np.mean(scores, axis=1)
        best_person = int(np.argmax(person_scores))

        kpts = kpts3d[best_person]    # (133, 3)
        scrs = scores[best_person]    # (133,)

        # Split by RTMPose segments
        segments = {}
        for name, (start, end) in RTMPOSE_SEGMENTS.items():
            # Combine xyz coords with score as 4th column
            seg_kpts = kpts[start:end]  # (n, 3)
            seg_scrs = scrs[start:end]  # (n,)
            # Store as (n, 4): x, y, z, score
            segments[name] = np.column_stack([seg_kpts, seg_scrs])

        return segments

    def process_video(self, video_path):
        """Process a single video file with RTMPose Wholebody.

        Args:
            video_path: path to input video

        Returns:
            result: dict with keys:
                - body: (numFrames, 17, 4) array [x, y, z, score]
                - feet: (numFrames, 6, 4) array
                - left_hand: (numFrames, 21, 4) array
                - right_hand: (numFrames, 21, 4) array
                - face: (numFrames, 68, 4) array
                - fps: video frame rate
                - image_size: (height, width) of video frames
        """
        self._init_model()

        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing: {video_path}")
        print(f"  Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")

        all_segments = {k: [] for k in RTMPOSE_SEGMENTS}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self._model(frame)
            segments = self._parse_result(result)

            if segments is not None:
                for key in all_segments:
                    all_segments[key].append(segments[key])
            else:
                # No detection — fill with NaN
                for key in all_segments:
                    n_pts = RTMPOSE_SEGMENTS[key][1] - RTMPOSE_SEGMENTS[key][0]
                    all_segments[key].append(np.full((n_pts, 4), np.nan))

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

        cap.release()
        print(f"  Done: {frame_idx} frames processed")

        # Stack into arrays
        output = {}
        for key in all_segments:
            output[key] = np.array(all_segments[key])

        output["fps"] = fps
        output["image_size"] = (height, width)
        return output

    def convert_to_freemocap_format(self, results):
        """Convert rtmlib results to FreeMoCap-compatible 553-point format.

        Maps RTMPose 133 keypoints to FreeMoCap's (numFrames, 553, 3) layout:
        [0:33]   = body (mapped from COCO 17 → MediaPipe 33)
        [33:54]  = right hand (21 keypoints)
        [54:75]  = left hand (21 keypoints)
        [75:143] = face (68 keypoints, padded to 478)

        NOTE: RTMPose outputs pixel coordinates, not world coordinates.
        The z values are camera-space depth, not world-space depth.
        For FreeMoCap pipeline compatibility, we output as-is and note this.

        Args:
            results: dict from process_video()

        Returns:
            freemocap_data: (numFrames, 553, 3) numpy array
        """
        body = results.get("body")
        left_hand = results.get("left_hand")
        right_hand = results.get("right_hand")
        face = results.get("face")

        if body is None or body.shape[0] == 0:
            return np.full((1, 553, 3), np.nan)

        num_frames = body.shape[0]
        output = np.full((num_frames, 553, 3), np.nan)

        # COCO 17 body -> MediaPipe 33 body mapping
        # body shape: (numFrames, 17, 4) — x, y, z, score
        if body is not None:
            for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
                if coco_idx < body.shape[1]:
                    pts = body[:, coco_idx, :3]  # (numFrames, 3) — x, y, z
                    output[:, mp_idx, :3] = pts

        # Right hand: RTMPose [112:133] → FreeMoCap [33:54]
        # shape: (numFrames, 21, 4)
        if right_hand is not None and right_hand.shape[1] >= 21:
            output[:, 33:54, :3] = right_hand[:, :21, :3]

        # Left hand: RTMPose [91:112] → FreeMoCap [54:75]
        # shape: (numFrames, 21, 4)
        if left_hand is not None and left_hand.shape[1] >= 21:
            output[:, 54:75, :3] = left_hand[:, :21, :3]

        # Face: RTMPose [23:91] (68 points) → FreeMoCap [75:553] (478 points)
        # Only 68 of 478 face points are available
        if face is not None:
            face_slots = min(face.shape[1], 478)
            output[:, 75:75 + face_slots, :3] = face[:, :face_slots, :3]

        return output

    def process_and_convert(self, video_path):
        """Convenience: process video and convert to FreeMoCap format in one call.

        Args:
            video_path: path to input video

        Returns:
            freemocap_data: (numFrames, 553, 3) numpy array
            metadata: dict with fps, image_size, etc.
        """
        results = self.process_video(video_path)
        freemocap_data = self.convert_to_freemocap_format(results)

        metadata = {
            "fps": results.get("fps"),
            "image_size": results.get("image_size"),
            "num_frames": freemocap_data.shape[0],
            "num_tracked_points": 553,
            "source": "rtmpose_wholebody133",
            "coordinate_space": "pixel_camera",
        }

        return freemocap_data, metadata
