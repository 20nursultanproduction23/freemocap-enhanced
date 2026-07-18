"""
Multi-Person Detector — Stage 1 of multi-actor pipeline

Detects multiple persons per frame and estimates whole-body pose for each.

Architecture:
  1. Person detection: YOLOX or RTMDet (bounding boxes per frame)
  2. Per-person pose: RTMPose Wholebody (133 keypoints per bbox)

Output format:
  List[PersonDetection] per frame, where each PersonDetection contains:
    - person_id: int (detection index within frame, 0-indexed)
    - bbox: (4,) array [x1, y1, x2, y2]
    - keypoints_133: (133, 3) array [x, y, score]
    - confidence: float (mean body keypoint score)
    - segments: dict with body/feet/face/left_hand/right_hand splits

This module does NOT associate persons across cameras (that's Stage 2).
It operates on a single camera view independently.

Usage:
    detector = MultiPersonDetector(device='cpu')
    frame_detections = detector.detect_frame(frame_image)
    video_detections = detector.detect_video(video_path, frame_range=(0, 50))
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import os


RTMPOSE_SEGMENTS = {
    "body": (0, 17),
    "feet": (17, 23),
    "face": (23, 91),
    "left_hand": (91, 112),
    "right_hand": (112, 133),
}

NUM_KEYPOINTS = 133


@dataclass
class PersonDetection:
    """Single person detection in one frame from one camera."""
    person_id: int
    bbox: np.ndarray
    keypoints_133: np.ndarray
    confidence: float
    segments: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence,
            "num_keypoints": self.keypoints_133.shape[0],
        }


class MultiPersonDetector:
    """Detect and estimate pose for multiple persons per frame.

    Uses rtmlib's YOLOX for person detection and RTMPose for per-bbox
    whole-body pose estimation (133 keypoints).

    Args:
        detector_model: 'yolox' or 'rtmdet'
        pose_model: 'rtmpose-s', 'rtmpose-m', or 'rtmpose-l'
        device: 'cpu' or 'cuda'
        backend: 'onnxruntime', 'opencv', or 'openvino'
        mode: rtmlib mode — 'balanced', 'performance', or 'lightweight'
        det_conf_threshold: minimum detection confidence (0-1)
        pose_conf_threshold: minimum keypoint confidence (0-1)
    """

    def __init__(
        self,
        device: str = "cpu",
        backend: str = "onnxruntime",
        mode: str = "balanced",
        det_conf_threshold: float = 0.3,
        pose_conf_threshold: float = 0.3,
    ):
        self.device = device
        self.backend = backend
        self.mode = mode
        self.det_conf_threshold = det_conf_threshold
        self.pose_conf_threshold = pose_conf_threshold

        self._wholebody = None

    def _init_models(self):
        """Lazy-initialize rtmlib Wholebody model.

        Uses rtmlib's Wholebody class which internally composes YOLOX
        (person detection) + RTMPose (per-bbox whole-body pose estimation).
        Model URLs and input sizes are managed by rtmlib automatically.
        """
        if self._wholebody is not None:
            return

        from rtmlib import Wholebody

        print(f"Initializing Wholebody ({self.mode}, {self.backend}, {self.device})...")
        self._wholebody = Wholebody(
            mode=self.mode,
            backend=self.backend,
            device=self.device,
        )
        print("  Model initialized successfully")

    def _parse_multi_person_results(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bboxes: np.ndarray,
    ) -> List[PersonDetection]:
        """Parse rtmlib RTMPose output into per-person PersonDetection objects.

        Args:
            keypoints: (num_persons, num_keypoints, 2) — x, y pixel coords
            scores: (num_persons, num_keypoints) — confidence per keypoint
            bboxes: (num_persons, 4) — [x1, y1, x2, y2]

        Returns:
            List of PersonDetection, one per detected person
        """
        if keypoints is None or scores is None or bboxes is None:
            return []

        num_persons = keypoints.shape[0]
        detections = []

        for person_idx in range(num_persons):
            kpts = keypoints[person_idx]   # (133, 2)
            scrs = scores[person_idx]      # (133,)
            bbox = bboxes[person_idx]       # (4,)

            # Compute body confidence (first 17 keypoints = body)
            body_scores = scrs[:17]
            body_confidence = float(np.mean(body_scores))

            # Skip persons with very low body confidence
            if body_confidence < self.pose_conf_threshold:
                continue

            # Build (133, 3) array: [x, y, score]
            kpts_with_score = np.column_stack([kpts, scrs])

            # Split into segments
            segments = {}
            for name, (start, end) in RTMPOSE_SEGMENTS.items():
                segments[name] = kpts_with_score[start:end].copy()

            det = PersonDetection(
                person_id=person_idx,
                bbox=np.array(bbox, dtype=np.float64),
                keypoints_133=kpts_with_score,
                confidence=body_confidence,
                segments=segments,
            )
            detections.append(det)

        return detections

    def detect_frame(self, frame: np.ndarray) -> List[PersonDetection]:
        """Detect all persons in a single frame.

        Uses rtmlib's Wholebody which runs YOLOX detection then
        RTMPose per-bbox. Returns ALL detected persons.

        Args:
            frame: (H, W, 3) BGR image (uint8)

        Returns:
            List of PersonDetection, sorted by confidence (highest first)
        """
        self._init_models()

        # Wholebody.__call__ runs: YOLOX(frame) -> bboxes -> RTMPose(frame, bboxes)
        # Returns: (keypoints, scores, keypoints_simcc, keypoints_2d)
        # where keypoints: (num_persons, 133, 2), scores: (num_persons, 133)
        result = self._wholebody(frame)

        if result is None:
            return []

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            kpts = result[0]   # (num_persons, 133, 2)
            scrs = result[1]   # (num_persons, 133)
        else:
            return []

        if kpts is None or scrs is None:
            return []

        # Extract bboxes from YOLOX for the PersonDetection records
        bboxes = self._wholebody.det_model(frame)
        if bboxes is None or len(bboxes) == 0:
            return []

        # Ensure bboxes numpy
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float64)
        else:
            bboxes = bboxes.astype(np.float64)

        # Handle case where detector found more persons than pose estimator processed
        num_kpts_persons = kpts.shape[0]
        num_det_persons = bboxes.shape[0]
        if num_det_persons > num_kpts_persons:
            bboxes = bboxes[:num_kpts_persons]

        detections = self._parse_multi_person_results(kpts, scrs, bboxes)

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections

    def detect_video(
        self,
        video_path: str,
        frame_range: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ) -> List[List[PersonDetection]]:
        """Detect all persons in all frames of a video.

        Args:
            video_path: path to input video file
            frame_range: optional (start, end) inclusive tuple to process subset
            max_frames: optional maximum number of frames to process

        Returns:
            List of lists: outer list indexed by frame, inner list = PersonDetections
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = 0
        end_frame = total_frames
        if frame_range is not None:
            start_frame = frame_range[0]
            end_frame = min(frame_range[1] + 1, total_frames)

        if max_frames is not None:
            end_frame = min(start_frame + max_frames, end_frame)

        print(f"Detecting persons in: {video_path}")
        print(f"  Frames: {start_frame}-{end_frame-1} of {total_frames}, "
              f"FPS: {fps}, Size: {width}x{height}")

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        all_detections = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_frame(frame)
            all_detections.append(detections)

            if (frame_idx - start_frame) % 50 == 0:
                num_persons = len(detections)
                print(f"  Frame {frame_idx}: {num_persons} person(s) detected")

            frame_idx += 1

        cap.release()
        print(f"  Done: {len(all_detections)} frames processed")

        return all_detections

    def detect_from_arrays(
        self,
        frames: np.ndarray,
    ) -> List[List[PersonDetection]]:
        """Detect persons from a numpy array of frames.

        Args:
            frames: (num_frames, H, W, 3) uint8 BGR array

        Returns:
            List of lists: outer list indexed by frame
        """
        self._init_models()

        all_detections = []
        num_frames = frames.shape[0]

        for i in range(num_frames):
            detections = self.detect_frame(frames[i])
            all_detections.append(detections)

            if i % 50 == 0:
                print(f"  Frame {i}/{num_frames}: {len(detections)} person(s)")

        return all_detections


def aggregate_detections(
    all_detections: List[List[PersonDetection]],
    min_confidence: float = 0.0,
) -> dict:
    """Summarize detection results across all frames.

    Args:
        all_detections: output from detect_video() or detect_from_arrays()
        min_confidence: minimum confidence to count a detection

    Returns:
        dict with statistics:
            - total_frames: int
            - frames_with_detections: int
            - detection_rate: float
            - persons_per_frame: list of int
            - mean_persons: float
            - min_persons: int
            - max_persons: int
            - confidence_stats: dict with mean/min/max
    """
    total_frames = len(all_detections)
    persons_per_frame = []
    all_confidences = []

    for frame_dets in all_detections:
        filtered = [d for d in frame_dets if d.confidence >= min_confidence]
        persons_per_frame.append(len(filtered))
        all_confidences.extend([d.confidence for d in filtered])

    frames_with = sum(1 for n in persons_per_frame if n > 0)

    stats = {
        "total_frames": total_frames,
        "frames_with_detections": frames_with,
        "detection_rate": frames_with / max(total_frames, 1),
        "persons_per_frame": persons_per_frame,
        "mean_persons": float(np.mean(persons_per_frame)) if persons_per_frame else 0.0,
        "min_persons": min(persons_per_frame) if persons_per_frame else 0,
        "max_persons": max(persons_per_frame) if persons_per_frame else 0,
    }

    if all_confidences:
        stats["confidence_stats"] = {
            "mean": float(np.mean(all_confidences)),
            "min": float(np.min(all_confidences)),
            "max": float(np.max(all_confidences)),
        }
    else:
        stats["confidence_stats"] = {"mean": 0.0, "min": 0.0, "max": 0.0}

    return stats
