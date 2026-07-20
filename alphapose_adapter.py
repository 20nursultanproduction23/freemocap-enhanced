"""
AlphaPose Adapter — drop-in alternative to RTMPose for crowded/contact scenes.

AlphaPose uses top-down approach (detect persons → crop → pose estimate),
which handles occlusion and physical contact better than RTMPose's bottom-up.

Requires: pip install AlphaPose (or alphapose from source)
Falls back to rtmlib RTMPose if AlphaPose is not installed.

Usage:
    detector = AlphaPoseAdapter(device='cuda')
    results = detector.detect_frame(frame_bgr)
    # Returns same format as MultiPersonDetector: List[PersonDetection]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PoseResult:
    """Single person pose result from any detector."""
    person_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: np.ndarray
    confidence: float
    segment: str = "body"
    scores: Optional[np.ndarray] = None


ALPHAMPOSE_133_BODY_MAP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_big_toe": 17, "left_small_toe": 18, "left_heel": 19,
    "right_big_toe": 20, "right_small_toe": 21, "right_heel": 22,
    "chest": 23, "neck": 24,
    "left_palm": 25, "right_palm": 26,
    "left_middle_finger": 27, "right_middle_finger": 28,
}

COCO_17_TO_133 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
    10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
    15: 15, 16: 16,
}


class AlphaPoseAdapter:
    """
    Adapter that wraps AlphaPose to provide the same interface as MultiPersonDetector.

    If AlphaPose is not installed, falls back to rtmlib RTMPose (top-down mode).
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5,
                 max_persons: int = 10, input_size: Tuple[int, int] = (640, 640)):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_persons = max_persons
        self.input_size = input_size
        self._model = None
        self._detector = None
        self._backend = None
        self._initialized = False

    def _init_alphapose(self):
        """Initialize AlphaPose model."""
        if self._initialized:
            return

        try:
            from alphapose.alphapose import AlphaPose as APModel
            self._model = APModel(device=self.device)
            self._backend = "alphapose"
            self._initialized = True
            logger.info("AlphaPose backend initialized successfully")
            return
        except ImportError:
            logger.info("AlphaPose not installed, trying RTMPose top-down fallback")
        except Exception as e:
            logger.warning(f"AlphaPose init failed: {e}, trying fallback")

        try:
            from rtmlib import Wholebody
            self._model = Wholebody(
                det_backend="ultralytics",
                pose_backend="onnx",
                device=self.device,
            )
            self._backend = "rtmpose_topdown"
            self._initialized = True
            logger.info("RTMPose top-down fallback initialized")
            return
        except Exception as e:
            logger.warning(f"RTMPose fallback also failed: {e}")

        self._backend = "none"
        self._initialized = True
        logger.error("No pose estimation backend available")

    def detect_frame(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Detect all persons and their poses in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List[PoseResult] — one per detected person
        """
        self._init_alphapose()

        if self._backend == "none":
            return []

        if self._backend == "alphapose":
            return self._detect_alphapose(frame)
        else:
            return self._detect_rtmpose_topdown(frame)

    def _detect_alphapose(self, frame: np.ndarray) -> List[PoseResult]:
        """Run AlphaPose detection."""
        try:
            results = self._model(frame)
            poses = []

            if results is None:
                return []

            for i, result in enumerate(results):
                if hasattr(result, 'keypoints') and hasattr(result, 'scores'):
                    kpts = result.keypoints
                    scores = result.scores
                    bbox = getattr(result, 'bbox', [0, 0, frame.shape[1], frame.shape[0]])

                    mean_conf = float(np.mean(scores[scores > 0])) if np.any(scores > 0) else 0.0
                    if mean_conf < self.confidence_threshold:
                        continue

                    kpts_133 = self._map_to_133(kpts, scores)

                    poses.append(PoseResult(
                        person_id=i,
                        bbox=tuple(int(x) for x in bbox[:4]),
                        keypoints=kpts_133["keypoints"],
                        confidence=mean_conf,
                        scores=kpts_133["scores"],
                    ))

                    if len(poses) >= self.max_persons:
                        break

            return poses

        except Exception as e:
            logger.error(f"AlphaPose detection failed: {e}")
            return []

    def _detect_rtmpose_topdown(self, frame: np.ndarray) -> List[PoseResult]:
        """Run RTMPose in top-down mode (similar to AlphaPose approach)."""
        try:
            wholebody_results = self._model(frame)
            if wholebody_results is None:
                return []

            poses = []
            keypoints_all = wholebody_results.get("keypoints", np.array([]))
            scores_all = wholebody_results.get("scores", np.array([]))
            bboxes = wholebody_results.get("bboxes", np.array([]))

            if len(keypoints_all) == 0:
                return []

            for i in range(len(keypoints_all)):
                kpts = keypoints_all[i]
                scrs = scores_all[i] if i < len(scores_all) else np.ones(len(kpts))

                mean_conf = float(np.mean(scrs[scrs > 0])) if np.any(scrs > 0) else 0.0
                if mean_conf < self.confidence_threshold:
                    continue

                if len(bboxes) > i:
                    bbox = bboxes[i]
                else:
                    valid = kpts[scrs > 0.3]
                    if len(valid) == 0:
                        continue
                    x_min, y_min = valid.min(axis=0)
                    x_max, y_max = valid.max(axis=0)
                    pad = 20
                    bbox = [max(0, x_min - pad), max(0, y_min - pad),
                            min(frame.shape[1], x_max + pad), min(frame.shape[0], y_max + pad)]

                kpts_133 = self._map_to_133(kpts, scrs)

                poses.append(PoseResult(
                    person_id=i,
                    bbox=tuple(int(x) for x in bbox[:4]),
                    keypoints=kpts_133["keypoints"],
                    confidence=mean_conf,
                    scores=kpts_133["scores"],
                ))

                if len(poses) >= self.max_persons:
                    break

            return poses

        except Exception as e:
            logger.error(f"RTMPose top-down detection failed: {e}")
            return []

    def _map_to_133(self, keypoints: np.ndarray, scores: np.ndarray) -> dict:
        """Map detector output to 133-keypoint format."""
        kpts_133 = np.full((133, 3), np.nan, dtype=np.float32)
        scores_133 = np.zeros(133, dtype=np.float32)

        kpts = np.array(keypoints)
        scrs = np.array(scores)

        if kpts.ndim == 1:
            kpts = kpts.reshape(-1, 2) if len(kpts) >= 2 else kpts.reshape(-1, 3)

        n_detected = min(len(kpts), len(scrs))
        n_mapped = min(n_detected, 17)

        for coco_idx in range(n_mapped):
            if coco_idx in COCO_17_TO_133:
                target_idx = COCO_17_TO_133[coco_idx]
                if coco_idx < len(kpts):
                    kp = kpts[coco_idx]
                    if len(kp) >= 2:
                        kpts_133[target_idx, 0] = kp[0]
                        kpts_133[target_idx, 1] = kp[1]
                        kpts_133[target_idx, 2] = 0.0
                    scores_133[target_idx] = float(scrs[coco_idx])

        if n_detected > 17:
            extra = kpts[17:]
            extra_s = scrs[17:]
            for j in range(min(len(extra), 133 - 25)):
                target = 25 + j
                if target < 133 and j < len(extra):
                    kp = extra[j]
                    if len(kp) >= 2:
                        kpts_133[target, 0] = kp[0]
                        kpts_133[target, 1] = kp[1]
                        kpts_133[target, 2] = 0.0
                    if j < len(extra_s):
                        scores_133[target] = float(extra_s[j])

        return {"keypoints": kpts_133, "scores": scores_133}

    def detect_video(self, video_path: str, frame_range: Optional[Tuple[int, int]] = None) -> List[List[PoseResult]]:
        """Detect poses across all frames of a video file."""
        if not HAS_CV2:
            raise RuntimeError("opencv-python required for video detection")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = frame_range[0] if frame_range else 0
        end_frame = frame_range[1] if frame_range else total_frames

        all_results = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < start_frame:
                continue
            if frame_idx > end_frame:
                break

            results = self.detect_frame(frame)
            all_results.append(results)

        cap.release()
        return all_results

    def detect_from_arrays(self, frames: np.ndarray) -> List[List[PoseResult]]:
        """Detect poses from a numpy array of frames (N, H, W, 3)."""
        all_results = []
        for i in range(len(frames)):
            results = self.detect_frame(frames[i])
            all_results.append(results)
        return all_results

    def get_backend_info(self) -> dict:
        """Return information about which backend is being used."""
        self._init_alphapose()
        return {
            "backend": self._backend,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "initialized": self._initialized,
        }


def compare_detectors(frame: np.ndarray, alphapose_adapter: AlphaPoseAdapter,
                      rtm_detector=None) -> dict:
    """
    Compare AlphaPose and RTMPose results on the same frame.

    Useful for deciding which detector to use for a given scene.
    """
    ap_results = alphapose_adapter.detect_frame(frame)

    rt_results = []
    if rtm_detector is not None:
        try:
            rt_results = rtm_detector.detect_frame(frame)
        except Exception:
            pass

    comparison = {
        "alphapose_count": len(ap_results),
        "rtmpose_count": len(rt_results),
        "alphapose_confidences": [r.confidence for r in ap_results],
        "rtmpose_confidences": [r.confidence for r in rt_results],
        "alphapose_mean_confidence": float(np.mean([r.confidence for r in ap_results])) if ap_results else 0.0,
        "rtmpose_mean_confidence": float(np.mean([r.confidence for r in rt_results])) if rt_results else 0.0,
    }

    if ap_results and rt_results:
        ap_boxes = np.array([r.bbox for r in ap_results])
        rt_boxes = np.array([r.bbox for r in rt_results])
        comparison["bbox_iou_max"] = float(_max_bbox_iou(ap_boxes, rt_boxes))

    return comparison


def _max_bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> float:
    """Compute maximum IoU between two sets of bounding boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return 0.0

    max_iou = 0.0
    for a in boxes_a:
        for b in boxes_b:
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            union = area_a + area_b - inter

            iou = inter / union if union > 0 else 0.0
            max_iou = max(max_iou, iou)

    return max_iou


def suggest_detector(n_persons: int, physical_contact: bool = False,
                     scene_description: str = "") -> dict:
    """
    Suggest which detector to use based on scene characteristics.

    Returns recommendation with reasoning.
    """
    score_alphapose = 0.0
    score_rtmpose = 0.0
    reasons = []

    if n_persons >= 3:
        score_alphapose += 2.0
        reasons.append("3+ persons: AlphaPose handles crowds better")
    elif n_persons == 2:
        score_alphapose += 0.5
        reasons.append("2 persons: slight AlphaPose advantage")

    if physical_contact:
        score_alphapose += 1.5
        reasons.append("Physical contact: AlphaPose top-down approach handles occlusion")
    else:
        score_rtmpose += 0.5
        reasons.append("No contact: RTMPose is faster and sufficient")

    if "dark" in scene_description.lower() or "low light" in scene_description.lower():
        score_rtmpose += 0.5
        reasons.append("Low light: RTMPose has better low-light training")

    if "fast" in scene_description.lower() or "dance" in scene_description.lower():
        score_alphapose += 0.5
        reasons.append("Fast motion: AlphaPose better at catching partial poses")

    recommended = "alphapose" if score_alphapose > score_rtmpose else "rtmpose"

    return {
        "recommended": recommended,
        "alphapose_score": score_alphapose,
        "rtmpose_score": score_rtmpose,
        "reasons": reasons,
    }
