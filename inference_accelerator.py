"""
Inference Acceleration Wrapper

Provides batch processing and ONNX Runtime optimization for RTMDet/RTMPose
models. Frames from multiple cameras are collected into batches for a single
inference call instead of N sequential calls. ONNX Runtime sessions are
configured with graph optimization and appropriate execution providers.

All functions are optional wrappers — existing sequential inference code
continues to work without modification.
"""
import numpy as np
from typing import List, Optional, Any, Dict, Tuple
import time
import warnings

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False


class InferenceAccelerator:
    """Batch inference and ONNX session optimization for pose estimation.

    Wraps onnxruntime with convenience methods for multi-camera batch
    processing, frame skipping with interpolation, and session tuning.

    Args:
        default_providers: list of execution provider strings to try.
            Default tries CUDA first, falls back to CPU.
        default_threads: number of intra-op threads for CPU execution.
            None lets onnxruntime decide.
    """

    DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(
        self,
        default_providers: Optional[List[str]] = None,
        default_threads: Optional[int] = None,
    ):
        if not _HAS_ORT:
            warnings.warn(
                "onnxruntime not available. Session optimization and "
                "batch inference will raise errors."
            )

        self.default_providers = default_providers or self.DEFAULT_PROVIDERS
        self.default_threads = default_threads
        self._active_sessions: Dict[str, Any] = {}

    def optimize_session(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
        thread_count: Optional[int] = None,
        enable_graph_optimization: bool = True,
        optimization_level: str = "all",
        session_name: Optional[str] = None,
    ) -> Any:
        """Create an optimized ONNX Runtime inference session.

        Applies graph optimization, sets thread count, and selects the
        best available execution provider.

        Args:
            onnx_path: path to .onnx model file.
            providers: execution providers override. Falls back through
                the list until one is available.
            thread_count: intra-op thread count. None for default.
            enable_graph_optimization: apply ORT graph optimizations.
            optimization_level: 'basic', 'extended', or 'all'.
            session_name: optional key for caching in _active_sessions.

        Returns:
            onnxruntime.InferenceSession

        Raises:
            ImportError: if onnxruntime is not installed.
            RuntimeError: if no providers are available.
        """
        if not _HAS_ORT:
            raise ImportError(
                "onnxruntime is required for optimize_session(). "
                "Install with: pip install onnxruntime"
            )

        providers = providers or self.default_providers
        threads = thread_count or self.default_threads

        available = ort.get_available_providers()
        selected = [p for p in providers if p in available]
        if not selected:
            raise RuntimeError(
                f"None of the requested providers {providers} are available. "
                f"Available: {available}"
            )

        opts = ort.SessionOptions()

        if enable_graph_optimization:
            level_map = {
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            opts.graph_optimization_level = level_map.get(
                optimization_level,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            )
        else:
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

        if threads is not None:
            opts.intra_op_num_threads = max(1, int(threads))
            opts.inter_op_num_threads = max(1, int(threads))

        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(onnx_path, opts, providers=selected)

        if session_name:
            self._active_sessions[session_name] = session

        return session

    def batch_infer(
        self,
        session: Any,
        frames_list: List[np.ndarray],
        input_name: Optional[str] = None,
        auto_transpose: bool = True,
    ) -> List[Any]:
        """Run inference on a batch of frames in a single call.

        Stacks frames into a (N, C, H, W) or (N, H, W, C) batch tensor
        and runs one session.run() call instead of N separate calls.

        Args:
            session: onnxruntime.InferenceSession.
            frames_list: list of numpy arrays, each (H, W, C) uint8 or
                (C, H, W) float. All frames must have the same shape.
            input_name: model input tensor name. Auto-detected if None.
            auto_transpose: if True, converts (H, W, C) → (C, H, W) and
                normalizes uint8 to float32 [0, 1].

        Returns:
            List of output arrays, one per frame in the batch. Each element
            has the same structure as a single session.run() output.
        """
        if not frames_list:
            return []

        if input_name is None:
            input_name = session.get_inputs()[0].name

        batch = []
        for f in frames_list:
            arr = np.asarray(f)
            if auto_transpose and arr.ndim == 3:
                if arr.shape[2] in (1, 3, 4):
                    arr = arr.transpose(2, 0, 1)
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
            batch.append(arr)

        batch_tensor = np.stack(batch, axis=0)

        outputs = session.run(None, {input_name: batch_tensor})

        results = []
        for i in range(len(frames_list)):
            frame_results = []
            for out in outputs:
                if out.ndim > 0 and out.shape[0] == len(frames_list):
                    frame_results.append(out[i])
                else:
                    frame_results.append(out)
            results.append(frame_results if len(frame_results) > 1 else frame_results[0])

        return results

    def skip_interpolate(
        self,
        detections: List[Optional[Any]],
        skip_factor: int = 3,
        method: str = "linear",
    ) -> List[Any]:
        """Interpolate detections for skipped frames.

        When frame skipping is active, only every Nth frame is detected.
        This method fills in intermediate frames by linear (or nearest)
        interpolation of the detection results.

        Args:
            detections: list indexed by frame number. Each element is either
                a detection result (array, dict, etc.) or None for skipped
                frames. Must have detections[0] and detections[-1] populated.
            skip_factor: only every skip_factor-th frame has real detections.
            method: 'linear' for linear interpolation, 'nearest' to copy
                the closest detected frame.

        Returns:
            List of same length with no None entries. Detections that were
            already present are returned unchanged; None slots are filled.
        """
        n = len(detections)
        if n == 0:
            return []

        result = list(detections)

        detected_indices = [i for i, d in enumerate(detections) if d is not None]

        if len(detected_indices) < 2:
            for i in range(n):
                if result[i] is None and detected_indices:
                    result[i] = detections[detected_indices[0]]
            return result

        for i in range(n):
            if result[i] is not None:
                continue

            prev_idx = max((idx for idx in detected_indices if idx < i), default=None)
            next_idx = min((idx for idx in detected_indices if idx > i), default=None)

            if prev_idx is None and next_idx is None:
                continue

            if prev_idx is None:
                result[i] = detections[next_idx]
                continue

            if next_idx is None:
                result[i] = detections[prev_idx]
                continue

            if method == "nearest":
                dist_prev = i - prev_idx
                dist_next = next_idx - i
                result[i] = detections[prev_idx if dist_prev <= dist_next else next_idx]
                continue

            alpha = (i - prev_idx) / (next_idx - prev_idx)
            result[i] = self._interpolate_detection(
                detections[prev_idx], detections[next_idx], alpha
            )

        return result

    @staticmethod
    def _interpolate_detection(
        det_a: Any, det_b: Any, alpha: float
    ) -> Any:
        """Interpolate between two detection results.

        Supports numpy arrays (linear blend), dicts with numeric values,
        and tuples/lists of numpy arrays.
        """
        if isinstance(det_a, np.ndarray) and isinstance(det_b, np.ndarray):
            if det_a.shape == det_b.shape:
                return (1.0 - alpha) * det_a + alpha * det_b

        if isinstance(det_a, dict) and isinstance(det_b, dict):
            merged = {}
            for key in set(list(det_a.keys()) + list(det_b.keys())):
                va = det_a.get(key)
                vb = det_b.get(key)
                if va is None:
                    merged[key] = vb
                elif vb is None:
                    merged[key] = va
                elif isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                    if va.shape == vb.shape:
                        merged[key] = (1.0 - alpha) * va + alpha * vb
                    else:
                        merged[key] = va if alpha < 0.5 else vb
                else:
                    merged[key] = va if alpha < 0.5 else vb
            return merged

        if isinstance(det_a, (list, tuple)) and isinstance(det_b, (list, tuple)):
            out = []
            for a, b in zip(det_a, det_b):
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    if a.shape == b.shape:
                        out.append((1.0 - alpha) * a + alpha * b)
                    else:
                        out.append(a if alpha < 0.5 else b)
                else:
                    out.append(a if alpha < 0.5 else b)
            return type(det_a)(out)

        return det_a if alpha < 0.5 else det_b

    def get_session_info(self, session: Any) -> Dict[str, Any]:
        """Return metadata about an ONNX Runtime session.

        Args:
            session: onnxruntime.InferenceSession.

        Returns:
            dict with provider, input/output names and shapes, optimization level.
        """
        info = {
            "providers": session.get_providers(),
            "inputs": [],
            "outputs": [],
        }
        for inp in session.get_inputs():
            info["inputs"].append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })
        for out in session.get_outputs():
            info["outputs"].append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })
        return info

    @property
    def available_providers(self) -> List[str]:
        """List of ONNX Runtime execution providers currently available."""
        if not _HAS_ORT:
            return []
        return ort.get_available_providers()

    @property
    def ort_available(self) -> bool:
        """Whether onnxruntime is importable."""
        return _HAS_ORT


def create_optimized_session(
    onnx_path: str,
    providers: Optional[List[str]] = None,
    thread_count: Optional[int] = None,
) -> Any:
    """Convenience function: create an optimized ONNX Runtime session.

    Args:
        onnx_path: path to .onnx model file.
        providers: execution providers to try. Default: CUDA → CPU.
        thread_count: intra-op threads. None for default.

    Returns:
        onnxruntime.InferenceSession with graph optimization enabled.

    Raises:
        ImportError: if onnxruntime is not installed.
    """
    acc = InferenceAccelerator(default_threads=thread_count)
    return acc.optimize_session(onnx_path, providers=providers)


def batch_detect(
    session: Any,
    frames: List[np.ndarray],
    input_name: Optional[str] = None,
) -> List[Any]:
    """Convenience function: run batch inference on multiple frames.

    Args:
        session: onnxruntime.InferenceSession.
        frames: list of numpy image arrays (H, W, C) or (C, H, W).
        input_name: model input name. Auto-detected if None.

    Returns:
        List of per-frame detection results.
    """
    acc = InferenceAccelerator()
    return acc.batch_infer(session, frames, input_name=input_name)
