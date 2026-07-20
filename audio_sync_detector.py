"""
Audio Sync Detector — synchronizes multi-camera recordings using audio clap/pulse.

Detects a sync clap (or other sharp sound) in audio tracks from multiple cameras
and computes frame-level time offsets between them.

This is the simplest and most reliable synchronization method:
  1. Record audio from each camera
  2. Clap once at the start
  3. This module finds the clap in each audio track
  4. Computes offsets so all cameras are frame-synchronized

Usage:
    detector = AudioSyncDetector(fps=25.0)
    offsets = detector.detect_from_files(audio_paths)
    print(offsets)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncPulse:
    """Detected sync pulse (clap) in an audio track."""
    camera_id: int
    peak_sample: int
    peak_time_s: float
    peak_energy: float
    confidence: float
    snr_db: float


@dataclass
class SyncResult:
    """Synchronization result for all cameras."""
    reference_camera: int
    offsets_frames: dict
    offsets_ms: dict
    offsets_samples: dict
    pulses: List[SyncPulse]
    all_synchronized: bool
    max_offset_ms: float
    quality: str


class AudioSyncDetector:
    """
    Detects sync claps in audio tracks and computes camera time offsets.

    Works by:
    1. Computing short-time energy (STE) of each audio track
    2. Finding energy peaks that exceed a threshold (the clap)
    3. Cross-correlating peaks between cameras
    4. Converting sample offsets to frame offsets
    """

    def __init__(self, fps: float = 25.0, sample_rate: int = 44100,
                 energy_window_ms: float = 10.0,
                 peak_threshold_db: float = 20.0,
                 min_peak_distance_ms: float = 100.0,
                 max_sync_offset_ms: float = 200.0):
        """
        Args:
            fps: Recording frame rate
            sample_rate: Audio sample rate in Hz
            energy_window_ms: Window size for energy computation
            peak_threshold_db: dB threshold above background for clap detection
            min_peak_distance_ms: Minimum distance between consecutive claps
            max_sync_offset_ms: Maximum acceptable sync offset
        """
        self.fps = fps
        self.sample_rate = sample_rate
        self.energy_window_samples = int(sample_rate * energy_window_ms / 1000)
        self.peak_threshold_db = peak_threshold_db
        self.min_peak_distance_samples = int(sample_rate * min_peak_distance_ms / 1000)
        self.max_sync_offset_samples = int(sample_rate * max_sync_offset_ms / 1000)

    def detect_from_arrays(self, audio_arrays: List[np.ndarray]) -> SyncResult:
        """
        Detect sync pulses from audio numpy arrays.

        Args:
            audio_arrays: List of 1D numpy arrays (mono audio per camera)

        Returns:
            SyncResult with per-camera offsets
        """
        if len(audio_arrays) < 2:
            raise ValueError("Need at least 2 audio tracks for synchronization")

        pulses = []
        for cam_id, audio in enumerate(audio_arrays):
            pulse = self._find_sync_pulse(audio, cam_id)
            if pulse is not None:
                pulses.append(pulse)

        return self._compute_offsets(pulses, n_cameras=len(audio_arrays))

    def detect_from_files(self, audio_paths: List[str]) -> SyncResult:
        """
        Detect sync pulses from audio file paths.

        Requires scipy or wave module for reading WAV files.
        """
        import wave

        audio_arrays = []
        for path in audio_paths:
            try:
                with wave.open(path, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)

                    if sample_width == 2:
                        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
                    elif sample_width == 4:
                        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
                    else:
                        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128

                    if n_channels > 1:
                        audio = audio[::n_channels]

                    audio = audio / max(1, np.max(np.abs(audio)))
                    audio_arrays.append(audio)

            except Exception as e:
                logger.error(f"Failed to read audio from {path}: {e}")
                audio_arrays.append(np.zeros(self.sample_rate))

        return self.detect_from_arrays(audio_arrays)

    def detect_from_energy(self, energy_profiles: List[np.ndarray],
                           audio_sample_rates: Optional[List[int]] = None) -> SyncResult:
        """
        Detect sync from pre-computed energy profiles (for integration with FreeMoCap).

        Args:
            energy_profiles: List of 1D energy arrays (one per camera)
            audio_sample_rates: Sample rates for each profile (default: self.sample_rate)
        """
        pulses = []
        for cam_id, energy in enumerate(energy_profiles):
            sr = audio_sample_rates[cam_id] if audio_sample_rates else self.sample_rate
            pulse = self._find_sync_from_energy(energy, cam_id, sr)
            if pulse is not None:
                pulses.append(pulse)

        return self._compute_offsets(pulses, n_cameras=len(energy_profiles))

    def _find_sync_pulse(self, audio: np.ndarray, cam_id: int) -> Optional[SyncPulse]:
        """Find the most prominent sync pulse in an audio track."""
        if len(audio) == 0:
            return None

        energy = self._compute_short_time_energy(audio)

        if len(energy) == 0:
            return None

        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))

        median_energy = float(np.median(energy_db))
        threshold = median_energy + self.peak_threshold_db

        peaks = self._find_peaks(energy_db, threshold, self.min_peak_distance_samples)

        if not peaks:
            threshold = median_energy + self.peak_threshold_db * 0.5
            peaks = self._find_peaks(energy_db, threshold, self.min_peak_distance_samples)

        if not peaks:
            return None

        best_peak_idx = max(peaks, key=lambda i: energy_db[i] if i < len(energy_db) else 0)
        best_peak_idx = min(best_peak_idx, len(energy) - 1)

        peak_energy = float(energy[best_peak_idx])
        peak_energy_db = float(energy_db[best_peak_idx])

        peak_sample = best_peak_idx * self.energy_window_samples
        peak_time = peak_sample / self.sample_rate

        snr = peak_energy_db - median_energy
        confidence = min(1.0, max(0.0, snr / 40.0))

        return SyncPulse(
            camera_id=cam_id,
            peak_sample=peak_sample,
            peak_time_s=round(peak_time, 6),
            peak_energy=round(peak_energy, 6),
            confidence=round(confidence, 4),
            snr_db=round(snr, 1),
        )

    def _find_sync_from_energy(self, energy: np.ndarray, cam_id: int,
                                sample_rate: int) -> Optional[SyncPulse]:
        """Find sync pulse from pre-computed energy."""
        if len(energy) == 0:
            return None

        energy_db = 10 * np.log10(np.maximum(energy, 1e-10))
        median_energy = float(np.median(energy_db))
        threshold = median_energy + self.peak_threshold_db

        min_dist = int(sample_rate * 100 / 1000)
        peaks = self._find_peaks(energy_db, threshold, min_dist)

        if not peaks:
            return None

        best_peak_idx = max(peaks, key=lambda i: energy_db[i] if i < len(energy_db) else 0)
        best_peak_idx = min(best_peak_idx, len(energy) - 1)

        peak_energy_db = float(energy_db[best_peak_idx])
        snr = peak_energy_db - median_energy

        return SyncPulse(
            camera_id=cam_id,
            peak_sample=best_peak_idx,
            peak_time_s=round(best_peak_idx / sample_rate, 6),
            peak_energy=round(float(energy[best_peak_idx]), 6),
            confidence=round(min(1.0, max(0.0, snr / 40.0)), 4),
            snr_db=round(snr, 1),
        )

    def _compute_short_time_energy(self, audio: np.ndarray) -> np.ndarray:
        """Compute short-time energy of audio signal."""
        window = self.energy_window_samples
        if window < 1:
            window = 1

        n_windows = max(1, len(audio) // window)
        energy = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * window
            end = min(start + window, len(audio))
            segment = audio[start:end]
            energy[i] = float(np.mean(segment ** 2))

        return energy

    def _find_peaks(self, signal: np.ndarray, threshold: float,
                     min_distance: int) -> List[int]:
        """Find peaks in a signal above threshold with minimum distance."""
        peaks = []

        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)

        return peaks

    def _compute_offsets(self, pulses: List[SyncPulse], n_cameras: int) -> SyncResult:
        """Compute frame offsets from detected pulses."""
        if not pulses:
            return SyncResult(
                reference_camera=0,
                offsets_frames={i: 0 for i in range(n_cameras)},
                offsets_ms={i: 0.0 for i in range(n_cameras)},
                offsets_samples={i: 0 for i in range(n_cameras)},
                pulses=[],
                all_synchronized=False,
                max_offset_ms=0.0,
                quality="no_pulses_found",
            )

        best_pulses = {}
        for pulse in pulses:
            if pulse.camera_id not in best_pulses or pulse.confidence > best_pulses[pulse.camera_id].confidence:
                best_pulses[pulse.camera_id] = pulse

        ref_camera = max(best_pulses.keys(), key=lambda c: best_pulses[c].confidence)
        ref_sample = best_pulses[ref_camera].peak_sample

        offsets_samples = {}
        offsets_frames = {}
        offsets_ms = {}

        for cam_id in range(n_cameras):
            if cam_id in best_pulses:
                offset_s = best_pulses[cam_id].peak_sample - ref_sample
                offsets_samples[cam_id] = offset_s
                offsets_frames[cam_id] = round(offset_s / (self.sample_rate / self.fps))
                offsets_ms[cam_id] = round(offset_s / self.sample_rate * 1000, 2)
            else:
                offsets_samples[cam_id] = 0
                offsets_frames[cam_id] = 0
                offsets_ms[cam_id] = 0.0

        max_offset_ms = max(abs(v) for v in offsets_ms.values()) if offsets_ms else 0.0

        if max_offset_ms < 5.0 / self.fps:
            quality = "excellent"
        elif max_offset_ms < 20.0 / self.fps:
            quality = "good"
        elif max_offset_ms < self.max_sync_offset_samples / self.sample_rate * 1000:
            quality = "acceptable"
        else:
            quality = "poor"

        return SyncResult(
            reference_camera=ref_camera,
            offsets_frames=offsets_frames,
            offsets_ms=offsets_ms,
            offsets_samples=offsets_samples,
            pulses=list(best_pulses.values()),
            all_synchronized=quality in ("excellent", "good"),
            max_offset_ms=round(max_offset_ms, 2),
            quality=quality,
        )

    @staticmethod
    def format_result(result: SyncResult) -> str:
        """Format a human-readable sync result."""
        lines = [
            "=== Audio Sync Result ===",
            f"Reference camera: {result.reference_camera}",
            f"Quality: {result.quality}",
            f"Max offset: {result.max_offset_ms:.1f}ms ({result.max_offset_ms * result.pulses[0].peak_time_s / max(0.001, result.max_offset_ms / 1000):.1f} frames)" if result.pulses else f"Max offset: {result.max_offset_ms:.1f}ms",
            "",
            "Per-camera offsets:",
        ]

        for cam_id in sorted(result.offsets_frames.keys()):
            frames = result.offsets_frames[cam_id]
            ms = result.offsets_ms[cam_id]
            ref = " (reference)" if cam_id == result.reference_camera else ""
            lines.append(f"  Camera {cam_id}: {frames:+d} frames ({ms:+.1f}ms){ref}")

        if not result.all_synchronized:
            lines.append("\nWARNING: Cameras are not well synchronized. Consider re-recording with a sync clap.")

        return "\n".join(lines)

    @staticmethod
    def generate_sync_signal(duration_s: float = 1.0, sample_rate: int = 44100,
                              clap_time_s: float = 0.5) -> np.ndarray:
        """
        Generate a synthetic sync signal (clap) for testing.

        Creates a short burst of white noise simulating a hand clap.
        """
        n_samples = int(duration_s * sample_rate)
        signal = np.random.randn(n_samples) * 0.01

        clap_start = int(clap_time_s * sample_rate)
        clap_duration = int(0.01 * sample_rate)
        clap_end = min(clap_start + clap_duration, n_samples)

        signal[clap_start:clap_end] = np.random.randn(clap_end - clap_start) * 0.8

        return signal.astype(np.float64)
