"""
ECG Analyzer for Polar H10 / JSONL Recordings
Processes raw ECG jsonl data, filters signal, detects R-peaks,
computes comprehensive time & frequency domain HRV metrics,
and analyzes beat morphology.
"""

import json
import os
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.interpolate import interp1d

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _parabolic_peak_offset(y_prev: float, y_curr: float, y_next: float) -> float:
    """
    Sub-sample peak refinement via 3-point parabolic interpolation.
    Returns a fractional-sample offset in [-0.5, 0.5] to add to the integer peak index.
    Reduces R-peak timing quantization from the raw sample grid (e.g. +/-3.8ms at 130Hz)
    down to a fraction of that, which matters directly for RMSSD/pNN50/SD1 - metrics
    defined on millisecond-scale successive RR differences.
    """
    denom = y_prev - 2.0 * y_curr + y_next
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(offset, -0.5, 0.5))


def _dfa_alpha(nn_ms: np.ndarray, box_sizes: np.ndarray) -> Optional[float]:
    """
    Detrended Fluctuation Analysis scaling exponent (Peng et al. 1995) over the given
    box sizes (in beats). Returns None if there isn't enough data for >= 3 box sizes.
    """
    x = np.cumsum(nn_ms - np.mean(nn_ms))
    n = len(x)
    fluctuations = []
    valid_boxes = []
    for bs in box_sizes:
        bs = int(bs)
        n_boxes = n // bs
        if n_boxes < 2:
            continue
        trimmed = x[: n_boxes * bs].reshape(n_boxes, bs)
        idx = np.arange(bs)
        # Fit and remove a linear local trend per box, vectorized across all boxes
        coeffs = np.polynomial.polynomial.polyfit(idx, trimmed.T, 1)
        trend = np.polynomial.polynomial.polyval(idx, coeffs)
        f2 = np.mean((trimmed - trend) ** 2, axis=1)
        fluctuations.append(np.sqrt(np.mean(f2)))
        valid_boxes.append(bs)

    if len(valid_boxes) < 3:
        return None

    log_n = np.log10(valid_boxes)
    log_f = np.log10(np.maximum(fluctuations, 1e-9))
    alpha, _ = np.polyfit(log_n, log_f, 1)
    return float(alpha)


@dataclass
class ECGMetadata:
    device_id: str = "Unknown"
    recording_name: str = "Unknown"
    data_type: str = "ECG"
    start_phone_timestamp_ms: Optional[int] = None
    start_iso_time: Optional[str] = None
    total_samples: int = 0
    duration_seconds: float = 0.0
    sampling_rate_hz: float = 130.0


@dataclass
class BeatAnomaly:
    beat_index: int                # 1-indexed beat number
    sample_index: int             # index in signal array
    time_seconds: float           # timestamp in seconds
    anomaly_type: str             # 'PAC', 'PVC', 'PAUSE', 'TACHYCARDIA', 'BRADYCARDIA'
    rr_pre_ms: float              # RR interval before beat
    rr_post_ms: float             # RR interval after beat
    pre_ratio: float              # rr_pre / local_median_rr (< 0.85 = premature)
    post_ratio: float             # rr_post / local_median_rr (> 1.15 = compensatory pause)
    qrs_correlation: float        # correlation with normal sinus template (1.0 = identical)
    description: str              # Clinical description / details
    pattern: Optional[str] = None # 'Couplet', 'Bigeminy', 'Trigeminy', None


@dataclass
class WindowMetrics:
    """Short-term HRV metrics for one non-overlapping time window (for 24h trending)."""
    start_s: float
    end_s: float
    n_beats: int
    mean_hr_bpm: float = 0.0
    sdnn_ms: float = 0.0
    rmssd_ms: float = 0.0
    pnn50_pct: float = 0.0
    lf_power_ms2: float = 0.0
    hf_power_ms2: float = 0.0
    lf_hf_ratio: float = 0.0
    resp_rate_bpm: float = 0.0
    pac_count: int = 0
    pvc_count: int = 0
    quality_pct: float = 100.0


@dataclass
class HRVMetrics:
    # Time-domain
    total_beats: int = 0
    mean_hr_bpm: float = 0.0
    std_hr_bpm: float = 0.0
    min_hr_bpm: float = 0.0
    max_hr_bpm: float = 0.0
    mean_rr_ms: float = 0.0
    std_rr_ms: float = 0.0
    median_rr_ms: float = 0.0
    sdnn_ms: float = 0.0
    rmssd_ms: float = 0.0
    sdsd_ms: float = 0.0
    pnn50_pct: float = 0.0
    pnn20_pct: float = 0.0
    
    # Frequency-domain (Lomb-Scargle PSD, variance-calibrated - handles uneven RR sampling
    # natively, no resampling/interpolation artifacts)
    ulf_power_ms2: float = 0.0  # < 0.0033 Hz (only computed for recordings > 2h)
    vlf_power_ms2: float = 0.0  # 0.0033 - 0.04 Hz
    lf_power_ms2: float = 0.0   # 0.04 - 0.15 Hz
    hf_power_ms2: float = 0.0   # 0.15 - 0.40 Hz
    total_power_ms2: float = 0.0
    lf_hf_ratio: float = 0.0
    lf_nu: float = 0.0          # Normalized LF: LF / (Total - VLF) * 100
    hf_nu: float = 0.0          # Normalized HF: HF / (Total - VLF) * 100

    # Non-linear (Poincaré)
    sd1_ms: float = 0.0         # Short-term variability
    sd2_ms: float = 0.0         # Long-term variability
    sd1_sd2_ratio: float = 0.0

    # Non-linear (Detrended Fluctuation Analysis)
    dfa_alpha1: float = 0.0     # Short-term scaling exponent (4-16 beat boxes)
    dfa_alpha2: float = 0.0     # Long-term scaling exponent (16-64 beat boxes, needs longer recordings)

    # ECG-derived respiration (R-wave amplitude modulation)
    resp_rate_bpm: float = 0.0

    # Heart Rate Turbulence (approximate, Schmidt et al. 1999 convention) - computed
    # around detected PVCs when enough qualifying events exist
    hrt_turbulence_onset_pct: float = 0.0
    hrt_turbulence_slope_ms: float = 0.0
    hrt_qualifying_events: int = 0

    # Arrhythmia & Ectopy counts
    pvc_count: int = 0
    pac_count: int = 0
    pause_count: int = 0
    tachycardia_episodes: int = 0
    bradycardia_episodes: int = 0
    ectopic_burden_pct: float = 0.0

    # Signal Quality
    quality_score_pct: float = 100.0


class ECGAnalyzer:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.metadata = ECGMetadata()
        self.raw_voltages: np.ndarray = np.array([], dtype=float)
        self.timestamps_ns: np.ndarray = np.array([], dtype=np.int64)
        self.time_seconds: np.ndarray = np.array([], dtype=float)
        self.filtered_voltages: np.ndarray = np.array([], dtype=float)
        
        # Detection results
        self.r_peaks_idx: np.ndarray = np.array([], dtype=int)
        self.r_peak_times: np.ndarray = np.array([], dtype=float)
        self.rr_intervals_ms: np.ndarray = np.array([], dtype=float)
        self.instant_hr_bpm: np.ndarray = np.array([], dtype=float)
        self.valid_beat_mask: np.ndarray = np.array([], dtype=bool)
        
        # Metrics
        self.hrv = HRVMetrics()
        
        # Beat morphology
        self.average_beat_time_ms: np.ndarray = np.array([])
        self.average_beat_mv: np.ndarray = np.array([])
        self.average_beat_std_mv: np.ndarray = np.array([])
        self.all_beat_waves: List[np.ndarray] = []
        
        # Arrhythmia & Ectopic beats
        self.anomalies: List[BeatAnomaly] = []
        self.beat_labels: List[str] = []
        
        # PSD data
        self.psd_freqs: np.ndarray = np.array([])
        self.psd_values: np.ndarray = np.array([])

        # ECG-derived respiration
        self.resp_signal_time: np.ndarray = np.array([])
        self.resp_signal_amp: np.ndarray = np.array([])

        # Windowed (short-term segment) HRV trend, for long/24h recordings
        self.windowed_hrv: List["WindowMetrics"] = []

        if filepath:
            self.load_jsonl(filepath)

    def load_jsonl(self, filepaths: Any) -> "ECGAnalyzer":
        """
        Load and parse Polar H10 ECG jsonl format.
        Supports single filepath, list/tuple of split filepaths, or glob pattern.
        Seamlessly stitches and sorts multi-file split logs without gaps.
        """
        if isinstance(filepaths, str):
            if any(char in filepaths for char in ["*", "?", "["]):
                import glob
                files = sorted(glob.glob(filepaths))
            else:
                files = [filepaths]
        elif isinstance(filepaths, (list, tuple)):
            files = list(filepaths)
        else:
            raise TypeError("filepaths must be a string or list of strings")

        if not files:
            raise FileNotFoundError(f"No files found matching: {filepaths}")

        self.filepath = files[0] if len(files) == 1 else f"{len(files)} files"
        
        voltages = []
        timestamps = []
        phone_timestamps = []
        device_ids = set()
        rec_names = set()
        data_types = set()

        for fpath in files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"File not found: {fpath}")
            
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if "phoneTimestamp" in obj:
                        phone_timestamps.append(obj["phoneTimestamp"])
                    if "deviceId" in obj:
                        device_ids.add(obj["deviceId"])
                    if "recordingName" in obj:
                        rec_names.add(obj["recordingName"])
                    if "dataType" in obj:
                        data_types.add(obj["dataType"])
                    
                    pts = obj.get("data", [])
                    for pt in pts:
                        voltages.append(pt.get("voltage", 0.0))
                        timestamps.append(pt.get("timeStamp", 0))

        if not voltages:
            raise ValueError(f"No ECG sample data found in {filepaths}")

        raw_v = np.array(voltages, dtype=float)
        ts_arr = np.array(timestamps, dtype=np.int64)

        # Sort chronologically in case split files were out of order
        sort_order = np.argsort(ts_arr)
        ts_sorted = ts_arr[sort_order]
        v_sorted = raw_v[sort_order]

        # Deduplicate consecutive identical timestamps if split logs overlap
        unique_mask = np.diff(ts_sorted, prepend=ts_sorted[0] - 1) > 0
        self.timestamps_ns = ts_sorted[unique_mask]
        self.raw_voltages = v_sorted[unique_mask]

        # Elapsed time in seconds
        t_zero = self.timestamps_ns[0]
        self.time_seconds = (self.timestamps_ns - t_zero) / 1e9

        # Calculate sampling rate
        if len(self.timestamps_ns) > 1:
            diffs_ns = np.diff(self.timestamps_ns)
            median_dt_s = np.median(diffs_ns) / 1e9
            if median_dt_s > 0:
                self.metadata.sampling_rate_hz = round(1.0 / median_dt_s, 2)
            else:
                self.metadata.sampling_rate_hz = 130.0
        else:
            self.metadata.sampling_rate_hz = 130.0

        self.metadata.device_id = list(device_ids)[0] if device_ids else "Unknown"
        self.metadata.recording_name = list(rec_names)[0] if rec_names else os.path.basename(str(files[0]))
        self.metadata.data_type = list(data_types)[0] if data_types else "ECG"
        self.metadata.total_samples = len(self.raw_voltages)
        self.metadata.duration_seconds = round(self.time_seconds[-1] - self.time_seconds[0], 2)
        
        if phone_timestamps:
            self.metadata.start_phone_timestamp_ms = min(phone_timestamps)
            try:
                dt = datetime.fromtimestamp(min(phone_timestamps) / 1000.0)
                self.metadata.start_iso_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                self.metadata.start_iso_time = "Unknown"

        return self

    def filter_signal(
        self,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: Optional[float] = 50.0,
        notch_q: float = 30.0
    ) -> np.ndarray:
        """
        Apply zero-phase Butterworth bandpass filter to remove baseline wander
        and high-frequency muscle noise, plus an optional mains notch filter (50Hz or 60Hz).
        """
        fs = self.metadata.sampling_rate_hz
        nyq = 0.5 * fs
        sig = self.raw_voltages.copy()

        # Bandpass filter (Butterworth order 2)
        # Ensure highcut is strictly below Nyquist
        effective_highcut = min(highcut, nyq - 1.0)
        low = max(0.1, lowcut) / nyq
        high = effective_highcut / nyq
        
        sos_bp = signal.butter(2, [low, high], btype="bandpass", output="sos")
        filtered = signal.sosfiltfilt(sos_bp, sig)

        # Notch filter for mains hum (if within Nyquist)
        if notch_freq and 0 < notch_freq < nyq:
            b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs=fs)
            filtered = signal.filtfilt(b_notch, a_notch, filtered)

        self.filtered_voltages = filtered
        return self.filtered_voltages

    def detect_r_peaks(
        self,
        min_bpm: float = 40.0,
        max_bpm: float = 210.0,
        settle_seconds: float = 2.0
    ) -> np.ndarray:
        """
        Adaptive QRS / R-peak detector for Polar H10 data.
        Uses enhanced derivative + energy envelope with adaptive local thresholding,
        followed by local parabolic/maximum refinement on the filtered ECG trace.
        """
        if len(self.filtered_voltages) == 0:
            self.filter_signal()

        fs = self.metadata.sampling_rate_hz
        min_distance_samples = int(fs * (60.0 / max_bpm))  # e.g. ~37 samples at 210 bpm / 130 Hz

        # 1. Bandpass filter for QRS energy extraction (8-22 Hz)
        sos_qrs = signal.butter(2, [8.0, 22.0], btype="bandpass", fs=fs, output="sos")
        sig_qrs = signal.sosfiltfilt(sos_qrs, self.raw_voltages)

        # 2. 5-point first derivative
        diff = np.diff(sig_qrs, prepend=sig_qrs[0])

        # 3. Squaring
        squared = diff ** 2

        # 4. Moving window integration (~120ms window)
        win_len = max(3, int(0.12 * fs))
        integrated = signal.convolve(squared, np.ones(win_len) / win_len, mode="same")

        # 5. Adaptive windowed peak detection (processes in 6-second overlapping chunks)
        chunk_len = int(fs * 6.0)
        overlap = int(fs * 1.0)
        step = chunk_len - overlap
        
        candidate_peaks = []
        total_len = len(integrated)

        for start in range(0, total_len, step):
            end = min(start + chunk_len, total_len)
            chunk = integrated[start:end]
            if len(chunk) < min_distance_samples:
                continue
            
            # Robust local threshold: median + factor * (90th percentile - median)
            med = np.median(chunk)
            p90 = np.percentile(chunk, 90)
            threshold = med + 0.32 * max(1e-5, p90 - med)
            
            pks, _ = signal.find_peaks(chunk, distance=min_distance_samples, height=threshold)
            
            # Refine peak to actual local extreme in filtered ECG
            search_radius = int(fs * 0.06)  # 60ms window
            for p in pks:
                global_idx = start + p
                if self.time_seconds[global_idx] < settle_seconds:
                    # Skip initial contact settling period if requested
                    continue
                
                l_idx = max(0, global_idx - search_radius)
                r_idx = min(len(self.filtered_voltages), global_idx + search_radius + 1)
                
                # Check whether R-wave is positive or inverted locally
                local_seg = self.filtered_voltages[l_idx:r_idx]
                if len(local_seg) > 0:
                    max_val = np.max(local_seg)
                    min_val = np.min(local_seg)
                    if abs(max_val) >= abs(min_val):
                        refined_idx = l_idx + int(np.argmax(local_seg))
                    else:
                        refined_idx = l_idx + int(np.argmin(local_seg))
                    candidate_peaks.append(refined_idx)

        # 6. Deduplicate and enforce minimum distance between peaks
        candidate_peaks = sorted(list(set(candidate_peaks)))
        refined_peaks = []
        for p in candidate_peaks:
            if not refined_peaks:
                refined_peaks.append(p)
            else:
                if (p - refined_peaks[-1]) >= min_distance_samples:
                    refined_peaks.append(p)
                else:
                    # Keep the one with larger absolute amplitude
                    prev = refined_peaks[-1]
                    if abs(self.filtered_voltages[p]) > abs(self.filtered_voltages[prev]):
                        refined_peaks[-1] = p

        self.r_peaks_idx = np.array(refined_peaks, dtype=int)

        # Sub-sample R-peak timing via parabolic interpolation around the integer peak,
        # instead of pinning every beat to the raw sample grid (see _parabolic_peak_offset).
        if len(self.r_peaks_idx) > 0:
            n_sig = len(self.filtered_voltages)
            peak_times = np.empty(len(self.r_peaks_idx), dtype=float)
            for k, idx in enumerate(self.r_peaks_idx):
                if 0 < idx < n_sig - 1:
                    v = self.filtered_voltages
                    offset = _parabolic_peak_offset(float(v[idx - 1]), float(v[idx]), float(v[idx + 1]))
                    dt_local = 0.5 * (self.time_seconds[idx + 1] - self.time_seconds[idx - 1])
                    peak_times[k] = self.time_seconds[idx] + offset * dt_local
                else:
                    peak_times[k] = self.time_seconds[idx]
            self.r_peak_times = peak_times
        else:
            self.r_peak_times = np.array([])

        # Calculate RR intervals and instantaneous HR
        if len(self.r_peak_times) > 1:
            self.rr_intervals_ms = np.diff(self.r_peak_times) * 1000.0  # in ms
            # Valid RR intervals (between 300 ms and 1800 ms = 33 bpm to 200 bpm)
            self.valid_beat_mask = (self.rr_intervals_ms >= 300.0) & (self.rr_intervals_ms <= 1800.0)
            
            # Reject sudden physiological jumps (> 25% difference from median of neighbors)
            if len(self.rr_intervals_ms) > 4:
                med_rr = np.median(self.rr_intervals_ms[self.valid_beat_mask]) if np.any(self.valid_beat_mask) else 600.0
                rel_diff = np.abs(self.rr_intervals_ms - med_rr) / med_rr
                self.valid_beat_mask = self.valid_beat_mask & (rel_diff < 0.45)
            
            self.instant_hr_bpm = 60000.0 / self.rr_intervals_ms
        else:
            self.rr_intervals_ms = np.array([])
            self.valid_beat_mask = np.array([])
            self.instant_hr_bpm = np.array([])

        return self.r_peaks_idx

    def compute_hrv_and_morphology(self) -> HRVMetrics:
        """Compute complete clinical / sports HRV metrics and average beat template."""
        if len(self.r_peaks_idx) == 0:
            self.detect_r_peaks()

        if len(self.rr_intervals_ms) < 3:
            return self.hrv

        # Normal-to-Normal (NN) intervals
        valid_mask = self.valid_beat_mask
        nn_ms = self.rr_intervals_ms[valid_mask] if np.any(valid_mask) else self.rr_intervals_ms
        valid_hr_bpm = 60000.0 / nn_ms

        self.hrv.total_beats = len(self.r_peaks_idx)
        self.hrv.mean_hr_bpm = round(float(np.mean(valid_hr_bpm)), 1)
        self.hrv.std_hr_bpm = round(float(np.std(valid_hr_bpm)), 1)
        self.hrv.min_hr_bpm = round(float(np.min(valid_hr_bpm)), 1)
        self.hrv.max_hr_bpm = round(float(np.max(valid_hr_bpm)), 1)

        self.hrv.mean_rr_ms = round(float(np.mean(nn_ms)), 1)
        self.hrv.std_rr_ms = round(float(np.std(nn_ms)), 1)
        self.hrv.median_rr_ms = round(float(np.median(nn_ms)), 1)
        self.hrv.sdnn_ms = round(float(np.std(nn_ms, ddof=1)), 1)

        # Successive differences
        diff_nn = np.diff(nn_ms)
        if len(diff_nn) > 0:
            self.hrv.rmssd_ms = round(float(np.sqrt(np.mean(diff_nn ** 2))), 1)
            self.hrv.sdsd_ms = round(float(np.std(diff_nn, ddof=1)), 1)
            self.hrv.pnn50_pct = round(float(100.0 * np.sum(np.abs(diff_nn) > 50.0) / len(diff_nn)), 1)
            self.hrv.pnn20_pct = round(float(100.0 * np.sum(np.abs(diff_nn) > 20.0) / len(diff_nn)), 1)

            # Poincaré geometry
            # SD1 = sqrt(0.5 * SDSD^2)
            sd1 = float(np.sqrt(0.5 * (self.hrv.sdsd_ms ** 2)))
            # SD2 = sqrt(2 * SDNN^2 - 0.5 * SDSD^2)
            sd2_sq = 2.0 * (self.hrv.sdnn_ms ** 2) - 0.5 * (self.hrv.sdsd_ms ** 2)
            sd2 = float(np.sqrt(max(0.0, sd2_sq)))
            self.hrv.sd1_ms = round(sd1, 1)
            self.hrv.sd2_ms = round(sd2, 1)
            self.hrv.sd1_sd2_ratio = round(sd1 / sd2, 3) if sd2 > 0 else 0.0

        # Quality score
        total_rr = len(self.rr_intervals_ms)
        valid_rr = int(np.sum(valid_mask))
        self.hrv.quality_score_pct = round(100.0 * valid_rr / max(1, total_rr), 1)

        # Frequency Domain: Lomb-Scargle PSD directly on the uneven NN time series.
        # Avoids the resample+interpolate step (which smears/distorts band power,
        # worse the longer and less stationary the recording e.g. over 24h).
        nn_times = self.r_peak_times[1:][valid_mask] if np.any(valid_mask) else self.r_peak_times[1:]
        long_recording = self.metadata.duration_seconds > 7200.0  # > 2h: resolve a ULF band too
        low_freq = 0.0001 if long_recording else 0.0033
        if len(nn_times) > 8 and (nn_times[-1] - nn_times[0]) > 30.0:
            try:
                freqs, psd = self._lomb_scargle_psd(nn_times, nn_ms, low_freq=low_freq)
                self.psd_freqs = freqs
                self.psd_values = psd

                df = freqs[1] - freqs[0]
                ulf_mask = freqs < 0.0033
                vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.40)

                if long_recording:
                    self.hrv.ulf_power_ms2 = round(float(np.sum(psd[ulf_mask]) * df), 1)
                self.hrv.vlf_power_ms2 = round(float(np.sum(psd[vlf_mask]) * df), 1)
                self.hrv.lf_power_ms2 = round(float(np.sum(psd[lf_mask]) * df), 1)
                self.hrv.hf_power_ms2 = round(float(np.sum(psd[hf_mask]) * df), 1)
                self.hrv.total_power_ms2 = round(
                    self.hrv.ulf_power_ms2 + self.hrv.vlf_power_ms2 + self.hrv.lf_power_ms2 + self.hrv.hf_power_ms2, 1
                )
                if self.hrv.hf_power_ms2 > 0:
                    self.hrv.lf_hf_ratio = round(self.hrv.lf_power_ms2 / self.hrv.hf_power_ms2, 2)

                lf_hf_sum = self.hrv.lf_power_ms2 + self.hrv.hf_power_ms2
                if lf_hf_sum > 0:
                    self.hrv.lf_nu = round(100.0 * self.hrv.lf_power_ms2 / lf_hf_sum, 1)
                    self.hrv.hf_nu = round(100.0 * self.hrv.hf_power_ms2 / lf_hf_sum, 1)
            except Exception:
                pass

        # Non-linear: Detrended Fluctuation Analysis (short/long-term scaling exponents)
        if len(nn_ms) >= 20:
            alpha1 = _dfa_alpha(nn_ms, np.unique(np.linspace(4, 16, 8).astype(int)))
            if alpha1 is not None:
                self.hrv.dfa_alpha1 = round(alpha1, 3)
        if len(nn_ms) >= 300:
            alpha2 = _dfa_alpha(nn_ms, np.unique(np.geomspace(16, min(64, len(nn_ms) // 4), 8).astype(int)))
            if alpha2 is not None:
                self.hrv.dfa_alpha2 = round(alpha2, 3)

        # ECG-derived respiration rate (R-wave amplitude modulation)
        self.compute_respiration_rate()

        # Ensemble Average Beat Morphology
        self._compute_average_beat()

        # Clinical Arrhythmia & Ectopic Detection
        self.detect_arrhythmias_and_ectopy()

        # Heart Rate Turbulence around detected PVCs (needs anomalies from above)
        self.compute_heart_rate_turbulence()

        return self.hrv

    def _lomb_scargle_psd(
        self,
        times: np.ndarray,
        values_ms: np.ndarray,
        low_freq: float = 0.0033,
        high_freq: float = 0.5,
        n_freqs: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lomb-Scargle periodogram of an unevenly-sampled NN-interval series, rescaled so
        that integrating the PSD over frequency reproduces the exact time-domain sample
        variance (SDNN^2). This calibrates absolute units without needing to derive the
        Lomb-Scargle normalization constant by hand, while preserving the true relative
        VLF/LF/HF power apportionment that a resample+Welch approach would distort.
        """
        centered = values_ms - np.mean(values_ms)
        freqs = np.linspace(low_freq, high_freq, n_freqs)
        ang_freqs = 2.0 * np.pi * freqs
        raw_power = signal.lombscargle(times, centered, ang_freqs, precenter=True)

        raw_total = np.trapezoid(raw_power, freqs)
        target_variance = float(np.var(centered, ddof=1))
        if raw_total > 1e-12 and target_variance > 0:
            scale = target_variance / raw_total
        else:
            scale = 0.0
        psd = raw_power * scale
        return freqs, psd

    def compute_respiration_rate(self) -> Optional[float]:
        """
        ECG-derived respiration (EDR): estimates breathing rate from R-wave amplitude
        modulation (respiratory sinus arrhythmia moves the heart relative to the strap,
        modulating apparent R amplitude at the breathing frequency), with no extra
        hardware needed. Requires >= 30s of beats.
        """
        if len(self.r_peaks_idx) < 20 or len(self.filtered_voltages) == 0:
            return None

        t = self.r_peak_times
        if (t[-1] - t[0]) < 30.0:
            return None

        v_mv = self.filtered_voltages / 1000.0
        r_amp = v_mv[self.r_peaks_idx]

        try:
            fs_resample = 4.0
            f_interp = interp1d(t, r_amp, kind="linear", fill_value="extrapolate")
            t_uniform = np.arange(t[0], t[-1], 1.0 / fs_resample)
            if len(t_uniform) < 16:
                return None
            amp_uniform = f_interp(t_uniform)
            amp_detrended = signal.detrend(amp_uniform)

            sos_resp = signal.butter(2, [0.1, 0.5], btype="bandpass", fs=fs_resample, output="sos")
            amp_filt = signal.sosfiltfilt(sos_resp, amp_detrended)

            nperseg = min(len(amp_filt), 256)
            freqs, psd = signal.welch(amp_filt, fs=fs_resample, nperseg=nperseg)
            band = (freqs >= 0.1) & (freqs <= 0.5)
            if not np.any(band) or np.max(psd[band]) <= 0:
                return None

            peak_freq = float(freqs[band][np.argmax(psd[band])])
            resp_bpm = round(peak_freq * 60.0, 1)

            self.resp_signal_time = t_uniform
            self.resp_signal_amp = amp_filt
            self.hrv.resp_rate_bpm = resp_bpm
            return resp_bpm
        except Exception:
            return None

    def compute_heart_rate_turbulence(self, n_pre: int = 2, n_post: int = 20) -> Optional[Tuple[float, float]]:
        """
        Approximate Heart Rate Turbulence (Schmidt et al. 1999 convention): for each
        detected PVC with a clean run of normal beats before and after, measures how
        the sinus rate accelerates then decelerates in response to the compensatory
        pause. Turbulence Onset (TO) and Turbulence Slope (TS) are averaged across all
        qualifying events. This is a research-grade approximation (beat classification
        here is a heuristic, not a validated clinical HRT implementation).
        """
        pvc_events = [a for a in self.anomalies if a.anomaly_type == "PVC"]
        if not pvc_events or len(self.rr_intervals_ms) < (n_pre + n_post + 3):
            return None

        rr = self.rr_intervals_ms
        labels = self.beat_labels
        to_values = []
        ts_values = []

        for an in pvc_events:
            i = an.beat_index - 1  # 0-indexed position of the PVC beat in r_peaks_idx
            pre_start = i - 1 - n_pre
            post_end = i + 1 + n_post
            if pre_start < 0 or post_end > len(rr):
                continue
            # Require the surrounding beats to be classified normal (clean window)
            surrounding = labels[max(0, i - 1 - n_pre) : min(len(labels), i + 2 + n_post)]
            if any(lbl in ("PAC", "PVC") for k, lbl in enumerate(surrounding) if k != (i - pre_start)):
                continue

            rr_pre = rr[pre_start:i - 1]      # RR(-n_pre) .. RR(-1)
            rr_post = rr[i + 1:post_end]        # RR(+1) .. RR(+n_post), after the compensatory pause
            if len(rr_pre) < n_pre or len(rr_post) < 4:
                continue

            pre_mean = np.mean(rr_pre)
            if pre_mean <= 0:
                continue
            to = 100.0 * (np.mean(rr_post[:2]) - pre_mean) / pre_mean
            to_values.append(to)

            # Turbulence slope: steepest positive slope over any 5 consecutive post-beats
            best_slope = 0.0
            for j in range(len(rr_post) - 4):
                seg = rr_post[j:j + 5]
                slope = np.polyfit(np.arange(5), seg, 1)[0]
                if slope > best_slope:
                    best_slope = slope
            ts_values.append(best_slope)

        if not to_values:
            return None

        self.hrv.hrt_turbulence_onset_pct = round(float(np.mean(to_values)), 2)
        self.hrv.hrt_turbulence_slope_ms = round(float(np.mean(ts_values)), 2)
        self.hrv.hrt_qualifying_events = len(to_values)
        return self.hrv.hrt_turbulence_onset_pct, self.hrv.hrt_turbulence_slope_ms

    def compute_windowed_hrv(self, window_minutes: float = 5.0, step_minutes: Optional[float] = None) -> List["WindowMetrics"]:
        """
        Split the recording into (by default non-overlapping) short-term windows and
        compute HRV within each - the standard Task Force (1996) way to handle a long,
        non-stationary recording (e.g. 24h: sleep vs. waking vs. exertion) instead of
        one single-window spectral estimate that would blur all of that together.
        Also the basis for a day/night HR & HRV trend plot.
        """
        if len(self.r_peak_times) < 10:
            return []
        if len(self.rr_intervals_ms) == 0:
            self.compute_hrv_and_morphology()

        step_minutes = step_minutes or window_minutes
        win_s = window_minutes * 60.0
        step_s = step_minutes * 60.0

        beat_times = self.r_peak_times[1:]  # aligned with rr_intervals_ms
        anomalies_by_time = self.anomalies

        windows: List[WindowMetrics] = []
        w_start = beat_times[0] if len(beat_times) else 0.0
        t_end = beat_times[-1] if len(beat_times) else 0.0

        while w_start < t_end:
            w_end = w_start + win_s
            in_win = (beat_times >= w_start) & (beat_times < w_end)
            n_in_win = int(np.sum(in_win))

            if n_in_win >= 5:
                valid_in_win = in_win & self.valid_beat_mask
                nn = self.rr_intervals_ms[valid_in_win] if np.any(valid_in_win) else self.rr_intervals_ms[in_win]

                if len(nn) >= 3:
                    hr_vals = 60000.0 / nn
                    diffs = np.diff(nn)
                    sdnn = float(np.std(nn, ddof=1)) if len(nn) > 1 else 0.0
                    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
                    pnn50 = float(100.0 * np.sum(np.abs(diffs) > 50.0) / len(diffs)) if len(diffs) > 0 else 0.0

                    lf_p, hf_p, lf_hf = 0.0, 0.0, 0.0
                    win_beat_times = beat_times[valid_in_win] if np.any(valid_in_win) else beat_times[in_win]
                    if len(win_beat_times) > 8 and (win_beat_times[-1] - win_beat_times[0]) > 60.0:
                        try:
                            freqs, psd = self._lomb_scargle_psd(win_beat_times, nn, low_freq=0.04, high_freq=0.45, n_freqs=300)
                            df = freqs[1] - freqs[0]
                            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
                            lf_p = float(np.sum(psd[lf_mask]) * df)
                            hf_p = float(np.sum(psd[hf_mask]) * df)
                            lf_hf = round(lf_p / hf_p, 2) if hf_p > 0 else 0.0
                        except Exception:
                            pass

                    pac_c = sum(1 for a in anomalies_by_time if a.anomaly_type == "PAC" and w_start <= a.time_seconds < w_end)
                    pvc_c = sum(1 for a in anomalies_by_time if a.anomaly_type == "PVC" and w_start <= a.time_seconds < w_end)
                    quality = round(100.0 * np.sum(valid_in_win) / max(1, n_in_win), 1)

                    windows.append(WindowMetrics(
                        start_s=round(float(w_start), 1),
                        end_s=round(float(w_end), 1),
                        n_beats=n_in_win,
                        mean_hr_bpm=round(float(np.mean(hr_vals)), 1),
                        sdnn_ms=round(sdnn, 1),
                        rmssd_ms=round(rmssd, 1),
                        pnn50_pct=round(pnn50, 1),
                        lf_power_ms2=round(lf_p, 1),
                        hf_power_ms2=round(hf_p, 1),
                        lf_hf_ratio=lf_hf,
                        pac_count=pac_c,
                        pvc_count=pvc_c,
                        quality_pct=quality,
                    ))

            w_start += step_s

        self.windowed_hrv = windows
        return windows

    def export_windowed_csv(self, output_path: str, window_minutes: float = 5.0):
        """Export the short-term HRV trend (windowed_hrv) to CSV for external plotting."""
        if not self.windowed_hrv:
            self.compute_windowed_hrv(window_minutes=window_minutes)

        lines = ["window_start_s,window_end_s,n_beats,mean_hr_bpm,sdnn_ms,rmssd_ms,pnn50_pct,lf_power_ms2,hf_power_ms2,lf_hf_ratio,pac_count,pvc_count,quality_pct\n"]
        for w in self.windowed_hrv:
            lines.append(
                f"{w.start_s},{w.end_s},{w.n_beats},{w.mean_hr_bpm},{w.sdnn_ms},{w.rmssd_ms},"
                f"{w.pnn50_pct},{w.lf_power_ms2},{w.hf_power_ms2},{w.lf_hf_ratio},{w.pac_count},{w.pvc_count},{w.quality_pct}\n"
            )
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def detect_arrhythmias_and_ectopy(self) -> List[BeatAnomaly]:
        """
        Classifies every beat into Normal (N), PAC (Premature Atrial Contraction),
        PVC (Premature Ventricular Contraction), Sinus Pause, or Tachy/Bradycardia bursts.
        """
        if len(self.r_peaks_idx) < 5:
            return []

        fs = self.metadata.sampling_rate_hz
        v_mv = self.filtered_voltages / 1000.0
        n_beats = len(self.r_peaks_idx)
        self.beat_labels = ["N"] * n_beats
        self.anomalies = []

        # 1. Build normal sinus QRS template (-80ms to +120ms)
        w_pre = max(2, int(0.08 * fs))
        w_post = max(2, int(0.12 * fs))
        
        # Collect candidate beats from middle region with normal RR intervals
        normal_qrs = []
        med_rr_overall = np.median(self.rr_intervals_ms) if len(self.rr_intervals_ms) > 0 else 600.0
        for i in range(min(15, n_beats), min(n_beats - 5, 80)):
            if i - 1 < len(self.rr_intervals_ms):
                if 0.88 * med_rr_overall <= self.rr_intervals_ms[i - 1] <= 1.15 * med_rr_overall:
                    p = self.r_peaks_idx[i]
                    if p - w_pre >= 0 and p + w_post <= len(v_mv):
                        seg = v_mv[p - w_pre : p + w_post]
                        normal_qrs.append(seg - np.mean(seg))

        if not normal_qrs:
            for p in self.r_peaks_idx[5:25]:
                if p - w_pre >= 0 and p + w_post <= len(v_mv):
                    seg = v_mv[p - w_pre : p + w_post]
                    normal_qrs.append(seg - np.mean(seg))

        median_template = np.median(normal_qrs, axis=0) if normal_qrs else None

        # 2. Analyze beat-by-beat coupling and morphology
        # Precompute the local rolling median RR (window of 11 beats, shrinking at the
        # edges) for every beat in one vectorized pass instead of a per-beat np.median
        # call - matters once this loop is running over ~100k+ beats on a 24h recording.
        local_median_rr_arr = (
            pd.Series(self.rr_intervals_ms).rolling(window=11, center=True, min_periods=1).median().to_numpy()
        )

        for i in range(1, n_beats - 1):
            rr_pre = self.rr_intervals_ms[i - 1]
            rr_post = self.rr_intervals_ms[i]

            local_median_rr = float(local_median_rr_arr[i])

            pre_ratio = rr_pre / max(1.0, local_median_rr)
            post_ratio = rr_post / max(1.0, local_median_rr)
            sum_ratio = (rr_pre + rr_post) / (2.0 * max(1.0, local_median_rr))

            # Beat QRS morphology correlation
            p = self.r_peaks_idx[i]
            corr = 1.0
            if median_template is not None and p - w_pre >= 0 and p + w_post <= len(v_mv):
                qrs_seg = v_mv[p - w_pre : p + w_post] - np.mean(v_mv[p - w_pre : p + w_post])
                std_prod = (np.std(median_template) * np.std(qrs_seg))
                if std_prod > 1e-6:
                    corr = float(np.corrcoef(median_template, qrs_seg)[0, 1])

            # Case A: Premature Beat (RR_pre <= 0.85 * local median)
            if pre_ratio <= 0.85:
                # PVC if correlation is low (< 0.80) or inverted morphology
                if corr < 0.80:
                    anomaly_type = "PVC"
                    desc = f"PVC: Premature ventricular beat ({rr_pre:.0f}ms, {pre_ratio:.2f}x normal). Aberrant/wide QRS (r={corr:.2f}). Compensatory pause={rr_post:.0f}ms."
                    self.beat_labels[i] = "PVC"
                else:
                    anomaly_type = "PAC"
                    desc = f"PAC: Premature atrial beat ({rr_pre:.0f}ms, {pre_ratio:.2f}x normal). Normal narrow QRS (r={corr:.2f}). Compensatory pause={rr_post:.0f}ms."
                    self.beat_labels[i] = "PAC"

                self.anomalies.append(BeatAnomaly(
                    beat_index=i + 1,
                    sample_index=p,
                    time_seconds=float(self.r_peak_times[i]),
                    anomaly_type=anomaly_type,
                    rr_pre_ms=round(rr_pre, 1),
                    rr_post_ms=round(rr_post, 1),
                    pre_ratio=round(pre_ratio, 2),
                    post_ratio=round(post_ratio, 2),
                    qrs_correlation=round(corr, 3),
                    description=desc
                ))

            # Case B: Sinus Pause / Dropped Beat
            elif pre_ratio >= 1.48:
                # Ensure this isn't simply the compensatory pause of the previous premature beat
                prev_is_premature = (i >= 2 and self.beat_labels[i - 1] in ["PAC", "PVC"])
                if not prev_is_premature:
                    anomaly_type = "PAUSE"
                    desc = f"Sinus Pause: Prolonged RR interval of {rr_pre:.0f}ms ({pre_ratio:.2f}x normal)."
                    self.beat_labels[i] = "PAUSE"
                    self.anomalies.append(BeatAnomaly(
                        beat_index=i + 1,
                        sample_index=p,
                        time_seconds=float(self.r_peak_times[i]),
                        anomaly_type=anomaly_type,
                        rr_pre_ms=round(rr_pre, 1),
                        rr_post_ms=round(rr_post, 1),
                        pre_ratio=round(pre_ratio, 2),
                        post_ratio=round(post_ratio, 2),
                        qrs_correlation=round(corr, 3),
                        description=desc
                    ))

        # 3. Detect Couplets, Bigeminy, Trigeminy
        for idx, an in enumerate(self.anomalies):
            curr_b = an.beat_index
            if idx > 0 and self.anomalies[idx - 1].beat_index == curr_b - 1:
                an.pattern = "Couplet"
                self.anomalies[idx - 1].pattern = "Couplet"
            elif idx > 0 and self.anomalies[idx - 1].beat_index == curr_b - 2:
                an.pattern = "Bigeminy"

        # 4. Tachycardia & Bradycardia runs (>= 3 consecutive beats)
        tachy_runs = 0
        brady_runs = 0
        current_tachy = 0
        current_brady = 0

        for hr in self.instant_hr_bpm:
            if hr > 115.0:
                current_tachy += 1
            else:
                if current_tachy >= 3:
                    tachy_runs += 1
                current_tachy = 0

            if hr < 55.0:
                current_brady += 1
            else:
                if current_brady >= 3:
                    brady_runs += 1
                current_brady = 0

        if current_tachy >= 3:
            tachy_runs += 1
        if current_brady >= 3:
            brady_runs += 1

        # Update HRV anomaly counts
        self.hrv.pvc_count = sum(1 for a in self.anomalies if a.anomaly_type == "PVC")
        self.hrv.pac_count = sum(1 for a in self.anomalies if a.anomaly_type == "PAC")
        self.hrv.pause_count = sum(1 for a in self.anomalies if a.anomaly_type == "PAUSE")
        self.hrv.tachycardia_episodes = tachy_runs
        self.hrv.bradycardia_episodes = brady_runs
        
        total_ectopics = self.hrv.pvc_count + self.hrv.pac_count
        self.hrv.ectopic_burden_pct = round(100.0 * total_ectopics / max(1, n_beats), 2)

        return self.anomalies

    def _compute_average_beat(self, pre_ms: float = 200.0, post_ms: float = 400.0):
        """Extract individual beats aligned on R-peaks and compute canonical template."""
        fs = self.metadata.sampling_rate_hz
        pre_samples = int(fs * (pre_ms / 1000.0))
        post_samples = int(fs * (post_ms / 1000.0))
        total_beat_len = pre_samples + post_samples

        beats = []
        # Convert voltage from µV to mV for standard clinical display (1 mV = 1000 µV)
        v_mv = self.filtered_voltages / 1000.0

        for r_idx in self.r_peaks_idx:
            start_idx = r_idx - pre_samples
            end_idx = r_idx + post_samples
            if start_idx >= 0 and end_idx <= len(v_mv):
                segment = v_mv[start_idx:end_idx].copy()
                # Baseline align to start of P-wave region (mean of first 30ms)
                p_base = np.mean(segment[:max(1, int(0.03 * fs))])
                segment -= p_base
                # Discard wild motion artifact spikes (> 5 mV or < -5 mV)
                if np.max(np.abs(segment)) < 5.0:
                    beats.append(segment)

        if len(beats) > 3:
            beats_arr = np.array(beats)
            self.all_beat_waves = beats[: min(60, len(beats))]  # sample of individual beats
            self.average_beat_mv = np.mean(beats_arr, axis=0)
            self.average_beat_std_mv = np.std(beats_arr, axis=0)
            self.average_beat_time_ms = np.linspace(-pre_ms, post_ms, total_beat_len)

    def export_rr_csv(self, output_path: str):
        """Export RR intervals and instantaneous heart rates to CSV."""
        if len(self.rr_intervals_ms) == 0:
            self.compute_hrv_and_morphology()

        lines = ["beat_index,r_peak_time_sec,rr_interval_ms,heart_rate_bpm,classification,pattern,is_valid\n"]
        for i, (t_peak, rr, hr, is_valid) in enumerate(
            zip(self.r_peak_times[1:], self.rr_intervals_ms, self.instant_hr_bpm, self.valid_beat_mask)
        ):
            beat_idx = i + 2
            lbl = self.beat_labels[i + 1] if i + 1 < len(self.beat_labels) else "N"
            # Find if this beat has an anomaly
            matched_an = next((a for a in self.anomalies if a.beat_index == beat_idx), None)
            pat = matched_an.pattern if matched_an and matched_an.pattern else "None"
            lines.append(f"{beat_idx},{t_peak:.3f},{rr:.1f},{hr:.1f},{lbl},{pat},{1 if is_valid else 0}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def print_summary(self):
        """Print a clinical & signal summary report to stdout."""
        if len(self.r_peaks_idx) == 0:
            self.compute_hrv_and_morphology()

        m = self.metadata
        h = self.hrv

        if RICH_AVAILABLE:
            console = Console()
            
            # Title Panel
            title_text = f"[bold cyan]Polar H10 ECG Analysis Report[/bold cyan]\n" \
                         f"Recording: [yellow]{m.recording_name}[/yellow] | Device: [green]{m.device_id}[/green]\n" \
                         f"Date: [white]{m.start_iso_time or 'Unknown'}[/white] | Duration: [magenta]{m.duration_seconds:.1f}s ({m.duration_seconds/60:.2f} min)[/magenta] | Rate: [cyan]{m.sampling_rate_hz} Hz[/cyan]"
            console.print(Panel(title_text, box=box.ROUNDED, expand=False))

            # Table for HR & Time Domain
            t_time = Table(title="Heart Rate & Time-Domain HRV", box=box.SIMPLE_HEAVY)
            t_time.add_column("Metric", style="bold white")
            t_time.add_column("Value", style="bold green", justify="right")
            t_time.add_column("Unit", style="dim")
            t_time.add_column("Clinical / Reference Context", style="cyan")

            t_time.add_row("Total Beats Detected", str(h.total_beats), "beats", "R-peaks validated")
            t_time.add_row("Mean Heart Rate", f"{h.mean_hr_bpm:.1f}", "BPM", "Resting normal: 60 - 100")
            t_time.add_row("HR Range (Min / Max)", f"{h.min_hr_bpm:.1f} - {h.max_hr_bpm:.1f}", "BPM", f"Std Dev: {h.std_hr_bpm:.1f}")
            t_time.add_row("Mean RR (NN)", f"{h.mean_rr_ms:.1f}", "ms", "Average beat-to-beat time")
            t_time.add_row("SDNN", f"{h.sdnn_ms:.1f}", "ms", "Overall HRV (>50ms normal)")
            t_time.add_row("RMSSD", f"{h.rmssd_ms:.1f}", "ms", "Parasympathetic / Vagal tone (>42ms high)")
            t_time.add_row("pNN50", f"{h.pnn50_pct:.1f}", "%", "Beats differing >50ms (>3% normal)")
            t_time.add_row("pNN20", f"{h.pnn20_pct:.1f}", "%", "Beats differing >20ms")
            t_time.add_row("Signal Quality Score", f"{h.quality_score_pct:.1f}", "%", "Valid NN beats percentage")
            t_time.add_row("Respiration Rate (EDR)", f"{h.resp_rate_bpm:.1f}" if h.resp_rate_bpm else "N/A", "breaths/min", "ECG-derived (R-amplitude modulation)")
            t_time.add_row("DFA alpha1", f"{h.dfa_alpha1:.3f}" if h.dfa_alpha1 else "N/A", "-", "Short-term scaling (~1.0 balanced; needs 4-16 beat boxes)")
            if h.dfa_alpha2:
                t_time.add_row("DFA alpha2", f"{h.dfa_alpha2:.3f}", "-", "Long-term scaling (needs >=300 beats)")
            console.print(t_time)

            # Table for Arrhythmia & Ectopic Beats
            t_arrhythmia = Table(title="Arrhythmia & Ectopic Beat Detection", box=box.SIMPLE_HEAVY)
            t_arrhythmia.add_column("Category / Finding", style="bold white")
            t_arrhythmia.add_column("Count", style="bold yellow", justify="right")
            t_arrhythmia.add_column("Burden / Details", style="cyan")

            t_arrhythmia.add_row(
                "Premature Atrial Contractions (PAC)",
                str(h.pac_count),
                f"{h.pac_count / max(1, h.total_beats) * 100:.1f}% burden (Supraventricular ectopy)"
            )
            t_arrhythmia.add_row(
                "Premature Ventricular Contractions (PVC)",
                str(h.pvc_count),
                f"{h.pvc_count / max(1, h.total_beats) * 100:.1f}% burden (Ventricular ectopy)"
            )
            t_arrhythmia.add_row("Sinus Pauses / Dropped Beats", str(h.pause_count), "RR interval > 1.5x baseline")
            t_arrhythmia.add_row("Tachycardia Episodes (>115 BPM)", str(h.tachycardia_episodes), "Runs >= 3 consecutive beats")
            t_arrhythmia.add_row("Bradycardia Episodes (<55 BPM)", str(h.bradycardia_episodes), "Runs >= 3 consecutive beats")
            t_arrhythmia.add_row("Total Ectopic Burden", f"{h.ectopic_burden_pct:.1f}%", "Overall arrhythmic beat percentage")
            if h.hrt_qualifying_events > 0:
                t_arrhythmia.add_row(
                    "Heart Rate Turbulence (approx.)",
                    f"TO={h.hrt_turbulence_onset_pct:.2f}% / TS={h.hrt_turbulence_slope_ms:.2f} ms/beat",
                    f"From {h.hrt_qualifying_events} qualifying PVC(s); research-grade, not clinically validated"
                )
            console.print(t_arrhythmia)

            # Detailed Anomaly Log (if any found)
            if self.anomalies:
                t_log = Table(title=f"Detected Rhythm Anomalies ({len(self.anomalies)} events)", box=box.SIMPLE)
                t_log.add_column("Beat #", style="bold white", justify="right")
                t_log.add_column("Time (s)", justify="right")
                t_log.add_column("Type", style="bold magenta")
                t_log.add_column("Pattern", style="bold yellow")
                t_log.add_column("RR Pre", justify="right")
                t_log.add_column("Pause Post", justify="right")
                t_log.add_column("QRS Corr", justify="right")
                t_log.add_column("Clinical Finding", style="dim")

                for an in self.anomalies[:15]:  # Top 15 in console
                    t_log.add_row(
                        f"#{an.beat_index}",
                        f"{an.time_seconds:.2f}s",
                        an.anomaly_type,
                        an.pattern or "Isolated",
                        f"{an.rr_pre_ms:.0f}ms ({an.pre_ratio:.2f}x)",
                        f"{an.rr_post_ms:.0f}ms ({an.post_ratio:.2f}x)",
                        f"{an.qrs_correlation:.2f}",
                        an.description.split(".")[0]
                    )
                if len(self.anomalies) > 15:
                    t_log.add_row("...", "...", "...", "...", "...", "...", "...", f"+ {len(self.anomalies) - 15} more in HTML report")
                console.print(t_log)

            # Table for Frequency Domain & Non-linear
            t_freq = Table(title="Frequency Domain (Welch PSD) & Poincaré", box=box.SIMPLE_HEAVY)
            t_freq.add_column("Parameter", style="bold white")
            t_freq.add_column("Value", style="bold magenta", justify="right")
            t_freq.add_column("Band / Formula", style="dim")
            t_freq.add_column("Autonomic Significance", style="cyan")

            if h.ulf_power_ms2:
                t_freq.add_row("ULF Power", f"{h.ulf_power_ms2:.1f}", "< 0.0033 Hz", "Ultra Low Frequency (24h+ only)")
            t_freq.add_row("VLF Power", f"{h.vlf_power_ms2:.1f}", "0.0033 - 0.04 Hz", "Very Low Frequency")
            t_freq.add_row("LF Power", f"{h.lf_power_ms2:.1f}", "0.04 - 0.15 Hz", "Sympathetic & Baroreflex")
            t_freq.add_row("HF Power", f"{h.hf_power_ms2:.1f}", "0.15 - 0.40 Hz", "Parasympathetic / Vagal (RSA)")
            t_freq.add_row("Total Power", f"{h.total_power_ms2:.1f}", "VLF + LF + HF", "Overall autonomic power")
            t_freq.add_row("LF / HF Ratio", f"{h.lf_hf_ratio:.2f}", "LF / HF", "Autonomic balance (0.5 - 2.0)")
            t_freq.add_row("Poincaré SD1", f"{h.sd1_ms:.1f}", "ms", "Short-term beat variability")
            t_freq.add_row("Poincaré SD2", f"{h.sd2_ms:.1f}", "ms", "Long-term beat variability")
            t_freq.add_row("SD1 / SD2 Ratio", f"{h.sd1_sd2_ratio:.3f}", "Ratio", "Non-linear autonomic balance")
            console.print(t_freq)

            if self.windowed_hrv:
                console.print(f"[dim]Windowed HRV trend: {len(self.windowed_hrv)} short-term window(s) computed - export with export_windowed_csv().[/dim]")
        else:
            print("=" * 60)
            print(f"Polar H10 ECG Analysis: {m.recording_name}")
            print(f"Device: {m.device_id} | Date: {m.start_iso_time} | Duration: {m.duration_seconds:.1f}s")
            print("-" * 60)
            print(f"Beats: {h.total_beats} | Mean HR: {h.mean_hr_bpm:.1f} BPM (range: {h.min_hr_bpm:.1f} - {h.max_hr_bpm:.1f})")
            print(f"PACs: {h.pac_count} | PVCs: {h.pvc_count} | Pauses: {h.pause_count} | Ectopic Burden: {h.ectopic_burden_pct:.1f}%")
            print(f"Mean RR: {h.mean_rr_ms:.1f} ms | SDNN: {h.sdnn_ms:.1f} ms | RMSSD: {h.rmssd_ms:.1f} ms")
            print(f"pNN50: {h.pnn50_pct:.1f}% | Quality: {h.quality_score_pct:.1f}%")
            print(f"LF/HF Ratio: {h.lf_hf_ratio:.2f} (LF: {h.lf_power_ms2:.1f} ms², HF: {h.hf_power_ms2:.1f} ms²)")
            print(f"Poincaré: SD1={h.sd1_ms:.1f} ms, SD2={h.sd2_ms:.1f} ms (SD1/SD2: {h.sd1_sd2_ratio:.3f})")
            dfa_line = f"DFA alpha1: {h.dfa_alpha1:.3f}"
            if h.dfa_alpha2:
                dfa_line += f" | alpha2: {h.dfa_alpha2:.3f}"
            if h.resp_rate_bpm:
                dfa_line += f" | Respiration Rate: {h.resp_rate_bpm:.1f} breaths/min"
            print(dfa_line)
            if h.hrt_qualifying_events > 0:
                print(f"Heart Rate Turbulence (approx., {h.hrt_qualifying_events} events): TO={h.hrt_turbulence_onset_pct:.2f}% TS={h.hrt_turbulence_slope_ms:.2f} ms/beat")
            if self.windowed_hrv:
                print(f"Windowed HRV trend: {len(self.windowed_hrv)} short-term window(s) computed - export with export_windowed_csv().")
            print("=" * 60)
