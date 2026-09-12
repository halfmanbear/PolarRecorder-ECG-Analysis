"""
ECG Visualizer & Interactive HTML Dashboard Generator
Creates a clinical & athletic grade interactive ECG visualization suite using Plotly.
Includes:
- Zoomable & Pannable ECG Lead Strip with color-coded ectopic beat highlights (PAC, PVC, Pauses)
- Interactive Arrhythmia Log Table with one-click "Jump to Beat" zoom
- Heart Rate & RR Interval Tachogram with Bradycardia/Tachycardia threshold bands
- Ensemble Average Beat Waveform (P-QRS-T template with ±1 SD envelope and Ectopic overlays)
- Poincaré Non-Linear HRV Plot with SD1/SD2 Ellipse
- HRV Welch Power Spectral Density (VLF, LF, HF bands)
- Responsive Metric Cards & Arrhythmia Burden Summary
"""

import json
import os
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ecg_processor import ECGAnalyzer, BeatAnomaly


def robust_axis_range(values, low_pct: float = 1.0, high_pct: float = 99.0, pad_frac: float = 0.12) -> Optional[list]:
    """Percentile-based y-axis range so a handful of outlier beats (e.g. a single
    ectopic-driven HR spike) don't stretch the default view and flatten the rest
    of the trace. Points outside the range are still reachable by zooming/autoscale
    since this only sets the initial `range`, not `fixedrange`."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        pad = max(abs(lo) * pad_frac, 1.0)
        return [lo - pad, hi + pad]
    pad = (hi - lo) * pad_frac
    return [lo - pad, hi + pad]


def min_max_decimate(x: np.ndarray, y: np.ndarray, max_points: int = 25000) -> tuple:
    """
    Peak-preserving min-max decimation for multi-hour/24h recordings.
    Guarantees 100% of R-peaks and negative Q/S deflections are preserved
    while reducing millions of points to a smooth 60fps rendering size.
    """
    n = len(x)
    if n <= max_points:
        return x, y
    bucket_size = int(np.ceil(n / (max_points / 2)))
    n_full = (n // bucket_size) * bucket_size
    if n_full == 0:
        return x, y
    
    y_full = y[:n_full].reshape(-1, bucket_size)
    x_full = x[:n_full].reshape(-1, bucket_size)
    
    min_idx = np.argmin(y_full, axis=1)
    max_idx = np.argmax(y_full, axis=1)
    row_idx = np.arange(y_full.shape[0])
    
    i1 = np.minimum(min_idx, max_idx)
    i2 = np.maximum(min_idx, max_idx)
    
    x1 = x_full[row_idx, i1]
    y1 = y_full[row_idx, i1]
    x2 = x_full[row_idx, i2]
    y2 = y_full[row_idx, i2]
    
    x_dec = np.empty(len(x1) * 2, dtype=x.dtype)
    y_dec = np.empty(len(y1) * 2, dtype=y.dtype)
    x_dec[0::2] = x1
    x_dec[1::2] = x2
    y_dec[0::2] = y1
    y_dec[1::2] = y2
    return x_dec, y_dec


class ECGVisualizer:
    def __init__(self, analyzer: ECGAnalyzer):
        self.analyzer = analyzer

    def generate_html_report(
        self,
        output_filepath: str,
        initial_window_seconds: float = 12.0,
        dark_mode: bool = True
    ) -> str:
        """
        Build an interactive HTML dashboard with arrhythmia highlighting and save to disk.
        """
        a = self.analyzer
        if len(a.r_peaks_idx) == 0:
            a.compute_hrv_and_morphology()

        m = a.metadata
        h = a.hrv
        resp_rate_display = f"{h.resp_rate_bpm:.1f}" if h.resp_rate_bpm else "N/A"
        dfa_alpha2_display = f"(alpha2: {h.dfa_alpha2:.3f})" if h.dfa_alpha2 else ""

        bg_color = "#0f172a" if dark_mode else "#ffffff"
        card_bg = "#1e293b" if dark_mode else "#f8fafc"
        text_color = "#f8fafc" if dark_mode else "#0f172a"
        text_muted = "#94a3b8" if dark_mode else "#64748b"
        border_color = "#334155" if dark_mode else "#e2e8f0"
        grid_color = "#334155" if dark_mode else "#f1f5f9"
        accent_blue = "#38bdf8"
        accent_red = "#f43f5e"
        accent_green = "#10b981"
        accent_purple = "#a855f7"
        accent_amber = "#f59e0b"
        accent_pvc = "#ef4444"
        accent_pac = "#f97316"

        # -------------------------------------------------------------
        # 1. ECG Strip Plot (Filtered + Raw + Color-coded Beats)
        # -------------------------------------------------------------
        fig_ecg = go.Figure()

        raw_mv = a.raw_voltages / 1000.0
        filt_mv = a.filtered_voltages / 1000.0

        # Peak-preserving decimation for smooth 60fps rendering on 24h recordings
        x_filt_plot, y_filt_plot = min_max_decimate(a.time_seconds, filt_mv, max_points=26000)
        x_raw_plot, y_raw_plot = min_max_decimate(a.time_seconds, raw_mv, max_points=26000)

        # Raw ECG trace
        fig_ecg.add_trace(go.Scattergl(
            x=x_raw_plot,
            y=y_raw_plot,
            mode="lines",
            name="Raw ECG (µV/1000)",
            line=dict(color="#475569" if dark_mode else "#cbd5e1", width=1),
            opacity=0.4,
            visible="legendonly",
            hovertemplate="Time: %{x:.3f} s<br>Raw: %{y:.3f} mV<extra></extra>"
        ))

        # Filtered ECG trace
        fig_ecg.add_trace(go.Scattergl(
            x=x_filt_plot,
            y=y_filt_plot,
            mode="lines",
            name="Filtered ECG (0.5 - 40 Hz)",
            line=dict(color=accent_blue, width=1.6),
            hovertemplate="Time: %{x:.3f} s<br>Voltage: %{y:.3f} mV<extra></extra>"
        ))

        # Classify and separate beat markers
        normal_x, normal_y, normal_txt = [], [], []
        pac_x, pac_y, pac_txt = [], [], []
        pvc_x, pvc_y, pvc_txt = [], [], []
        pause_x, pause_y, pause_txt = [], [], []

        # Map anomalies by beat index (1-indexed)
        anomaly_map = {an.beat_index: an for an in a.anomalies}

        for i, r_idx in enumerate(a.r_peaks_idx):
            beat_num = i + 1
            t_sec = a.r_peak_times[i]
            v_peak = filt_mv[r_idx]
            
            hr_str = f"{a.instant_hr_bpm[i-1]:.1f} BPM" if (i > 0 and i - 1 < len(a.instant_hr_bpm)) else "N/A"
            rr_str = f"{a.rr_intervals_ms[i-1]:.1f} ms" if (i > 0 and i - 1 < len(a.rr_intervals_ms)) else "N/A"

            if beat_num in anomaly_map:
                an = anomaly_map[beat_num]
                if an.anomaly_type == "PAC":
                    pac_x.append(t_sec)
                    pac_y.append(v_peak)
                    pac_txt.append(
                        f"<b>⚠️ Premature Atrial Contraction (PAC)</b><br>"
                        f"Beat #{beat_num} at {t_sec:.3f}s<br>"
                        f"Coupling RR: {an.rr_pre_ms:.0f} ms ({int((1-an.pre_ratio)*100)}% early)<br>"
                        f"Post Pause: {an.rr_post_ms:.0f} ms ({an.post_ratio:.2f}x)<br>"
                        f"QRS Match: {int(an.qrs_correlation*100)}% (Narrow QRS)<br>"
                        f"Pattern: {an.pattern or 'Isolated'}"
                    )
                elif an.anomaly_type == "PVC":
                    pvc_x.append(t_sec)
                    pvc_y.append(v_peak)
                    pvc_txt.append(
                        f"<b>🚨 Premature Ventricular Contraction (PVC)</b><br>"
                        f"Beat #{beat_num} at {t_sec:.3f}s<br>"
                        f"Coupling RR: {an.rr_pre_ms:.0f} ms ({int((1-an.pre_ratio)*100)}% early)<br>"
                        f"Post Pause: {an.rr_post_ms:.0f} ms ({an.post_ratio:.2f}x)<br>"
                        f"QRS Match: {int(an.qrs_correlation*100)}% (Aberrant/Wide)<br>"
                        f"Pattern: {an.pattern or 'Isolated'}"
                    )
                elif an.anomaly_type == "PAUSE":
                    pause_x.append(t_sec)
                    pause_y.append(v_peak)
                    pause_txt.append(
                        f"<b>⏸️ Sinus Pause / Prolonged Interval</b><br>"
                        f"Beat #{beat_num} at {t_sec:.3f}s<br>"
                        f"Pause Duration: {an.rr_pre_ms:.0f} ms ({an.pre_ratio:.2f}x baseline)"
                    )
            else:
                normal_x.append(t_sec)
                normal_y.append(v_peak)
                normal_txt.append(f"<b>Normal Sinus Beat #{beat_num}</b><br>Time: {t_sec:.3f}s<br>HR: {hr_str}<br>RR: {rr_str}")

        # Normal beats marker trace (Scattergl/WebGL: needed once a 24h recording puts
        # ~100k+ beats on this trace - plain SVG go.Scatter would be unusably slow)
        if normal_x:
            fig_ecg.add_trace(go.Scattergl(
                x=normal_x,
                y=normal_y,
                mode="markers",
                name="Normal Sinus Beats",
                marker=dict(symbol="circle", size=6, color=accent_green, opacity=0.7),
                text=normal_txt,
                hovertemplate="%{text}<extra></extra>"
            ))

        # PAC markers (hover-only label: an always-on text label per beat is fine at a
        # handful of events, but reads as solid clutter once a 24h recording surfaces
        # hundreds/thousands of them zoomed out - use "Inspect Beat" + hover instead)
        if pac_x:
            fig_ecg.add_trace(go.Scattergl(
                x=pac_x,
                y=pac_y,
                mode="markers",
                name=f"PAC ({len(pac_x)} beats)",
                marker=dict(
                    symbol="diamond",
                    size=11,
                    color=accent_pac,
                    line=dict(width=1.5, color="#ffffff")
                ),
                customdata=pac_txt,
                hovertemplate="%{customdata}<extra></extra>"
            ))

        # PVC markers
        if pvc_x:
            fig_ecg.add_trace(go.Scattergl(
                x=pvc_x,
                y=pvc_y,
                mode="markers",
                name=f"PVC ({len(pvc_x)} beats)",
                marker=dict(
                    symbol="triangle-up",
                    size=13,
                    color=accent_pvc,
                    line=dict(width=1.5, color="#ffffff")
                ),
                customdata=pvc_txt,
                hovertemplate="%{customdata}<extra></extra>"
            ))

        # Sinus Pause markers
        if pause_x:
            fig_ecg.add_trace(go.Scattergl(
                x=pause_x,
                y=pause_y,
                mode="markers",
                name=f"Sinus Pause ({len(pause_x)} events)",
                marker=dict(symbol="square", size=10, color=accent_purple, line=dict(width=1.5, color="#ffffff")),
                customdata=pause_txt,
                hovertemplate="%{customdata}<extra></extra>"
            ))

        # Default view: the full recording span. This trace is peak-preserving-decimated
        # (see min_max_decimate above), so on a multi-hour recording it renders as a clean
        # amplitude envelope - useful for orientation, but it is NOT high enough resolution
        # to show individual beat morphology when zoomed in. That's what the Event Detail
        # Viewer (full-resolution, built from the undecimated arrays) is for - see below.
        fig_ecg.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            margin=dict(l=55, r=25, t=35, b=45),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                title="Time (seconds)",
                range=[0.0, float(a.time_seconds[-1])],
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                    bgcolor=bg_color,
                    bordercolor=border_color
                ),
                showgrid=True,
                gridcolor=grid_color
            ),
            yaxis=dict(
                title="Voltage (mV)",
                showgrid=True,
                gridcolor=grid_color,
                zerolinecolor=grid_color
            ),
            height=440
        )

        # -------------------------------------------------------------
        # 2. Heart Rate & RR Tachogram with Ectopic Highlights
        # -------------------------------------------------------------
        fig_hr = make_subplots(specs=[[{"secondary_y": True}]])
        hr_range, rr_range = None, None

        if len(a.instant_hr_bpm) > 0:
            hr_range = robust_axis_range(a.instant_hr_bpm)
            rr_range = robust_axis_range(a.rr_intervals_ms)
            beat_times = a.r_peak_times[1:]

            # Normal HR range band (60 - 100 bpm)
            fig_hr.add_hrect(
                y0=60, y1=100, fillcolor=accent_green, opacity=0.08,
                line_width=0, annotation_text="Normal Resting Range (60-100 BPM)",
                annotation_position="top left",
                secondary_y=False
            )

            # Peak-preserving decimation (same technique as the ECG strip) - a 24h
            # recording puts ~100k points on this line; undecimated SVG lines are both
            # slow to render and, at a full 24h zoom, visually just a dense smear.
            x_hr_plot, y_hr_plot = min_max_decimate(beat_times, a.instant_hr_bpm, max_points=20000)
            x_rr_plot, y_rr_plot = min_max_decimate(beat_times, a.rr_intervals_ms, max_points=20000)

            # Heart Rate line
            fig_hr.add_trace(go.Scattergl(
                x=x_hr_plot,
                y=y_hr_plot,
                mode="lines",
                name="Heart Rate (BPM)",
                line=dict(color=accent_red, width=1.8),
                opacity=0.85,
                hovertemplate="Time: %{x:.1f} s<br>HR: %{y:.1f} BPM<extra></extra>"
            ), secondary_y=False)

            # Highlight ectopic beats on tachogram
            ectopic_times = [an.time_seconds for an in a.anomalies if an.anomaly_type in ["PAC", "PVC"]]
            ectopic_hrs = []
            for t_ec in ectopic_times:
                closest_idx = int(np.argmin(np.abs(beat_times - t_ec)))
                ectopic_hrs.append(a.instant_hr_bpm[closest_idx])

            if ectopic_times:
                fig_hr.add_trace(go.Scattergl(
                    x=ectopic_times,
                    y=ectopic_hrs,
                    mode="markers",
                    name="Ectopic Beats",
                    marker=dict(symbol="x", size=8, color=accent_pac, line=dict(width=1.5)),
                    hovertemplate="Ectopic Beat at %{x:.2f}s<br>Instant HR: %{y:.1f} BPM<extra></extra>"
                ), secondary_y=False)

            # RR Interval line
            fig_hr.add_trace(go.Scattergl(
                x=x_rr_plot,
                y=y_rr_plot,
                mode="lines",
                name="RR Interval (ms)",
                line=dict(color=accent_purple, width=1.5, dash="dot"),
                opacity=0.8,
                hovertemplate="Time: %{x:.1f} s<br>RR: %{y:.1f} ms<extra></extra>"
            ), secondary_y=True)

        fig_hr.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            margin=dict(l=55, r=55, t=35, b=45),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                title="Time (seconds)",
                showgrid=True,
                gridcolor=grid_color,
                rangeslider=dict(visible=True, thickness=0.08, bgcolor=bg_color, bordercolor=border_color)
            ),
            yaxis=dict(title="Heart Rate (BPM)", showgrid=True, gridcolor=grid_color, range=hr_range),
            yaxis2=dict(title="RR Interval (ms)", showgrid=False, range=rr_range),
            height=360
        )

        # -------------------------------------------------------------
        # 2b. Windowed (short-term) HRV Trend - day/night pattern on long recordings
        # -------------------------------------------------------------
        fig_trend = None
        if a.windowed_hrv and len(a.windowed_hrv) > 1:
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            w_mid = [(w.start_s + w.end_s) / 2.0 for w in a.windowed_hrv]
            w_hr = [w.mean_hr_bpm for w in a.windowed_hrv]
            w_sdnn = [w.sdnn_ms for w in a.windowed_hrv]
            w_rmssd = [w.rmssd_ms for w in a.windowed_hrv]
            w_text = [f"Window {w.start_s/60:.1f}-{w.end_s/60:.1f} min<br>Beats: {w.n_beats}<br>PACs: {w.pac_count} | PVCs: {w.pvc_count}<br>Quality: {w.quality_pct:.0f}%" for w in a.windowed_hrv]

            fig_trend.add_trace(go.Scatter(
                x=w_mid, y=w_hr, mode="lines+markers", name="Mean HR (BPM)",
                line=dict(color=accent_red, width=1.8), marker=dict(size=5),
                text=w_text, hovertemplate="%{text}<br>Mean HR: %{y:.1f} BPM<extra></extra>"
            ), secondary_y=False)
            fig_trend.add_trace(go.Scatter(
                x=w_mid, y=w_sdnn, mode="lines+markers", name="SDNN (ms)",
                line=dict(color=accent_blue, width=1.5, dash="dot"), marker=dict(size=4),
                hovertemplate="SDNN: %{y:.1f} ms<extra></extra>"
            ), secondary_y=True)
            trend_hr_range = robust_axis_range(w_hr)
            trend_hrv_range = robust_axis_range(list(w_sdnn) + list(w_rmssd))

            fig_trend.add_trace(go.Scatter(
                x=w_mid, y=w_rmssd, mode="lines+markers", name="RMSSD (ms)",
                line=dict(color=accent_green, width=1.5, dash="dot"), marker=dict(size=4),
                hovertemplate="RMSSD: %{y:.1f} ms<extra></extra>"
            ), secondary_y=True)

            fig_trend.update_layout(
                template="plotly_dark" if dark_mode else "plotly_white",
                paper_bgcolor=card_bg,
                plot_bgcolor=card_bg,
                margin=dict(l=55, r=55, t=35, b=45),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor=grid_color),
                yaxis=dict(title="Heart Rate (BPM)", showgrid=True, gridcolor=grid_color, range=trend_hr_range),
                yaxis2=dict(title="HRV (ms)", showgrid=False, range=trend_hrv_range),
                height=320
            )

        # -------------------------------------------------------------
        # 3. Average Beat Template with Ectopic Overlay
        # -------------------------------------------------------------
        fig_template = go.Figure()
        if len(a.average_beat_mv) > 0:
            t_ms = a.average_beat_time_ms
            mean_mv = a.average_beat_mv
            std_mv = a.average_beat_std_mv

            # Background normal individual beats
            for beat in a.all_beat_waves[:20]:
                fig_template.add_trace(go.Scatter(
                    x=t_ms,
                    y=beat,
                    mode="lines",
                    line=dict(color="#64748b", width=0.8),
                    opacity=0.18,
                    showlegend=False,
                    hoverinfo="skip"
                ))

            # Confidence envelope (Mean ± 1 SD)
            fig_template.add_trace(go.Scatter(
                x=np.concatenate([t_ms, t_ms[::-1]]),
                y=np.concatenate([mean_mv + std_mv, (mean_mv - std_mv)[::-1]]),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Normal Sinus ±1 SD",
                showlegend=True,
                hoverinfo="skip"
            ))

            # Mean normal sinus waveform
            fig_template.add_trace(go.Scatter(
                x=t_ms,
                y=mean_mv,
                mode="lines",
                name="Normal Sinus Template",
                line=dict(color=accent_blue, width=2.8),
                hovertemplate="Offset: %{x:.1f} ms<br>Amplitude: %{y:.3f} mV<extra></extra>"
            ))

            # Superimpose sample ectopic beats (PAC / PVC) for visual morphology comparison
            fs = a.metadata.sampling_rate_hz
            w_pre_s = int(fs * 0.20)
            w_post_s = int(fs * 0.40)
            
            ectopic_anomalies = [an for an in a.anomalies if an.anomaly_type in ["PAC", "PVC"]]
            for idx_e, an in enumerate(ectopic_anomalies[:4]):
                p = an.sample_index
                if p - w_pre_s >= 0 and p + w_post_s <= len(filt_mv):
                    seg = filt_mv[p - w_pre_s : p + w_post_s]
                    seg = seg - np.mean(seg[: max(1, int(0.03 * fs))])
                    color_e = accent_pvc if an.anomaly_type == "PVC" else accent_pac
                    fig_template.add_trace(go.Scatter(
                        x=t_ms,
                        y=seg,
                        mode="lines",
                        name=f"{an.anomaly_type} #{an.beat_index} ({an.time_seconds:.1f}s)",
                        line=dict(color=color_e, width=1.8, dash="dot"),
                        hovertemplate=f"<b>{an.anomaly_type} #{an.beat_index}</b><br>Offset: %{{x:.1f}} ms<br>Amplitude: %{{y:.3f}} mV<extra></extra>"
                    ))

        fig_template.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            margin=dict(l=45, r=20, t=35, b=45),
            xaxis=dict(title="Time relative to R-peak (ms)", showgrid=True, gridcolor=grid_color),
            yaxis=dict(title="Voltage (mV)", showgrid=True, gridcolor=grid_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320
        )

        # -------------------------------------------------------------
        # 4. Poincaré Plot (RR_n vs RR_n+1)
        # -------------------------------------------------------------
        fig_poincare = go.Figure()
        if len(a.rr_intervals_ms) > 2:
            rr_n = a.rr_intervals_ms[:-1]
            rr_n1 = a.rr_intervals_ms[1:]
            
            min_lim = max(300, min(np.min(rr_n), np.min(rr_n1)) - 50)
            max_lim = min(1500, max(np.max(rr_n), np.max(rr_n1)) + 50)
            
            fig_poincare.add_trace(go.Scatter(
                x=[min_lim, max_lim],
                y=[min_lim, max_lim],
                mode="lines",
                line=dict(color="#64748b", dash="dash", width=1.5),
                name="Identity Line (y = x)",
                hoverinfo="skip"
            ))

            fig_poincare.add_trace(go.Scattergl(
                x=rr_n,
                y=rr_n1,
                mode="markers",
                name="RR Intervals",
                marker=dict(
                    color=accent_amber,
                    size=6,
                    opacity=0.75,
                    line=dict(width=0.5, color="#ffffff")
                ),
                hovertemplate="RR[n]: %{x:.1f} ms<br>RR[n+1]: %{y:.1f} ms<extra></extra>"
            ))

            if h.sd1_ms > 0 and h.sd2_ms > 0:
                mean_rr = h.mean_rr_ms
                theta = np.linspace(0, 2 * np.pi, 100)
                a_axis = h.sd2_ms * 2.0
                b_axis = h.sd1_ms * 2.0
                
                cos_45 = np.cos(np.pi / 4)
                sin_45 = np.sin(np.pi / 4)
                x_el = a_axis * np.cos(theta)
                y_el = b_axis * np.sin(theta)
                
                x_rot = mean_rr + (x_el * cos_45 - y_el * sin_45)
                y_rot = mean_rr + (x_el * sin_45 + y_el * cos_45)

                fig_poincare.add_trace(go.Scatter(
                    x=x_rot,
                    y=y_rot,
                    mode="lines",
                    name=f"HRV Ellipse (SD1={h.sd1_ms}ms, SD2={h.sd2_ms}ms)",
                    line=dict(color=accent_green, width=2),
                    hoverinfo="skip"
                ))

        fig_poincare.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            margin=dict(l=45, r=20, t=35, b=45),
            xaxis=dict(title="RR[n] (ms)", showgrid=True, gridcolor=grid_color),
            yaxis=dict(title="RR[n+1] (ms)", showgrid=True, gridcolor=grid_color, scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320
        )

        # -------------------------------------------------------------
        # 5. Frequency Domain Welch PSD
        # -------------------------------------------------------------
        fig_psd = go.Figure()
        if len(a.psd_freqs) > 0:
            freqs = a.psd_freqs
            psd = a.psd_values

            valid_freq = freqs <= 0.45
            f_sub = freqs[valid_freq]
            p_sub = psd[valid_freq]

            if h.ulf_power_ms2:
                ulf_m = f_sub < 0.0033
                if np.any(ulf_m):
                    fig_psd.add_trace(go.Scatter(
                        x=f_sub[ulf_m], y=p_sub[ulf_m], fill="tozeroy",
                        fillcolor="rgba(100, 116, 139, 0.25)",
                        line=dict(color="#64748b", width=1),
                        name=f"ULF ({h.ulf_power_ms2:.0f} ms²)"
                    ))

            vlf_m = (f_sub >= 0.0033) & (f_sub < 0.04)
            if np.any(vlf_m):
                fig_psd.add_trace(go.Scatter(
                    x=f_sub[vlf_m], y=p_sub[vlf_m], fill="tozeroy",
                    fillcolor="rgba(148, 163, 184, 0.25)",
                    line=dict(color="#94a3b8", width=1),
                    name=f"VLF ({h.vlf_power_ms2:.0f} ms²)"
                ))

            lf_m = (f_sub >= 0.04) & (f_sub < 0.15)
            if np.any(lf_m):
                fig_psd.add_trace(go.Scatter(
                    x=f_sub[lf_m], y=p_sub[lf_m], fill="tozeroy",
                    fillcolor="rgba(245, 158, 11, 0.35)",
                    line=dict(color=accent_amber, width=1.5),
                    name=f"LF ({h.lf_power_ms2:.0f} ms²)"
                ))

            hf_m = (f_sub >= 0.15) & (f_sub <= 0.40)
            if np.any(hf_m):
                fig_psd.add_trace(go.Scatter(
                    x=f_sub[hf_m], y=p_sub[hf_m], fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.35)",
                    line=dict(color=accent_green, width=1.5),
                    name=f"HF ({h.hf_power_ms2:.0f} ms²)"
                ))

            fig_psd.add_trace(go.Scatter(
                x=f_sub, y=p_sub,
                mode="lines",
                line=dict(color=text_color, width=2),
                name="Power Spectral Density",
                hovertemplate="Freq: %{x:.3f} Hz<br>Power: %{y:.1f} ms²/Hz<extra></extra>"
            ))

        fig_psd.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            margin=dict(l=45, r=20, t=35, b=45),
            xaxis=dict(title="Frequency (Hz)", showgrid=True, gridcolor=grid_color),
            yaxis=dict(title="Power Density (ms²/Hz)", showgrid=True, gridcolor=grid_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320
        )

        # -------------------------------------------------------------
        # 5b. Event Detail Viewer data - full-resolution (undecimated) waveform slices
        # around each anomaly, for the click-through detail chart. Built from the raw
        # a.time_seconds/filtered_voltages arrays (not the decimated plot traces above),
        # so it shows true beat morphology no matter how long the full recording is.
        # -------------------------------------------------------------
        detail_half_window_s = max(2.0, initial_window_seconds / 2.0)
        marker_styles_json = json.dumps({
            "NORMAL": {"symbol": "circle", "size": 7, "color": accent_green},
            "PAC": {"symbol": "diamond", "size": 12, "color": accent_pac},
            "PVC": {"symbol": "triangle-up", "size": 14, "color": accent_pvc},
            "PAUSE": {"symbol": "square", "size": 11, "color": accent_purple},
        })

        def build_event_detail(t_center: float, event_type: str, label: str) -> dict:
            lo = t_center - detail_half_window_s
            hi = t_center + detail_half_window_s
            i0 = int(np.searchsorted(a.time_seconds, lo))
            i1 = int(np.searchsorted(a.time_seconds, hi))
            x_slice = (a.time_seconds[i0:i1] - t_center).round(4).tolist()
            yf_slice = filt_mv[i0:i1].round(3).tolist()

            markers = []
            for i, r_idx in enumerate(a.r_peaks_idx):
                beat_t = a.r_peak_times[i]
                if beat_t < lo or beat_t > hi:
                    continue
                beat_num = i + 1
                an = anomaly_map.get(beat_num)
                b_type = an.anomaly_type if an else "NORMAL"
                markers.append({
                    "x": round(beat_t - t_center, 4),
                    "y": round(float(filt_mv[r_idx]), 3),
                    "type": b_type,
                    "text": (an.description if an else f"Normal Sinus Beat #{beat_num}"),
                })

            return {
                "t": round(t_center, 3),
                "type": event_type,
                "label": label,
                "x": x_slice,
                "yf": yf_slice,
                "markers": markers,
            }

        event_details = [
            build_event_detail(an.time_seconds, an.anomaly_type, an.description)
            for an in a.anomalies
        ]
        if not event_details:
            # No anomalies: default the detail viewer to an early window of normal rhythm
            fallback_t = a.r_peak_times[min(5, len(a.r_peak_times) - 1)] if len(a.r_peak_times) else detail_half_window_s
            event_details = [build_event_detail(fallback_t, "NORMAL", "Normal sinus rhythm (no anomalies detected)")]

        event_details_json = json.dumps(event_details)

        # -------------------------------------------------------------
        # 6. HTML Tables and Log Rows for Arrhythmias
        # -------------------------------------------------------------
        table_rows_html = ""
        if a.anomalies:
            for idx, an in enumerate(a.anomalies):
                badge_class = "badge-pac" if an.anomaly_type == "PAC" else ("badge-pvc" if an.anomaly_type == "PVC" else "badge-pause")
                pattern_badge = f"<span class='badge-pat'>{an.pattern}</span>" if an.pattern else "<span style='color:var(--text-muted);'>Isolated</span>"

                table_rows_html += f"""
                <tr class="anomaly-row" data-type="{an.anomaly_type}">
                    <td style="font-weight:700;">#{an.beat_index}</td>
                    <td style="font-family:'JetBrains Mono',monospace;">{an.time_seconds:.2f}s</td>
                    <td><span class="badge {badge_class}">{an.anomaly_type}</span></td>
                    <td>{pattern_badge}</td>
                    <td style="font-family:'JetBrains Mono',monospace;">{an.rr_pre_ms:.0f} ms <span style="color:var(--text-muted); font-size:11px;">({an.pre_ratio:.2f}x)</span></td>
                    <td style="font-family:'JetBrains Mono',monospace;">{an.rr_post_ms:.0f} ms <span style="color:var(--text-muted); font-size:11px;">({an.post_ratio:.2f}x)</span></td>
                    <td style="font-family:'JetBrains Mono',monospace;">{int(an.qrs_correlation * 100)}%</td>
                    <td style="font-size:12px; color:var(--text-muted);">{an.description}</td>
                    <td>
                        <button class="btn-jump" onclick="jumpToTime({an.time_seconds}, {idx})">🔍 Inspect Beat</button>
                    </td>
                </tr>
                """
        else:
            table_rows_html = """
            <tr>
                <td colspan="9" style="text-align:center; padding:20px; color:var(--accent-green); font-weight:600;">
                    ✓ No significant ectopic beats, pauses, or arrhythmias detected. Normal sinus rhythm throughout.
                </td>
            </tr>
            """

        # Convert plots to HTML
        html_ecg = fig_ecg.to_html(full_html=False, include_plotlyjs="cdn")
        html_hr = fig_hr.to_html(full_html=False, include_plotlyjs=False)

        trend_section_html = ""
        if fig_trend is not None:
            html_trend = fig_trend.to_html(full_html=False, include_plotlyjs=False)
            trend_section_html = f"""
        <div class="chart-card">
            <div class="chart-header">
                <div>
                    <h2>Short-Term HRV Trend ({len(a.windowed_hrv)} x 5-min windows)</h2>
                    <span class="chart-desc">Mean HR, SDNN and RMSSD per short-term window across the full recording - the day/night autonomic pattern a single-window analysis would blur together.</span>
                </div>
            </div>
            {html_trend}
        </div>
"""
        html_template = fig_template.to_html(full_html=False, include_plotlyjs=False)
        html_poincare = fig_poincare.to_html(full_html=False, include_plotlyjs=False)
        html_psd = fig_psd.to_html(full_html=False, include_plotlyjs=False)

        # Full HTML page
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polar ECG & Arrhythmia Analysis - {m.recording_name}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-body: {bg_color};
            --bg-card: {card_bg};
            --text-primary: {text_color};
            --text-muted: {text_muted};
            --border: {border_color};
            --accent-blue: {accent_blue};
            --accent-red: {accent_red};
            --accent-green: {accent_green};
            --accent-purple: {accent_purple};
            --accent-amber: {accent_amber};
            --accent-pac: {accent_pac};
            --accent-pvc: {accent_pvc};
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background-color: var(--bg-body);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header-title h1 {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .badge {{
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
        }}
        .badge-device {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .badge-pac {{
            background: rgba(249, 115, 22, 0.2);
            color: var(--accent-pac);
            border: 1px solid var(--accent-pac);
        }}
        .badge-pvc {{
            background: rgba(239, 68, 68, 0.25);
            color: var(--accent-pvc);
            border: 1px solid var(--accent-pvc);
        }}
        .badge-pause {{
            background: rgba(168, 85, 247, 0.2);
            color: var(--accent-purple);
            border: 1px solid var(--accent-purple);
        }}
        .badge-pat {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--accent-amber);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
        }}
        .header-meta {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 6px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .header-meta span strong {{
            color: var(--text-primary);
        }}
        .btn-group {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        .btn:hover {{
            background: var(--border);
            border-color: var(--text-muted);
        }}
        .btn-jump {{
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: var(--accent-blue);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .btn-jump:hover {{
            background: var(--accent-blue);
            color: #0f172a;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }}
        .metric-card.hr::before {{ background: var(--accent-red); }}
        .metric-card.pac::before {{ background: var(--accent-pac); }}
        .metric-card.pvc::before {{ background: var(--accent-pvc); }}
        .metric-card.rmssd::before {{ background: var(--accent-green); }}
        .metric-card.sdnn::before {{ background: var(--accent-blue); }}
        .metric-card.lfhf::before {{ background: var(--accent-amber); }}
        .metric-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.02em;
            display: flex;
            align-items: baseline;
            gap: 6px;
        }}
        .metric-unit {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
        }}
        .metric-sub {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .chart-header h2 {{
            font-size: 16px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .chart-desc {{
            font-size: 12px;
            color: var(--text-muted);
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 900px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
        .table-responsive {{
            width: 100%;
            overflow-x: auto;
            margin-top: 12px;
        }}
        table.anomaly-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        table.anomaly-table th {{
            background: rgba(0, 0, 0, 0.2);
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}
        table.anomaly-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        table.anomaly-table tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        .table-controls select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
        }}
        .table-pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 16px;
            margin-top: 14px;
            font-size: 13px;
            color: var(--text-muted);
        }}
        footer {{
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>
                    <span>Polar ECG & Arrhythmia Inspector</span>
                    <span class="badge badge-device">Polar H10</span>
                </h1>
                <div class="header-meta">
                    <span>File: <strong>{m.recording_name}</strong></span>
                    <span>Device: <strong>{m.device_id}</strong></span>
                    <span>Start: <strong>{m.start_iso_time or 'N/A'}</strong></span>
                    <span>Duration: <strong>{m.duration_seconds:.1f}s ({m.duration_seconds/60:.2f} min)</strong></span>
                    <span>Rate: <strong>{m.sampling_rate_hz} Hz</strong></span>
                </div>
            </div>
            <div class="btn-group">
                <button class="btn" onclick="window.print()">Print / Export PDF</button>
            </div>
        </header>

        <!-- Metric Cards -->
        <div class="metrics-grid">
            <div class="metric-card hr">
                <div class="metric-label">Mean Heart Rate</div>
                <div class="metric-value">{h.mean_hr_bpm:.1f} <span class="metric-unit">BPM</span></div>
                <div class="metric-sub">Range: {h.min_hr_bpm:.1f} - {h.max_hr_bpm:.1f} BPM (σ={h.std_hr_bpm:.1f})</div>
            </div>
            <div class="metric-card pac">
                <div class="metric-label">PAC Count (Atrial)</div>
                <div class="metric-value">{h.pac_count} <span class="metric-unit">beats</span></div>
                <div class="metric-sub">Supraventricular ectopy ({h.pac_count / max(1, h.total_beats) * 100:.1f}% burden)</div>
            </div>
            <div class="metric-card pvc">
                <div class="metric-label">PVC Count (Ventricular)</div>
                <div class="metric-value">{h.pvc_count} <span class="metric-unit">beats</span></div>
                <div class="metric-sub">Ventricular ectopy ({h.pvc_count / max(1, h.total_beats) * 100:.1f}% burden)</div>
            </div>
            <div class="metric-card rmssd">
                <div class="metric-label">RMSSD (Vagal Tone)</div>
                <div class="metric-value">{h.rmssd_ms:.1f} <span class="metric-unit">ms</span></div>
                <div class="metric-sub">Parasympathetic benchmark (>42ms high)</div>
            </div>
            <div class="metric-card sdnn">
                <div class="metric-label">SDNN (Total HRV)</div>
                <div class="metric-value">{h.sdnn_ms:.1f} <span class="metric-unit">ms</span></div>
                <div class="metric-sub">pNN50: {h.pnn50_pct:.1f}% | Total power: {h.total_power_ms2:.0f} ms²</div>
            </div>
            <div class="metric-card lfhf">
                <div class="metric-label">LF / HF Ratio</div>
                <div class="metric-value">{h.lf_hf_ratio:.2f} <span class="metric-unit">ratio</span></div>
                <div class="metric-sub">Autonomic balance (Pauses: {h.pause_count})</div>
            </div>
            <div class="metric-card sdnn">
                <div class="metric-label">Respiration Rate (EDR)</div>
                <div class="metric-value">{resp_rate_display} <span class="metric-unit">breaths/min</span></div>
                <div class="metric-sub">ECG-derived, R-amplitude modulation</div>
            </div>
            <div class="metric-card rmssd">
                <div class="metric-label">DFA alpha1</div>
                <div class="metric-value">{h.dfa_alpha1:.3f} <span class="metric-unit">{dfa_alpha2_display}</span></div>
                <div class="metric-sub">Short-term fractal scaling exponent</div>
            </div>
        </div>

        <!-- 1. Interactive ECG Strip -->
        <div class="chart-card" id="ecg-strip-card">
            <div class="chart-header">
                <div>
                    <h2>ECG Voltage Strip & Anomaly Highlighting</h2>
                    <span class="chart-desc">Orange markers = PACs | Red markers = PVCs | Green dots = Normal sinus beats. Drag the range slider below or click an anomaly in the table to jump directly to it.</span>
                </div>
            </div>
            <div id="ecg-plot-container">
                {html_ecg}
            </div>
        </div>

        <!-- 1b. Event Detail Viewer: full-resolution, click-through per event -->
        <div class="chart-card" id="event-detail-card">
            <div class="chart-header">
                <div>
                    <h2>Event Detail Viewer (Full Resolution)</h2>
                    <span class="chart-desc" id="event-detail-caption">Loading...</span>
                </div>
                <div class="table-controls">
                    <button class="btn" onclick="eventDetailPrev()">‹ Prev Event</button>
                    <button class="btn" onclick="eventDetailNext()">Next Event ›</button>
                </div>
            </div>
            <div id="event-detail-plot" style="height:380px;"></div>
        </div>

        <!-- 2. Arrhythmia & Ectopic Event Log Table -->
        <div class="chart-card">
            <div class="chart-header">
                <div>
                    <h2>Detected Arrhythmia & Ectopic Beats Log ({len(a.anomalies)} events)</h2>
                    <span class="chart-desc">Click "Inspect Beat" to load that exact beat, at full resolution, into the Event Detail Viewer above. Paginated client-side so a long (e.g. 24h) event log stays scrollable rather than dumping thousands of rows at once.</span>
                </div>
                <div class="table-controls">
                    <select id="anomaly-filter" onchange="anomalyTableUpdate()">
                        <option value="ALL">All types ({len(a.anomalies)})</option>
                        <option value="PAC">PAC only ({h.pac_count})</option>
                        <option value="PVC">PVC only ({h.pvc_count})</option>
                        <option value="PAUSE">Pause only ({h.pause_count})</option>
                    </select>
                </div>
            </div>
            <div class="table-responsive">
                <table class="anomaly-table" id="anomaly-table">
                    <thead>
                        <tr>
                            <th>Beat #</th>
                            <th>Time</th>
                            <th>Category</th>
                            <th>Pattern</th>
                            <th>Coupling (RR Pre)</th>
                            <th>Pause (RR Post)</th>
                            <th>QRS Match</th>
                            <th>Clinical Finding</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
            <div class="table-pagination">
                <button class="btn" onclick="anomalyPagePrev()">‹ Prev</button>
                <span id="anomaly-page-label">Page 1</span>
                <button class="btn" onclick="anomalyPageNext()">Next ›</button>
            </div>
        </div>

        <!-- 3. Heart Rate & RR Tachogram -->
        <div class="chart-card">
            <div class="chart-header">
                <div>
                    <h2>Heart Rate & RR Interval Tachogram</h2>
                    <span class="chart-desc">Instantaneous beat-by-beat heart rate with shaded normal resting range (60–100 BPM) and marked ectopic beats.</span>
                </div>
            </div>
            {html_hr}
        </div>

        <!-- 2b. Windowed HRV Trend (only rendered when windowed_hrv was computed) -->
        {trend_section_html}

        <!-- 4 & 5. Average Beat & Poincaré -->
        <div class="charts-row">
            <div class="chart-card" style="margin-bottom: 0;">
                <div class="chart-header">
                    <div>
                        <h2>Beat Morphology & Ectopic Overlay</h2>
                        <span class="chart-desc">Canonical Normal Sinus template (blue) vs Superimposed Ectopic Beats (dotted lines).</span>
                    </div>
                </div>
                {html_template}
            </div>
            <div class="chart-card" style="margin-bottom: 0;">
                <div class="chart-header">
                    <div>
                        <h2>Poincaré Plot (RRₙ vs RRₙ₊₁)</h2>
                        <span class="chart-desc">Non-linear dynamics: SD1={h.sd1_ms}ms (short-term), SD2={h.sd2_ms}ms (long-term).</span>
                    </div>
                </div>
                {html_poincare}
            </div>
        </div>

        <!-- 6. Frequency Domain Welch PSD -->
        <div class="chart-card" style="margin-top: 24px;">
            <div class="chart-header">
                <div>
                    <h2>HRV Power Spectral Density (Welch PSD)</h2>
                    <span class="chart-desc">VLF (0.0033-0.04 Hz): {h.vlf_power_ms2:.0f} ms² | LF (0.04-0.15 Hz): {h.lf_power_ms2:.0f} ms² | HF (0.15-0.40 Hz): {h.hf_power_ms2:.0f} ms²</span>
                </div>
            </div>
            {html_psd}
        </div>

        <footer>
            Polar H10 ECG Analyzer & Anomaly Inspector • Processed {m.total_samples} samples ({m.duration_seconds:.1f}s)
        </footer>
    </div>

    <script>
        // Full-resolution (undecimated) waveform slice for each detected event, plus a
        // style lookup for beat-type markers. Built server-side from the raw sample
        // arrays so this stays crisp regardless of total recording length.
        const EVENT_DETAILS = {event_details_json};
        const MARKER_STYLES = {marker_styles_json};
        let currentEventIdx = 0;

        function renderEventDetail(idx) {{
            if (!EVENT_DETAILS.length) return;
            currentEventIdx = ((idx % EVENT_DETAILS.length) + EVENT_DETAILS.length) % EVENT_DETAILS.length;
            const ev = EVENT_DETAILS[currentEventIdx];

            const traces = [{{
                x: ev.x,
                y: ev.yf,
                mode: 'lines',
                name: 'Filtered ECG',
                line: {{ color: '{accent_blue}', width: 1.8 }},
                hovertemplate: 'Δt: %{{x:.3f}}s<br>%{{y:.3f}} mV<extra></extra>'
            }}];

            const byType = {{}};
            ev.markers.forEach(mk => {{
                (byType[mk.type] = byType[mk.type] || []).push(mk);
            }});
            Object.keys(byType).forEach(type => {{
                const pts = byType[type];
                const style = MARKER_STYLES[type] || MARKER_STYLES.NORMAL;
                traces.push({{
                    x: pts.map(p => p.x),
                    y: pts.map(p => p.y),
                    mode: 'markers',
                    name: type,
                    marker: {{ symbol: style.symbol, size: style.size, color: style.color, line: {{ width: 1.5, color: '#ffffff' }} }},
                    customdata: pts.map(p => p.text),
                    hovertemplate: '%{{customdata}}<extra></extra>'
                }});
            }});

            const layout = {{
                template: '{"plotly_dark" if dark_mode else "plotly_white"}',
                paper_bgcolor: '{card_bg}',
                plot_bgcolor: '{card_bg}',
                margin: {{ l: 55, r: 25, t: 20, b: 45 }},
                hovermode: 'closest',
                showlegend: false,
                xaxis: {{ title: 'Time relative to event (s)', showgrid: true, gridcolor: '{grid_color}', zeroline: true, zerolinecolor: '{grid_color}' }},
                yaxis: {{ title: 'Voltage (mV)', showgrid: true, gridcolor: '{grid_color}' }}
            }};

            Plotly.react('event-detail-plot', traces, layout, {{ responsive: true }});

            const caption = document.getElementById('event-detail-caption');
            if (caption) {{
                caption.textContent = `Event ${{currentEventIdx + 1}} of ${{EVENT_DETAILS.length}}: ${{ev.type}} at ${{ev.t.toFixed(2)}}s — ${{ev.label}}`;
            }}
        }}

        function eventDetailPrev() {{ renderEventDetail(currentEventIdx - 1); }}
        function eventDetailNext() {{ renderEventDetail(currentEventIdx + 1); }}

        function jumpToTime(timeSec, idx) {{
            const container = document.getElementById('ecg-plot-container');
            const plotDiv = container ? container.querySelector('.js-plotly-plot') : null;
            if (plotDiv) {{
                Plotly.relayout(plotDiv, {{
                    'xaxis.range': [timeSec - 2.5, timeSec + 2.5]
                }});
            }}
            if (typeof idx === 'number') {{
                renderEventDetail(idx);
                const card = document.getElementById('event-detail-card');
                if (card) {{
                    card.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}
        }}

        // Client-side pagination + type filter for the anomaly log table, so a long
        // (e.g. 24h) event list stays a fixed-size table instead of dumping every row.
        const ANOMALY_PAGE_SIZE = 100;
        let anomalyPage = 0;

        function anomalyFilteredRows() {{
            const filter = document.getElementById('anomaly-filter').value;
            const rows = Array.from(document.querySelectorAll('#anomaly-table tbody tr.anomaly-row'));
            return filter === 'ALL' ? rows : rows.filter(r => r.dataset.type === filter);
        }}

        function anomalyRenderPage() {{
            const allRows = Array.from(document.querySelectorAll('#anomaly-table tbody tr.anomaly-row'));
            const filtered = anomalyFilteredRows();
            const totalPages = Math.max(1, Math.ceil(filtered.length / ANOMALY_PAGE_SIZE));
            anomalyPage = Math.min(Math.max(0, anomalyPage), totalPages - 1);

            allRows.forEach(r => r.style.display = 'none');
            const start = anomalyPage * ANOMALY_PAGE_SIZE;
            filtered.slice(start, start + ANOMALY_PAGE_SIZE).forEach(r => r.style.display = '');

            const label = document.getElementById('anomaly-page-label');
            if (label) {{
                label.textContent = `Page ${{anomalyPage + 1}} of ${{totalPages}} (${{filtered.length}} matching)`;
            }}
        }}

        function anomalyTableUpdate() {{ anomalyPage = 0; anomalyRenderPage(); }}
        function anomalyPagePrev() {{ anomalyPage -= 1; anomalyRenderPage(); }}
        function anomalyPageNext() {{ anomalyPage += 1; anomalyRenderPage(); }}

        document.addEventListener('DOMContentLoaded', () => {{
            anomalyRenderPage();
            renderEventDetail(0);
        }});
    </script>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(full_html)

        return output_filepath
