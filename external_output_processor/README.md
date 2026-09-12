# Polar H10 ECG Analyzer & Interactive Visualizer

A clinical and sports-science grade Python toolkit for analyzing, plotting, and reporting on Polar H10 (and compatible) ECG `.jsonl` recordings.

![ECG Analysis](https://img.shields.io/badge/Signal-ECG_130Hz-brightgreen)
![HRV](https://img.shields.io/badge/HRV-Time_%26_Frequency_Domain-blue)
![Format](https://img.shields.io/badge/Format-JSONL-orange)

---

## ⚡ Quick Start

### 1. Run Analysis and Open Interactive Dashboard
```bash
python3 ecg_tool.py ECG.jsonl
```
This will:
- Parse and filter the raw 130 Hz ECG signal (removing baseline wander & 50Hz mains hum).
- Detect QRS complexes and R-peaks with millisecond precision.
- Compute complete **Time-Domain**, **Frequency-Domain (Welch PSD)**, and **Non-Linear (Poincaré)** HRV metrics.
- Print a formatted summary table in your terminal.
- Generate and automatically open `ECG_report.html` in your default web browser.

### 2. Export RR Intervals to CSV
```bash
python3 ecg_tool.py ECG.jsonl --csv rr_intervals.csv
```

### 3. Launch Local Interactive Web App
```bash
python3 ecg_tool.py --serve
```
Opens a local web server (`http://localhost:8080`) that lists all `.jsonl` files in the directory and provides instant one-click analysis.

### 4. Batch Compare Multiple Recordings
```bash
python3 ecg_tool.py --batch *.jsonl
```

---

## 📊 Features & Visualizations

The generated standalone HTML report (`ECG_report.html`) contains an interactive suite:

1. **Interactive ECG Voltage Strip**:
   - High-performance WebGL time-series rendering (zoom and pan down to individual milliseconds).
   - Inverted-triangle markers at each detected R-peak with instantaneous HR hover tooltips.
   - Toggle between **Filtered ECG** and **Raw ECG** in the chart legend.
   - Range slider / minimap for scrubbing through minutes of recording.

2. **Heart Rate & RR Tachogram**:
   - Beat-by-beat instantaneous heart rate in BPM.
   - Normal resting zone (60–100 BPM) highlighted.
   - Secondary y-axis displaying RR interval in milliseconds.

3. **Ensemble Average Beat Waveform (P-QRS-T)**:
   - Canonical average beat aligned at the R-peak ($t = 0$ ms).
   - Shaded $\pm 1$ standard deviation confidence envelope.
   - Superimposed transparent background traces of individual heartbeats to assess morphological stability.

4. **Poincaré Plot ($RR_n$ vs $RR_{n+1}$)**:
   - Non-linear heart rate dynamics scatter plot.
   - Line of identity ($y = x$).
   - Fitted confidence ellipse displaying short-term ($SD_1$) and long-term ($SD_2$) autonomic axes.

5. **HRV Frequency Power Spectral Density (Welch PSD)**:
   - Color-shaded frequency bands:
     - **VLF** (0.0033 – 0.04 Hz): Very Low Frequency
     - **LF** (0.04 – 0.15 Hz): Sympathetic & Baroreflex modulation
     - **HF** (0.15 – 0.40 Hz): Parasympathetic / Vagal tone (respiratory sinus arrhythmia)
   - Total autonomic power & $LF/HF$ ratio.

---

## 🩺 Arrhythmia & Ectopic Beat Detection

The toolkit automatically detects and classifies cardiac rhythm anomalies beat-by-beat:

| Rhythm Finding | Marker | Electrocardiographic Criteria | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Premature Atrial Contraction (PAC)** | 🔶 **Orange Diamond** | Early coupling ($RR_{pre} \le 0.85 \times \text{median}$), **narrow normal QRS** ($r \ge 0.80$), followed by pause | Supraventricular ectopy; common & benign in healthy individuals |
| **Premature Ventricular Contraction (PVC)** | 🔴 **Red Triangle** | Early coupling ($RR_{pre} \le 0.85 \times \text{median}$), **wide/aberrant QRS** ($r < 0.80$), full compensatory pause | Ventricular ectopy; indicates ectopic ventricular focus |
| **Sinus Pause / Arrest** | 🟣 **Purple Square** | Prolonged interval ($RR > 1.5 \times \text{median}$ or $>1.8$ s) without preceding premature beat | SA nodal pause or dropped beat |
| **Couplets / Bigeminy** | 🏷️ **Badge** | Two consecutive ectopic beats (couplet) or alternating normal-ectopic (bigeminy) | Arrhythmia pattern classification |
| **Tachycardia / Bradycardia** | 📈 **Band** | Sustained runs ($\ge 3$ beats) of $>115$ BPM or $<55$ BPM | Rate abnormalities |

### 🔍 Interactive Ectopic Beat Inspector
- In the generated HTML dashboard, every anomaly is listed in the **Detected Arrhythmia & Ectopic Beats Log**.
- Clicking **"🔍 Inspect Beat"** on any row immediately scrolls and zooms the high-resolution ECG trace onto that exact beat's millisecond waveform.
- The **Beat Morphology & Ectopic Overlay** chart directly superimposes the ectopic beats over the normal sinus template for visual verification.

---

## 📈 Interpreting the Metrics

| Metric | Normal / Reference | Physiological Significance |
| :--- | :--- | :--- |
| **Mean HR** | 60 – 100 BPM | Resting cardiovascular rate |
| **SDNN** | $> 50$ ms | Overall Heart Rate Variability |
| **RMSSD** | $> 42$ ms | Primary indicator of parasympathetic (vagal) tone |
| **pNN50** | $> 3\%$ | Percentage of successive intervals differing by $>50$ ms |
| **LF / HF** | $0.5 – 2.0$ | Sympathovagal balance ($<1.0$ indicates parasympathetic dominance) |
| **Poincaré SD1** | Correlates with RMSSD | Fast beat-to-beat variability (vagal activation) |
| **Poincaré SD2** | Correlates with SDNN | Long-term continuous autonomic variation |
| **Signal Quality** | $> 95\%$ | Usable beat intervals without motion/contact artifact |

---

## 💻 Python API Usage

You can also import and use the pipeline directly in your own Python scripts:

```python
from ecg_processor import ECGAnalyzer
from ecg_visualizer import ECGVisualizer

# 1. Load and process
analyzer = ECGAnalyzer("ECG.jsonl")
analyzer.filter_signal(lowcut=0.5, highcut=40.0, notch_freq=50.0)
analyzer.detect_r_peaks()
hrv = analyzer.compute_hrv_and_morphology()

# 2. Access metrics
print(f"Mean HR: {hrv.mean_hr_bpm} BPM")
print(f"RMSSD:   {hrv.rmssd_ms} ms")
print(f"SDNN:    {hrv.sdnn_ms} ms")
print(f"LF/HF:   {hrv.lf_hf_ratio}")

# 3. Export CSV
analyzer.export_rr_csv("rr_intervals.csv")

# 4. Generate Interactive HTML Dashboard
vis = ECGVisualizer(analyzer)
vis.generate_html_report("my_report.html", dark_mode=True)
```

---

## 📁 Repository Structure

- [`ecg_processor.py`](file:///Users/frasergough/Desktop/ECG/ecg_processor.py): Core signal processing, zero-phase filtering, adaptive QRS peak detection, time/frequency/non-linear HRV computation.
- [`ecg_visualizer.py`](file:///Users/frasergough/Desktop/ECG/ecg_visualizer.py): Plotly dashboard generator producing interactive, standalone HTML reports.
- [`ecg_tool.py`](file:///Users/frasergough/Desktop/ECG/ecg_tool.py): Command-line tool with batch comparison and built-in web server.
- [`ECG_report.html`](file:///Users/frasergough/Desktop/ECG/ECG_report.html): Generated interactive HTML report for `ECG.jsonl`.
- [`rr_intervals.csv`](file:///Users/frasergough/Desktop/ECG/rr_intervals.csv): Processed beat-by-beat RR intervals and instant HR.
