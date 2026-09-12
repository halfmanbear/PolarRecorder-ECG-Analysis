#!/usr/bin/env python3
"""
Polar H10 ECG Analysis & Plotting CLI
Comprehensive tool to analyze, visualize, and report on Polar ECG jsonl recordings.

Usage:
    python3 ecg_tool.py ECG.jsonl                   # Analyze and open interactive report
    python3 ecg_tool.py ECG.jsonl --csv rr.csv      # Export beat RR intervals to CSV
    python3 ecg_tool.py ECG.jsonl --no-open         # Generate report without auto-opening
    python3 ecg_tool.py --batch *.jsonl             # Compare multiple recordings
    python3 ecg_tool.py --serve                     # Launch local interactive web viewer
"""

import os
import sys
import glob
import argparse
import webbrowser
import http.server
import socketserver
import urllib.parse
from typing import List, Optional, Sequence, Union

from ecg_processor import ECGAnalyzer, RICH_AVAILABLE
from ecg_visualizer import ECGVisualizer


def process_file(
    filepath: str,
    output_html: Optional[str] = None,
    export_csv: Optional[str] = None,
    trend_csv: Optional[str] = None,
    notch_freq: Union[float, Sequence[float]] = (50.0, 60.0),
    dark_mode: bool = True,
    auto_open: bool = True
) -> str:
    print(f"\n[+] Loading and analyzing: {filepath}")
    analyzer = ECGAnalyzer(filepath)
    analyzer.filter_signal(notch_freq=notch_freq)
    analyzer.detect_r_peaks()
    analyzer.compute_hrv_and_morphology()

    # Short-term (5-min) windowed HRV trend - only meaningful once the recording
    # spans several windows (e.g. a 24h capture); cheap to compute either way.
    if analyzer.metadata.duration_seconds > 600.0:
        analyzer.compute_windowed_hrv(window_minutes=5.0)

    analyzer.print_summary()

    if export_csv:
        analyzer.export_rr_csv(export_csv)
        print(f"[+] RR intervals exported to: {export_csv}")

    if trend_csv:
        analyzer.export_windowed_csv(trend_csv)
        print(f"[+] Windowed HRV trend exported to: {trend_csv}")

    if not output_html:
        if isinstance(filepath, (list, tuple)):
            base_name = os.path.splitext(os.path.basename(str(filepath[0])))[0] + f"_stitched_{len(filepath)}parts"
        else:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
        output_html = f"{base_name}_report.html"

    vis = ECGVisualizer(analyzer)
    vis.generate_html_report(output_html, dark_mode=dark_mode)
    print(f"[+] Interactive visual report generated: {output_html}")

    if auto_open:
        abs_path = os.path.abspath(output_html)
        print(f"[+] Opening in web browser: file://{abs_path}")
        try:
            webbrowser.open(f"file://{abs_path}")
        except Exception:
            pass

    return output_html


def batch_compare(files: List[str]):
    """Analyze and compare multiple ECG recordings in a summary table."""
    results = []
    print(f"\n[+] Batch analyzing {len(files)} ECG recordings...")

    for f in sorted(files):
        try:
            a = ECGAnalyzer(f)
            a.filter_signal()
            a.detect_r_peaks()
            a.compute_hrv_and_morphology()
            results.append((f, a.metadata, a.hrv))
        except Exception as e:
            print(f"[-] Error processing {f}: {e}")

    if not results:
        print("No valid recordings found.")
        return

    if RICH_AVAILABLE:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        t = Table(title="ECG Batch Comparison Summary", box=box.ROUNDED)
        t.add_column("Recording / File", style="bold cyan")
        t.add_column("Duration", justify="right")
        t.add_column("Beats", justify="right")
        t.add_column("Mean HR", style="bold red", justify="right")
        t.add_column("PACs", style="bold yellow", justify="right")
        t.add_column("PVCs", style="bold red", justify="right")
        t.add_column("Pauses", style="bold purple", justify="right")
        t.add_column("RMSSD (ms)", style="bold green", justify="right")
        t.add_column("SDNN (ms)", style="bold blue", justify="right")
        t.add_column("LF/HF", justify="right")
        t.add_column("Quality", justify="right")

        for f, m, h in results:
            t.add_row(
                os.path.basename(f),
                f"{m.duration_seconds/60:.1f}m",
                str(h.total_beats),
                f"{h.mean_hr_bpm:.1f}",
                str(h.pac_count),
                str(h.pvc_count),
                str(h.pause_count),
                f"{h.rmssd_ms:.1f}",
                f"{h.sdnn_ms:.1f}",
                f"{h.lf_hf_ratio:.2f}",
                f"{h.quality_score_pct:.1f}%"
            )
        console.print(t)
    else:
        print(f"{'File':<25} {'Dur':<6} {'Beats':<6} {'Mean HR':<8} {'PACs':<5} {'PVCs':<5} {'RMSSD':<8} {'SDNN':<8}")
        print("-" * 80)
        for f, m, h in results:
            print(f"{os.path.basename(f):<25} {m.duration_seconds/60:4.1f}m {h.total_beats:6d} {h.mean_hr_bpm:8.1f} {h.pac_count:5d} {h.pvc_count:5d} {h.rmssd_ms:8.1f} {h.sdnn_ms:8.1f}")


def start_server(port: int = 8080, directory: str = "."):
    """Launch a local dashboard server."""
    class ECGHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                # Check available jsonl files and generate index
                jsonl_files = sorted(glob.glob("*.jsonl"))
                reports = sorted(glob.glob("*_report.html"))
                
                cards_html = ""
                for jf in jsonl_files:
                    base = os.path.splitext(jf)[0]
                    rep = f"{base}_report.html"
                    cards_html += f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:18px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:16px; font-weight:700; color:#38bdf8;">{jf}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Size: {os.path.getsize(jf)/1024:.1f} KB</div>
                        </div>
                        <div style="display:flex; gap:10px;">
                            <a href="/analyze?file={urllib.parse.quote(jf)}" style="background:#38bdf8; color:#0f172a; padding:8px 16px; border-radius:6px; font-size:13px; font-weight:700; text-decoration:none;">View / Analyze</a>
                        </div>
                    </div>
                    """

                if not jsonl_files:
                    cards_html = "<div style='color:#94a3b8;'>No .jsonl files found in current directory.</div>"

                landing_html = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>Polar ECG Analyzer Server</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{ background:#0f172a; color:#f8fafc; font-family:sans-serif; padding:40px 20px; }}
                        .container {{ max-width:800px; margin:0 auto; }}
                        h1 {{ font-size:24px; margin-bottom:8px; }}
                        p {{ color:#94a3b8; font-size:14px; margin-bottom:24px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Polar H10 ECG Visualizer Server</h1>
                        <p>Select an ECG recording below to analyze and inspect interactively.</p>
                        {cards_html}
                    </div>
                </body>
                </html>"""
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(landing_html.encode("utf-8"))
                return

            elif parsed.path == "/analyze":
                qs = urllib.parse.parse_qs(parsed.query)
                target = qs.get("file", [None])[0]
                if target and os.path.exists(target):
                    base = os.path.splitext(target)[0]
                    rep = f"{base}_report.html"
                    # Generate report if not already generated or if source is newer
                    if not os.path.exists(rep) or os.path.getmtime(target) > os.path.getmtime(rep):
                        process_file(target, output_html=rep, auto_open=False)
                    self.send_response(302)
                    self.send_header("Location", f"/{rep}")
                    self.end_headers()
                    return

            super().do_GET()

    print(f"\n[+] Starting ECG Web Server at: http://localhost:{port}")
    print("[+] Press Ctrl+C to stop.")
    with socketserver.TCPServer(("", port), ECGHandler) as httpd:
        webbrowser.open(f"http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Server stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Polar H10 ECG Signal Processor, Peak Detector & HRV Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="*", default=["ECG.jsonl"], help="Path to one or more split ECG .jsonl files (default: ECG.jsonl)")
    parser.add_argument("-o", "--output", help="Path for output HTML dashboard (default: <name>_report.html)")
    parser.add_argument("--csv", help="Optional path to export RR intervals and instant HR to CSV")
    parser.add_argument("--trend-csv", help="Optional path to export the 5-min windowed HRV trend to CSV (for 24h day/night trending)")
    parser.add_argument("--notch", type=float, nargs="+", default=[50.0, 60.0], help="Mains hum notch filter frequency/frequencies in Hz (default: 50.0 60.0, covering both mains regions since the recording region isn't known ahead of time)")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open report in default browser")
    parser.add_argument("--light", action="store_true", help="Use light theme for generated report")
    parser.add_argument("--batch", nargs="+", help="Batch analyze and compare multiple .jsonl files separately")
    parser.add_argument("--serve", action="store_true", help="Run local interactive web server")
    parser.add_argument("--port", type=int, default=8080, help="Port for local web server (default: 8080)")

    args = parser.parse_args()

    if args.serve:
        start_server(port=args.port)
        return

    if args.batch:
        batch_compare(args.batch)
        return

    inputs = args.input if isinstance(args.input, list) else [args.input]
    for inp in inputs:
        if not os.path.exists(inp):
            print(f"Error: Input file '{inp}' not found.")
            sys.exit(1)

    target_input = inputs[0] if len(inputs) == 1 else inputs
    process_file(
        filepath=target_input,
        output_html=args.output,
        export_csv=args.csv,
        trend_csv=args.trend_csv,
        notch_freq=args.notch,
        dark_mode=not args.light,
        auto_open=not args.no_open
    )


if __name__ == "__main__":
    main()
