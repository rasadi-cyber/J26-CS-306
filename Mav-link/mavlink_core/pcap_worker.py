"""
Thread-Safe Worker for Asynchronous PCAP/PCAPNG Decoding & Feature Extraction
Supports packet-level ground-truth, BAD_DATA frame elimination, and per-stream dynamics.
"""

import time
import threading
import traceback
from typing import Dict, Any, Optional
import pandas as pd

from PyQt5.QtCore import QObject, pyqtSignal
from .parser import MAVLinkPCAPParser


class WorkerSignals(QObject):
    """Signals for background PCAP extraction worker."""
    progress_changed = pyqtSignal(int, str)  # percent (0-100), status_text
    stats_updated = pyqtSignal(dict)         # dict of counts: total, valid, commands, dropped, mean_dt, corrupted
    log_emitted = pyqtSignal(str)            # log message string
    parsing_completed = pyqtSignal(object)   # pandas DataFrame result
    parsing_failed = pyqtSignal(str)         # error message


class PCAPParserWorker:
    """
    Background worker to decapsulate PCAP/PCAPNG captures asynchronously
    without blocking the UI or causing CPython thread-state attach conflicts.
    """

    def __init__(
        self,
        filepath: str,
        target_ports=(14550, 14551),
        commands_only: bool = False,
        target_label: Any = "auto",
        filter_bad_data: bool = True,
        parent=None,
    ):
        self.signals = WorkerSignals()
        # Direct signal aliases for seamless Qt connection compatibility
        self.progress_changed = self.signals.progress_changed
        self.stats_updated = self.signals.stats_updated
        self.log_emitted = self.signals.log_emitted
        self.parsing_completed = self.signals.parsing_completed
        self.parsing_failed = self.signals.parsing_failed

        self.filepath = filepath
        self.target_ports = target_ports
        self.commands_only = commands_only
        self.target_label = target_label
        self.filter_bad_data = filter_bad_data
        self._is_cancelled = False
        self._thread: Optional[threading.Thread] = None

        self.parser = MAVLinkPCAPParser(
            target_ports=self.target_ports,
            commands_only=self.commands_only,
            target_label=self.target_label,
            filter_bad_data=self.filter_bad_data,
        )

    def cancel(self):
        """Request worker cancellation."""
        self._is_cancelled = True

    def start(self):
        """Start worker in a native thread."""
        self._is_cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:
        """Check if worker thread is active."""
        return self._thread.is_alive() if self._thread else False

    def wait(self, timeout: Optional[float] = None):
        """Wait for worker thread to finish."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self):
        start_time = time.time()
        self.log_emitted.emit(f"[START] Initializing PCAP Decapsulation Worker for {self.filepath}...")
        self.progress_changed.emit(5, "Reading packet headers...")

        try:
            metadata = self.parser.get_file_metadata(self.filepath)
            self.log_emitted.emit(f"[METADATA] File: {metadata['filename']} ({metadata['filesize_str']})")
            self.log_emitted.emit(f"[METADATA] SHA-256: {metadata['sha256']}")

            estimated_packets = max(100, int(metadata["filesize_bytes"] / 120))

            def on_progress(total_pkts: int, records_count: int, stats: Dict[str, Any]):
                if self._is_cancelled:
                    return
                pct = min(95, int((total_pkts / estimated_packets) * 90) + 5)
                status_msg = f"Parsing Datagrams ({stats.get('mavlink_valid', 0):,} frames decoded, {stats.get('corrupted', 0):,} bad data filtered)..."
                self.progress_changed.emit(pct, status_msg)
                self.stats_updated.emit(stats)

            def on_log(msg: str):
                self.log_emitted.emit(msg)

            df = self.parser.parse_pcap(
                filepath=self.filepath,
                progress_callback=on_progress,
                log_callback=on_log,
                is_cancelled=lambda: self._is_cancelled,
            )

            if self._is_cancelled:
                self.log_emitted.emit("[CANCEL] Operation cancelled by user.")
                self.progress_changed.emit(0, "Cancelled")
                return

            elapsed = time.time() - start_time
            rate = len(df) / max(0.001, elapsed)

            mean_dt = float(df["delta_time_ms"].mean()) if not df.empty and "delta_time_ms" in df else 0.0
            anomalies = int((df["delta_seq"] != 1).sum()) if not df.empty and "delta_seq" in df else 0
            injections = int((df["label"] == 1).sum()) if not df.empty and "label" in df else 0
            replays = int((df["label"] == 2).sum()) if not df.empty and "label" in df else 0
            benign = int((df["label"] == 0).sum()) if not df.empty and "label" in df else 0

            final_stats = {
                "total_packets": self.parser.extractor.total_frames_processed,
                "mavlink_valid": len(df),
                "commands_count": int((df["command_id"] > 0).sum()) if not df.empty else 0,
                "anomalies_count": anomalies,
                "mean_delta_t_ms": round(mean_dt, 2),
                "elapsed_sec": round(elapsed, 2),
                "throughput_fps": round(rate, 1),
                "label_benign": benign,
                "label_injections": injections,
                "label_replays": replays,
            }

            self.log_emitted.emit(
                f"[SUCCESS] Extraction completed in {elapsed:.2f}s ({rate:.1f} frames/sec). "
                f"Generated {len(df):,} tabular ML feature rows. "
                f"[Ground-Truth Labels -> Benign: {benign:,}, Injections: {injections:,}, Replays: {replays:,}]"
            )
            self.stats_updated.emit(final_stats)
            self.progress_changed.emit(100, "Extraction & Decapsulation Complete")
            self.parsing_completed.emit(df)

        except Exception as e:
            tb = traceback.format_exc()
            self.log_emitted.emit(f"[FATAL ERROR] {str(e)}\n{tb}")
            self.parsing_failed.emit(str(e))
