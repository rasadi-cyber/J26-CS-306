"""MAVLink capture inference using the trained, shared feature pipeline."""

import os

import joblib
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mavlink_core.parser import MAVLinkPCAPParser

from .events import build_attack_events
from .features import CLASS_MAP, FEATURE_COLUMNS, build_features


def load_model_artifact(model):
    artifact = joblib.load(model) if isinstance(model, (str, os.PathLike)) else model
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise ValueError("Model file is not a valid SkyLink model.joblib artifact.")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")
    return artifact


def _score_ground_truth(pcap_path, packet_indices, predictions):
    base, _ = os.path.splitext(pcap_path)
    ground_truth_path = f"{base}_ground_truth.csv"
    if not os.path.isfile(ground_truth_path):
        return None
    truth_df = pd.read_csv(ground_truth_path)
    if not {"packet_idx", "label"}.issubset(truth_df.columns):
        raise ValueError(
            f"Ground-truth file must contain packet_idx and label columns: {ground_truth_path}"
        )
    truth = dict(
        zip(
            pd.to_numeric(truth_df["packet_idx"], errors="coerce"),
            pd.to_numeric(truth_df["label"], errors="coerce"),
        )
    )
    aligned = [
        (int(truth[index]), int(prediction))
        for index, prediction in zip(packet_indices, predictions)
        if index in truth and pd.notna(truth[index]) and int(truth[index]) in CLASS_MAP
    ]
    if not aligned:
        return {
            "rows_scored": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": [],
        }
    y_true, y_pred = zip(*aligned)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "rows_scored": len(aligned),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class": {
            CLASS_MAP[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate([0, 1, 2])
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def predict_pcap(pcap_path, model, progress_callback=None, log_callback=None):
    """Decode and classify MAVLink frames without loading their embedded/sidecar labels."""
    if not os.path.isfile(pcap_path):
        raise FileNotFoundError(f"Capture file does not exist: {pcap_path}")
    if os.path.getsize(pcap_path) == 0:
        raise ValueError("Capture file is empty.")
    artifact = load_model_artifact(model)
    expected_features = list(artifact["features"])
    if expected_features != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match this application's feature schema.")

    def log(message):
        if log_callback:
            log_callback(message)

    parser = MAVLinkPCAPParser(
        target_ports=(14550, 14551),
        target_label=0,
        filter_bad_data=True,
        use_ground_truth=False,
        include_packet_metadata=True,
    )

    def parse_progress(total_packets, decoded_frames, stats):
        approximate_total = max(100, int(os.path.getsize(pcap_path) / 120))
        percent = min(75, 5 + int(total_packets / approximate_total * 70))
        if progress_callback:
            progress_callback(percent, f"Decoding packets ({decoded_frames:,} MAVLink frames)...")

    log("[DETECT] Parsing capture with ground-truth labels disabled.")
    rows = parser.parse_pcap(
        pcap_path,
        progress_callback=parse_progress,
        log_callback=log,
    )
    if rows.empty:
        if parser.total_packets_seen == 0:
            raise ValueError("Capture contains no packets.")
        raise ValueError("Capture contains no valid MAVLink packets.")

    rows = rows.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    packet_indices = (
        pd.to_numeric(rows["_packet_index"], errors="coerce").fillna(-1).astype(int).tolist()
        if "_packet_index" in rows
        else list(range(len(rows)))
    )
    X = build_features(rows.drop(columns=["label"], errors="ignore")).reindex(columns=expected_features)
    classifier = artifact["model"]
    probabilities = classifier.predict_proba(X)
    classes = list(classifier.classes_)
    winning_indices = np.argmax(probabilities, axis=1)
    predicted = np.asarray(classes, dtype=int)[winning_indices]
    confidence = probabilities[np.arange(len(X)), winning_indices]

    internal_rows = rows.copy()
    internal_rows["packet_no"] = np.arange(1, len(rows) + 1, dtype=int)
    internal_rows["pred"] = predicted
    internal_rows["pred_name"] = [CLASS_MAP.get(int(value), "Unknown") for value in predicted]
    internal_rows["confidence"] = confidence.astype(float)
    events = build_attack_events(internal_rows)

    if progress_callback:
        progress_callback(90, "Scoring detections and building attack events...")
    try:
        ground_truth_metrics = _score_ground_truth(pcap_path, packet_indices, predicted)
    except Exception as exc:
        ground_truth_metrics = None
        log(f"[SCORE ERROR] Predictions are complete, but sidecar scoring failed: {exc}")

    output = internal_rows.drop(
        columns=["_packet_index", "_src_addr", "_dst_addr"],
        errors="ignore",
    ).copy()
    output["label"] = pd.NA
    output.attrs["attack_events"] = events
    output.attrs["ground_truth_metrics"] = ground_truth_metrics
    output.attrs["ground_truth_used_for_prediction"] = False
    log(
        f"[DETECT] Classified {len(output):,} MAVLink frames; "
        f"{sum(int(value) in (1, 2) for value in predicted):,} predicted attack packets, "
        f"{len(events)} attack events (minimum 3 packets)."
    )
    if ground_truth_metrics is not None:
        log(
            f"[SCORE] Sidecar ground truth loaded only after prediction: "
            f"{ground_truth_metrics['rows_scored']:,} aligned packets scored."
        )
    if progress_callback:
        progress_callback(100, "Attack detection complete")
    return output


class PredictorWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    log_emitted = pyqtSignal(str)
    detection_completed = pyqtSignal(object)
    detection_failed = pyqtSignal(str)

    def __init__(self, pcap_path, model_path, parent=None):
        super().__init__(parent)
        self.pcap_path = pcap_path
        self.model_path = model_path

    def run(self):
        try:
            result = predict_pcap(
                self.pcap_path,
                self.model_path,
                progress_callback=self.progress_changed.emit,
                log_callback=self.log_emitted.emit,
            )
            self.detection_completed.emit(result)
        except Exception as exc:
            self.log_emitted.emit(f"[DETECTION ERROR] {exc}")
            self.detection_failed.emit(str(exc))


class BatchPredictorWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    log_emitted = pyqtSignal(str)
    batch_completed = pyqtSignal(object)
    batch_failed = pyqtSignal(str)

    def __init__(self, pcap_paths, model_path, parent=None):
        super().__init__(parent)
        self.pcap_paths = list(dict.fromkeys(pcap_paths))
        self.model_path = model_path

    def run(self):
        try:
            results = []
            total = len(self.pcap_paths)
            for index, path in enumerate(self.pcap_paths):
                self.log_emitted.emit(f"[BATCH] Analysing {os.path.basename(path)} ({index + 1}/{total}).")

                def report_progress(percent, status, file_index=index):
                    combined = int(((file_index + percent / 100.0) / total) * 100)
                    self.progress_changed.emit(
                        combined,
                        f"{os.path.basename(path)}: {status}",
                    )

                try:
                    metadata = MAVLinkPCAPParser.get_file_metadata(path)
                    predictions = predict_pcap(
                        path,
                        self.model_path,
                        progress_callback=report_progress,
                        log_callback=self.log_emitted.emit,
                    )
                    results.append(
                        {
                            "path": path,
                            "metadata": metadata,
                            "predictions": predictions,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    self.log_emitted.emit(f"[BATCH ERROR] {os.path.basename(path)}: {exc}")
                    results.append(
                        {"path": path, "metadata": None, "predictions": None, "error": str(exc)}
                    )
            self.batch_completed.emit(results)
        except Exception as exc:
            self.log_emitted.emit(f"[BATCH FATAL ERROR] {exc}")
            self.batch_failed.emit(str(exc))
