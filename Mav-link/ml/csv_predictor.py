"""CSV-based, label-blind MAVLink attack detection."""

import os

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from .events import build_attack_events
from .features import CLASS_MAP, FEATURE_COLUMNS, build_features
from .predictor import load_model_artifact

REQUIRED_CSV_COLUMNS = {"timestamp", "seq", "sys_id", "comp_id", "msg_id"}


def predict_csv(df, model):
    """Classify a Reader-exported DataFrame; consult labels only after prediction."""
    artifact = load_model_artifact(model)
    if list(artifact["features"]) != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match this application's feature schema.")
    if df.empty:
        raise ValueError("CSV contains no packets.")
    missing = sorted(REQUIRED_CSV_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required Reader columns: {', '.join(missing)}")

    rows = df.copy()
    if "msg_name" in rows:
        bad_data = rows["msg_name"].astype(str).str.upper().eq("BAD_DATA")
    else:
        bad_data = pd.Series(False, index=rows.index)
    sys_id = pd.to_numeric(rows["sys_id"], errors="coerce").fillna(0)
    comp_id = pd.to_numeric(rows["comp_id"], errors="coerce").fillna(0)
    rows = rows.loc[~(bad_data | ((sys_id == 0) & (comp_id == 0)))].copy()
    if rows.empty:
        raise ValueError("CSV contains no valid MAVLink packets after BAD_DATA filtering.")

    rows["timestamp"] = pd.to_numeric(rows["timestamp"], errors="coerce")
    rows = rows.dropna(subset=["timestamp"]).sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)
    if rows.empty:
        raise ValueError("CSV contains no valid numeric timestamps.")

    features = build_features(rows)
    missing_features = set(FEATURE_COLUMNS) - set(features.columns)
    if missing_features:
        raise ValueError(f"Feature builder omitted required features: {sorted(missing_features)}")

    classifier = artifact["model"]
    probabilities = np.asarray(
        classifier.predict_proba(features.reindex(columns=FEATURE_COLUMNS)),
        dtype=float,
    )
    class_ids = np.asarray(classifier.classes_, dtype=int)
    winners = np.argmax(probabilities, axis=1)
    predictions = class_ids[winners]

    labels = (
        pd.to_numeric(rows["label"], errors="coerce")
        if "label" in rows.columns
        else None
    )
    result = rows.copy()
    for column in ("delta_time_ms", "delta_seq"):
        result[column] = features[column].to_numpy()
    result["seq_jump"] = features["seq_jump"].to_numpy()
    result["packet_no"] = np.arange(1, len(result) + 1, dtype=int)
    result["pred"] = predictions
    result["pred_name"] = [CLASS_MAP.get(int(value), "Unknown") for value in predictions]
    result["confidence"] = probabilities[np.arange(len(features)), winners].astype(float)

    events = build_attack_events(result)
    score = None
    if labels is not None:
        valid_labels = labels.isin([0, 1, 2])
        actual = labels.loc[valid_labels].astype(int)
        predicted = result.loc[valid_labels, "pred"].astype(int)
        true_attack = actual.isin([1, 2])
        predicted_attack = predicted.isin([1, 2])
        score = {
            "true_attack_packets": int(true_attack.sum()),
            "caught_attack_packets": int((true_attack & predicted_attack).sum()),
            "false_alarm_packets": int((~true_attack & predicted_attack).sum()),
        }
    event_types = sorted({event["attack_type"] for event in events})
    result.attrs["verdict"] = (
        f"ATTACK DETECTED: {' + '.join(event_types)}"
        if events
        else "Clean - no attack detected"
    )
    result.attrs["class_counts"] = {
        CLASS_MAP[class_id]: int((result["pred"] == class_id).sum())
        for class_id in (0, 1, 2)
    }
    result.attrs["duration_s"] = float(result["timestamp"].max() - result["timestamp"].min())
    result.attrs["packet_count"] = len(result)
    result.attrs["attack_events"] = events
    result.attrs["score_vs_label"] = score
    result.attrs["ground_truth_used_for_prediction"] = False
    result.attrs["source_kind"] = "csv"
    return result


class CsvAttackCheckWorker(QThread):
    """Run one or more CSV checks asynchronously."""

    progress_changed = pyqtSignal(int, str)
    log_emitted = pyqtSignal(str)
    check_completed = pyqtSignal(object)
    check_failed = pyqtSignal(str)

    def __init__(self, inputs, model, parent=None):
        super().__init__(parent)
        self.inputs = list(inputs)
        self.model = model

    def run(self):
        try:
            artifact = load_model_artifact(self.model)
            if not self.inputs:
                raise ValueError("Select one or more Reader CSV files or a CSV folder first.")
            expanded = []
            for name, source in self.inputs:
                if isinstance(source, pd.DataFrame):
                    expanded.append((name, source))
                elif os.path.isdir(source):
                    csv_files = sorted(
                        os.path.join(source, filename)
                        for filename in os.listdir(source)
                        if filename.lower().endswith(".csv")
                    )
                    expanded.extend((os.path.basename(path), path) for path in csv_files)
                else:
                    expanded.append((name, source))
            if not expanded:
                raise ValueError("The selected folder contains no CSV files.")

            results = []
            total = len(expanded)
            for index, (name, source) in enumerate(expanded):
                try:
                    self.progress_changed.emit(
                        int(index * 100 / total), f"Checking {name} ({index + 1}/{total})..."
                    )
                    rows = source if isinstance(source, pd.DataFrame) else pd.read_csv(source, low_memory=False)
                    predictions = predict_csv(rows, artifact)
                    results.append({"name": name, "source": source, "predictions": predictions, "error": None})
                    self.log_emitted.emit(
                        f"[CSV CHECK] {name}: {len(predictions):,} packets, "
                        f"{len(predictions.attrs['attack_events'])} attack event(s)."
                    )
                except Exception as exc:
                    self.log_emitted.emit(f"[CSV ERROR] {name}: {exc}")
                    results.append({"name": name, "source": source, "predictions": None, "error": str(exc)})
            self.progress_changed.emit(100, "CSV attack check complete")
            self.check_completed.emit(results)
        except Exception as exc:
            self.log_emitted.emit(f"[CSV CHECK ERROR] {exc}")
            self.check_failed.emit(str(exc))
