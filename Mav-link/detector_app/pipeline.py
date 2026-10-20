"""CSV validation, model training and inference for the standalone detector."""

import os
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .features import CLASS_MAP, FEATURE_COLUMNS, REQUIRED_COLUMNS, build_features

BAD_DATA_MESSAGE = "Convert the PCAP to CSV with SkyLink Analysis Pro first."


def validate_csv_path(path):
    extension = os.path.splitext(path)[1].lower()
    if extension in {".pcap", ".pcapng", ".cap"}:
        raise ValueError(BAD_DATA_MESSAGE)
    if extension != ".csv":
        raise ValueError("Only Reader-exported .csv files are accepted.")


def read_reader_csv(path):
    validate_csv_path(path)
    if os.path.basename(path).lower().endswith("_ground_truth.csv"):
        raise ValueError("Ground-truth sidecar CSV files are skipped.")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV contains no packet rows.")
    return frame


def preview_csv(path, frame):
    timestamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    duration = float(timestamps.max() - timestamps.min()) if len(timestamps) else 0.0
    bad = _bad_data_mask(frame)
    return {
        "file": os.path.basename(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "duration_s": duration,
        "streams": int(frame[["sys_id", "comp_id"]].drop_duplicates().shape[0]),
        "bad_data": int(bad.sum()),
        "has_label": "label" in frame.columns,
    }


def _bad_data_mask(frame):
    names = frame["msg_name"].fillna("").astype(str).str.strip().str.upper()
    sys_ids = pd.to_numeric(frame["sys_id"], errors="coerce").fillna(-1)
    comp_ids = pd.to_numeric(frame["comp_id"], errors="coerce").fillna(-1)
    return names.eq("BAD_DATA") | (sys_ids.eq(0) & comp_ids.eq(0))


def build_attack_events(predictions, max_gap_s=2.0, minimum_packets=3):
    """Group consecutive same-class attack frames, allocating a fresh list per group."""
    groups = []
    current_group = []
    current_class = None
    previous_timestamp = None

    for index, row in predictions.iterrows():
        prediction = int(row["pred"])
        timestamp = float(row["timestamp"])
        is_attack = prediction in (1, 2)
        continues = (
            is_attack
            and prediction == current_class
            and previous_timestamp is not None
            and timestamp - previous_timestamp <= max_gap_s
        )
        if current_group and not continues:
            groups.append((current_class, current_group))
            current_group = []
        if is_attack:
            if not current_group:
                current_class = prediction
            current_group.append(index)
            previous_timestamp = timestamp
        else:
            current_class = None
            previous_timestamp = None
    if current_group:
        groups.append((current_class, current_group))

    capture_start = float(predictions["timestamp"].min())
    events = []
    for attack_class, indexes in groups:
        if len(indexes) < minimum_packets:
            continue
        group = predictions.loc[indexes]
        start_epoch = float(group["timestamp"].min())
        end_epoch = float(group["timestamp"].max())
        command_names = [
            str(name).strip()
            for name in group.loc[
                pd.to_numeric(group["command_id"], errors="coerce").fillna(0).gt(0),
                "command_name",
            ]
            if pd.notna(name) and str(name).strip()
        ]
        commands = [
            f"{name} x{count}"
            for name, count in Counter(command_names).most_common()
        ]
        max_seq_jump = int(pd.to_numeric(group["seq_jump"], errors="coerce").fillna(0).max())
        source = Counter(
            zip(group["sys_id"].astype(int), group["comp_id"].astype(int))
        ).most_common(1)
        sys_id, comp_id = source[0][0] if source else (0, 0)
        duration = max(0.0, end_epoch - start_epoch)
        events.append(
            {
                "event_id": f"EVT-{len(events) + 1:04d}",
                "attack_type": CLASS_MAP[attack_class],
                "start_time_relative_s": round(start_epoch - capture_start, 6),
                "end_time_relative_s": round(end_epoch - capture_start, 6),
                "start_time_utc": datetime.fromtimestamp(start_epoch, timezone.utc).isoformat(),
                "end_time_utc": datetime.fromtimestamp(end_epoch, timezone.utc).isoformat(),
                "first_packet_no": int(group["packet_no"].min()),
                "last_packet_no": int(group["packet_no"].max()),
                "packet_count": len(group),
                "source_sys_id": sys_id,
                "source_comp_id": comp_id,
                "commands_involved": commands,
                "max_seq_jump": max_seq_jump,
                "avg_confidence": float(group["confidence"].mean()),
                "reason": (
                    f"{len(group)} consecutive {CLASS_MAP[attack_class]} predictions from "
                    f"sys {sys_id}/comp {comp_id}; maximum sequence jump {max_seq_jump} "
                    f"over {duration:.2f} seconds."
                ),
            }
        )
    return events


def predict_csv_frame(frame, model_artifact):
    if frame.empty:
        raise ValueError("CSV contains no packet rows.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    artifact = load_model_artifact(model_artifact)
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_numeric(ordered["timestamp"], errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values(
        "timestamp", kind="mergesort"
    ).copy()
    bad_mask = _bad_data_mask(ordered)
    valid = ordered.loc[~bad_mask].copy()
    if valid.empty:
        raise ValueError("CSV contains no valid packets after dropping BAD_DATA rows.")

    valid["packet_no"] = np.arange(1, len(valid) + 1, dtype=int)
    X = build_features(valid)
    classifier = artifact["model"]
    probabilities = np.asarray(classifier.predict_proba(X), dtype=float)
    classes = np.asarray(classifier.classes_, dtype=int)
    winners = np.argmax(probabilities, axis=1)
    predicted = classes[winners]
    valid["pred"] = predicted
    valid["pred_name"] = [CLASS_MAP.get(int(value), "Unknown") for value in predicted]
    valid["confidence"] = probabilities[np.arange(len(valid)), winners].astype(float)
    valid["seq_jump"] = X["seq_jump"].to_numpy()
    events = build_attack_events(valid.reset_index(drop=True))
    attack_types = list(dict.fromkeys(event["attack_type"] for event in events))
    verdict = (
        "ATTACK DETECTED: " + ", ".join(attack_types)
        if events
        else "Clean - no attack detected"
    )
    score = score_predictions(valid) if "label" in valid.columns else None
    return {
        "predictions": valid.reset_index(drop=True),
        "events": events,
        "verdict": verdict,
        "bad_data": int(bad_mask.sum()),
        "score": score,
        "file": "",
    }


def load_model_artifact(model):
    artifact = joblib.load(model) if isinstance(model, (str, os.PathLike)) else model
    if not isinstance(artifact, dict) or not {"model", "features"}.issubset(artifact):
        raise ValueError("Model file is not a valid detector model.joblib artifact.")
    if list(artifact["features"]) != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match this detector.")
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")
    return artifact


def score_predictions(frame):
    truth = pd.to_numeric(frame["label"], errors="coerce")
    mask = truth.isin([0, 1, 2])
    if not mask.any():
        return None
    y_true = truth[mask].astype(int)
    y_pred = frame.loc[mask, "pred"].astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "rows_scored": int(mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class": {
            CLASS_MAP[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate([0, 1, 2])
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def train_model(folder, model_type="Random Forest", use_smote=False, progress=None, log=None):
    progress = progress or (lambda value, message: None)
    log = log or (lambda message: None)
    paths = sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(".csv")
        and not filename.lower().endswith("_ground_truth.csv")
    )
    if len(paths) < 2:
        raise ValueError("At least two labeled Reader CSV files are required.")

    frames = []
    file_names = []
    for path in paths:
        frame = read_reader_csv(path)
        if "label" not in frame.columns:
            log(f"[TRAIN] Skipping CSV without label column: {os.path.basename(path)}")
            continue
        frame = frame.loc[~_bad_data_mask(frame)].copy()
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        frame = frame.loc[frame["label"].isin([0, 1, 2])].copy()
        if frame.empty:
            log(f"[TRAIN] Skipping CSV without valid labeled packets: {os.path.basename(path)}")
            continue
        frame["label"] = frame["label"].astype(int)
        frame["_file_group"] = os.path.basename(path)
        frames.append(frame)
        file_names.append(os.path.basename(path))
    if len(frames) < 2:
        raise ValueError("At least two non-empty labeled Reader CSV files are required.")

    dataset = pd.concat(frames, ignore_index=True)
    labels = dataset["label"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Training data is single-class; include normal and attack-labeled rows.")
    groups = dataset["_file_group"]
    split = None
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.4, random_state=42)
    for train_indexes, test_indexes in splitter.split(dataset, labels, groups):
        train_classes = set(labels.iloc[train_indexes].unique())
        test_classes = set(labels.iloc[test_indexes].unique())
        if train_classes == set(labels.unique()) and test_classes & {1, 2}:
            split = (train_indexes, test_indexes)
            break
    if split is None:
        raise ValueError(
            "Could not split files while keeping all classes in training and an attack class "
            "in testing. Add independently labeled capture files."
        )
    train_indexes, test_indexes = split
    train_rows = dataset.iloc[train_indexes].sort_values("timestamp", kind="mergesort")
    test_rows = dataset.iloc[test_indexes].sort_values("timestamp", kind="mergesort")
    X_train = build_features(train_rows)
    y_train = train_rows["label"].astype(int)
    X_test = build_features(test_rows)
    y_test = test_rows["label"].astype(int)
    train_files = sorted(groups.iloc[train_indexes].unique().tolist())
    test_files = sorted(groups.iloc[test_indexes].unique().tolist())
    steps = [("scaler", StandardScaler())]
    if use_smote:
        counts = y_train.value_counts()
        smallest = int(counts.min())
        if smallest < 2:
            raise ValueError(
                "SMOTE needs at least two training rows in every class; disable SMOTE or add labeled rows."
            )
        steps.append(("smote", SMOTE(k_neighbors=min(5, smallest - 1), random_state=42)))
    if model_type == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == "XGBoost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    steps.append(("classifier", classifier))
    model = Pipeline(steps)
    progress(20, "Training classifier...")
    model.fit(X_train, y_train)
    progress(75, "Scoring held-out files...")
    predicted = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=[0, 1, 2], zero_division=0
    )
    metrics = {
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            CLASS_MAP[label]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate([0, 1, 2])
        },
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1, 2]).tolist(),
        "test_rows": int(len(y_test)),
    }
    artifact = {
        "model": model,
        "features": list(FEATURE_COLUMNS),
        "metrics": metrics,
        "train_files": train_files,
        "test_files": test_files,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    output = os.path.join(folder, "model.joblib")
    joblib.dump(artifact, output)
    progress(100, "Training complete")
    log(f"[TRAINED] Saved {output}; held-out macro-F1={metrics['macro_f1']:.3f}.")
    return artifact, output
