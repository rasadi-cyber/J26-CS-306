"""Background model training with file-isolated train/test evaluation."""

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .features import CLASS_MAP, FEATURE_COLUMNS, build_features, compute_stream_dynamics

LABELS = [0, 1, 2]


def _load_reader_csvs(folder_path, log):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Dataset folder does not exist: {folder_path}")
    csv_paths = sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith(".csv")
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in dataset folder: {folder_path}")

    loaded = {}
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        is_reader_csv = "label" in df.columns and (
            "msg_id" in df.columns or "command_id" in df.columns or "msg_name" in df.columns
        )
        if not is_reader_csv:
            log(f"[DATASET] Skipping non-Reader CSV: {os.path.basename(path)}")
            continue
        df["file"] = os.path.basename(path)
        loaded[os.path.basename(path)] = df
    if not loaded:
        raise ValueError("No Reader feature CSVs with a label column were found.")
    return loaded


def _split_by_file(file_labels, test_fraction=0.4, random_state=42):
    """Choose a near-60/40 group split while preserving attack classes where possible."""
    names = list(file_labels)
    if len(names) < 2:
        raise ValueError("At least two Reader CSV files are required for a file-based train/test split.")
    all_classes = set().union(
        *(set(file_labels[name][1].unique().tolist()) for name in names)
    )
    if len(all_classes) < 2:
        raise ValueError("Training data is single-class; include normal and attack-labeled files.")
    if not any(label in (1, 2) for label in all_classes):
        raise ValueError("Training data contains no attack-labeled files.")

    class_file_counts = {
        label: sum(label in file_labels[name][1].unique() for name in names)
        for label in all_classes
    }
    target_test_files = min(len(names) - 1, max(1, round(len(names) * test_fraction)))
    rng = np.random.default_rng(random_state)
    best_split = None
    best_score = float("inf")

    for _ in range(2000):
        shuffled = list(rng.permutation(names))
        test_files = shuffled[:target_test_files]
        train_files = shuffled[target_test_files:]
        train_classes = set().union(
            *(set(file_labels[name][1].unique().tolist()) for name in train_files)
        )
        test_classes = set().union(
            *(set(file_labels[name][1].unique().tolist()) for name in test_files)
        )
        if train_classes != all_classes:
            continue
        if not any(label in (1, 2) for label in test_classes):
            continue
        if 0 in all_classes and class_file_counts[0] > 1 and 0 not in test_classes:
            continue
        if any(
            label in (1, 2) and class_file_counts[label] > 1 and label not in test_classes
            for label in all_classes
        ):
            continue

        total_rows = sum(len(file_labels[name][1]) for name in names)
        test_rows = sum(len(file_labels[name][1]) for name in test_files)
        actual_fraction = test_rows / total_rows
        per_class_score = 0.0
        for label in all_classes:
            total_class_rows = sum(
                int((file_labels[name][1] == label).sum()) for name in names
            )
            test_class_rows = sum(
                int((file_labels[name][1] == label).sum()) for name in test_files
            )
            if total_class_rows:
                per_class_score += abs(test_class_rows / total_class_rows - test_fraction)
        score = abs(actual_fraction - test_fraction) + 0.25 * per_class_score
        if score < best_score:
            best_score = score
            best_split = (train_files, test_files)
    if best_split is None:
        raise ValueError(
            "Could not create a file-isolated split with attack files on both sides. "
            "Add more independently labeled Reader CSV files."
        )
    return best_split


class ModelTrainer(QThread):
    progress_changed = pyqtSignal(int, str)
    log_emitted = pyqtSignal(str)
    training_completed = pyqtSignal(dict)
    training_failed = pyqtSignal(str)

    def __init__(
        self,
        dataset_folder,
        model_type="Random Forest",
        use_smote=False,
        model_path=None,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_folder = dataset_folder
        self.model_type = model_type
        self.use_smote = use_smote
        self.model_path = model_path or os.path.join(dataset_folder, "model.joblib")

    def _log(self, message):
        self.log_emitted.emit(message)

    def run(self):
        try:
            self.progress_changed.emit(2, "Loading Reader CSV files...")
            datasets = _load_reader_csvs(self.dataset_folder, self._log)
            rows = pd.concat(datasets.values(), ignore_index=True)
            rows["label"] = pd.to_numeric(rows["label"], errors="coerce")
            rows = rows[rows["label"].isin(LABELS)].copy()
            if rows.empty:
                raise ValueError("Reader CSV files contain no valid labels (0, 1, 2).")
            rows["label"] = rows["label"].astype(int)

            feature_rows = compute_stream_dynamics(rows)
            feature_rows["file"] = rows.loc[feature_rows.index, "file"].to_numpy()
            labels = feature_rows["label"].astype(int)
            X = build_features(feature_rows)
            file_labels = {
                filename: (feature_rows[feature_rows["file"] == filename], labels[feature_rows["file"] == filename])
                for filename in sorted(feature_rows["file"].unique())
            }
            train_files, test_files = _split_by_file(file_labels)
            train_mask = feature_rows["file"].isin(train_files)
            test_mask = feature_rows["file"].isin(test_files)
            X_train, y_train = X.loc[train_mask], labels.loc[train_mask]
            X_test, y_test = X.loc[test_mask], labels.loc[test_mask]
            self._log(
                f"[SPLIT] {len(train_files)} train files / {len(test_files)} test files "
                f"({len(X_train):,} / {len(X_test):,} rows); files are kept disjoint."
            )
            self.progress_changed.emit(15, "Preparing scaler and classifier...")

            steps = [("scaler", StandardScaler())]
            if self.use_smote:
                class_counts = y_train.value_counts()
                min_class_count = int(class_counts.min())
                if min_class_count < 2:
                    raise ValueError(
                        "SMOTE needs at least two training rows in every class; "
                        "add more labeled captures or disable SMOTE."
                    )
                neighbors = min(5, min_class_count - 1)
                steps.append(("smote", SMOTE(k_neighbors=neighbors, random_state=42)))
                self._log(f"[TRAIN] SMOTE enabled with k_neighbors={neighbors} (training rows only).")

            if self.model_type == "Random Forest":
                classifier = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                )
            elif self.model_type == "XGBoost":
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
                raise ValueError(f"Unsupported model type: {self.model_type}")
            steps.append(("classifier", classifier))
            model = Pipeline(steps)

            self.progress_changed.emit(30, "Training model...")
            model.fit(X_train, y_train)
            self.progress_changed.emit(75, "Evaluating held-out capture files...")
            predictions = model.predict(X_test)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test,
                predictions,
                labels=LABELS,
                zero_division=0,
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
                    for index, label in enumerate(LABELS)
                },
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
                "feature_importances": {
                    name: float(value)
                    for name, value in zip(FEATURE_COLUMNS, model.named_steps["classifier"].feature_importances_)
                },
                "test_rows": int(len(y_test)),
            }
            artifact = {
                "model": model,
                "features": list(FEATURE_COLUMNS),
                "metrics": metrics,
                "train_files": sorted(train_files),
                "test_files": sorted(test_files),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            joblib.dump(artifact, self.model_path)
            self.progress_changed.emit(100, "Training complete")
            self._log(
                f"[TRAINED] {self.model_type} saved to {self.model_path}; "
                f"held-out macro-F1={metrics['macro_f1']:.3f}."
            )
            self.training_completed.emit(artifact)
        except Exception as exc:
            self._log(f"[TRAINING ERROR] {exc}")
            self.training_failed.emit(str(exc))
