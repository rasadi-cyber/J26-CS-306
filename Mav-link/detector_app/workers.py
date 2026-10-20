"""Background CSV loading, training, and detection workers."""

import os

from PyQt5.QtCore import QThread, pyqtSignal

from .pipeline import predict_csv_frame, preview_csv, read_reader_csv, train_model, validate_csv_path


class CsvLoadWorker(QThread):
    loaded = pyqtSignal(list)
    failed = pyqtSignal(str)
    log_emitted = pyqtSignal(str)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = list(paths)

    def run(self):
        try:
            loaded = []
            for path in self.paths:
                validate_csv_path(path)
                if os.path.basename(path).lower().endswith("_ground_truth.csv"):
                    self.log_emitted.emit(f"[CSV] Skipping ground-truth sidecar: {path}")
                    continue
                frame = read_reader_csv(path)
                loaded.append(
                    {"path": path, "frame": frame, "preview": preview_csv(path, frame)}
                )
            if not loaded:
                raise ValueError("No Reader CSV files were selected.")
            self.loaded.emit(loaded)
        except Exception as exc:
            self.failed.emit(str(exc))


class DetectionWorker(QThread):
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)
    log_emitted = pyqtSignal(str)
    progress_changed = pyqtSignal(int, str)

    def __init__(self, paths, artifact, parent=None):
        super().__init__(parent)
        self.paths = list(paths)
        self.artifact = artifact

    def run(self):
        try:
            results = []
            total = max(len(self.paths), 1)
            for index, path in enumerate(self.paths):
                self.progress_changed.emit(
                    int(index * 100 / total), f"Checking {os.path.basename(path)}..."
                )
                validate_csv_path(path)
                if os.path.basename(path).lower().endswith("_ground_truth.csv"):
                    self.log_emitted.emit(f"[CSV] Skipping ground-truth sidecar: {path}")
                    continue
                frame = read_reader_csv(path)
                result = predict_csv_frame(frame, self.artifact)
                result["file"] = os.path.basename(path)
                result["path"] = path
                results.append(result)
                self.log_emitted.emit(
                    f"[DETECT] {result['file']}: {result['verdict']} "
                    f"({len(result['predictions']):,} packets, {len(result['events'])} events)."
                )
            if not results:
                raise ValueError("No valid Reader CSV files were selected for detection.")
            self.progress_changed.emit(100, "Detection complete")
            self.completed.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingWorker(QThread):
    completed = pyqtSignal(object, str)
    failed = pyqtSignal(str)
    log_emitted = pyqtSignal(str)
    progress_changed = pyqtSignal(int, str)

    def __init__(self, folder, model_type, use_smote, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.model_type = model_type
        self.use_smote = use_smote

    def run(self):
        try:
            artifact, output = train_model(
                self.folder,
                model_type=self.model_type,
                use_smote=self.use_smote,
                progress=self.progress_changed.emit,
                log=self.log_emitted.emit,
            )
            self.completed.emit(artifact, output)
        except Exception as exc:
            self.failed.emit(str(exc))
