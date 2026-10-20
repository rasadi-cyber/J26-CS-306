"""Standalone PyQt5 interface for CSV-only MAVLink attack detection."""

import base64
import html
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .features import CLASS_COLORS, CLASS_MAP
from .pipeline import BAD_DATA_MESSAGE, load_model_artifact
from .styles import CYBER_DARK_THEME
from .workers import CsvLoadWorker, DetectionWorker, TrainingWorker


def _table(headers, parent=None, row_height=28):
    widget = QTableWidget(parent)
    widget.setColumnCount(len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setAlternatingRowColors(True)
    widget.setSelectionBehavior(QTableWidget.SelectRows)
    widget.setEditTriggers(QTableWidget.NoEditTriggers)
    widget.verticalHeader().setVisible(False)
    widget.verticalHeader().setDefaultSectionSize(row_height)
    widget.horizontalHeader().setStretchLastSection(True)
    return widget


def _set_table_rows(table, rows):
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            table.setItem(row_index, column_index, item)


class Card(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.setStyleSheet("QFrame { background:#0d1322; border:1px solid #1e293b; border-radius:8px; }")
        self.layout_box = QVBoxLayout(self)
        self.layout_box.setContentsMargins(10, 9, 10, 9)
        self.layout_box.setSpacing(6)
        heading = QLabel(title.upper(), self)
        heading.setObjectName("CardTitle")
        self.layout_box.addWidget(heading)


class CsvDropZone(QFrame):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.setAcceptDrops(True)
        self.setMinimumHeight(84)
        self.setStyleSheet("QFrame { background:#0d1322; border:1px dashed #38bdf8; border-radius:8px; }")
        layout = QVBoxLayout(self)
        label = QLabel("Drop Reader CSV files or a CSV folder here", self)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.callback(paths)
        event.acceptProposedAction()


class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train New Model")
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        self.folder_label = QLabel("No CSV folder selected", self)
        self.folder_label.setWordWrap(True)
        self.folder = ""
        choose = QPushButton("Choose CSV Folder", self)
        choose.clicked.connect(self._choose_folder)
        self.model_type = QComboBox(self)
        self.model_type.addItems(["Random Forest", "XGBoost"])
        self.smote = QCheckBox("Apply SMOTE to training files only", self)
        self.start_button = QPushButton("Train Model", self)
        self.start_button.setObjectName("PrimaryButton")
        self.progress = QProgressBar(self)
        self.progress.setValue(0)
        self.status = QLabel("Training split is by capture file.", self)
        self.status.setWordWrap(True)
        layout.addWidget(self.folder_label)
        layout.addWidget(choose)
        layout.addWidget(self.model_type)
        layout.addWidget(self.smote)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        self.worker = None

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select labeled Reader CSV folder")
        if folder:
            self.folder = folder
            self.folder_label.setText(folder)

    def begin_training(self):
        if not self.folder:
            QMessageBox.warning(self, "No Dataset Folder", "Choose a folder with labeled Reader CSV files.")
            return
        self.start_button.setEnabled(False)
        self.worker = TrainingWorker(
            self.folder, self.model_type.currentText(), self.smote.isChecked(), self
        )
        self.worker.progress_changed.connect(self._progress)
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self._completed)
        self.worker.log_emitted.connect(self.parent().log)
        self.worker.start()

    def _progress(self, value, message):
        self.progress.setValue(value)
        self.status.setText(message)

    def _failed(self, message):
        self.parent().log(f"[TRAIN ERROR] {message}")
        QMessageBox.critical(self, "Training Failed", message)
        self.start_button.setEnabled(True)

    def _completed(self, artifact, path):
        self.parent().set_model(artifact, path)
        self.accept()


class PacketTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = pd.DataFrame()
        self.page = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        toolbar = QHBoxLayout()
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Filter packets...")
        self.search.textChanged.connect(self.refresh)
        self.attacks_only = QCheckBox("Attacks only", self)
        self.attacks_only.toggled.connect(self.refresh)
        self.next_alert = QPushButton("Jump to next alert", self)
        self.next_alert.clicked.connect(self.jump_next_alert)
        self.page_label = QLabel("Page 1", self)
        self.prev_page = QPushButton("‹", self)
        self.next_page = QPushButton("›", self)
        self.prev_page.clicked.connect(self.prev)
        self.next_page.clicked.connect(self.next)
        for widget in (self.search, self.attacks_only, self.next_alert, self.prev_page, self.page_label, self.next_page):
            toolbar.addWidget(widget)
        layout.addLayout(toolbar)
        self.table = _table(
            ["Packet", "Time (s)", "Message / Command", "Seq", "Δ Seq", "Δt (ms)", "Prediction", "Confidence"],
            self,
        )
        layout.addWidget(self.table)

    def set_data(self, frame, packet_range=None):
        self.data = frame.copy()
        self.packet_range = packet_range
        self.page = 0
        self.refresh()

    def refresh(self, *_):
        frame = self.data
        if frame.empty:
            self.table.setRowCount(0)
            self.page_label.setText("Page 1")
            return
        if getattr(self, "packet_range", None):
            frame = frame[frame["packet_no"].between(*self.packet_range)]
        if self.attacks_only.isChecked():
            frame = frame[frame["pred"].isin([1, 2])]
        query = self.search.text().strip().lower()
        if query:
            mask = (
                frame[["msg_name", "command_name", "pred_name"]]
                .fillna("").astype(str).apply(lambda col: col.str.lower().str.contains(query, regex=False))
                .any(axis=1)
            )
            frame = frame[mask]
        self.filtered = frame
        page_size = 300
        total_pages = max(1, (len(frame) + page_size - 1) // page_size)
        self.page = min(self.page, total_pages - 1)
        section = frame.iloc[self.page * page_size:(self.page + 1) * page_size]
        rows = []
        for _, row in section.iterrows():
            message = str(row.get("msg_name", ""))
            command = str(row.get("command_name", "") or "")
            if command:
                message += " / " + command
            rows.append([
                int(row["packet_no"]),
                f"{float(row['timestamp']) - float(self.data['timestamp'].min()):.3f}",
                message,
                row.get("seq", ""),
                row.get("delta_seq", ""),
                row.get("delta_time_ms", ""),
                row["pred_name"],
                f"{float(row['confidence']):.1%}",
            ])
        _set_table_rows(self.table, rows)
        for row_index, item in enumerate(section["pred"].tolist()):
            if int(item) in (1, 2):
                for column in range(self.table.columnCount()):
                    cell = self.table.item(row_index, column)
                    if cell:
                        cell.setBackground(Qt.darkRed if int(item) == 1 else Qt.darkBlue)
        self.page_label.setText(f"Page {self.page + 1} of {total_pages}")

    def jump_next_alert(self):
        if self.data.empty:
            return
        start = int(self.data["packet_no"].iloc[self.page * 300]) if len(self.data) > self.page * 300 else 0
        alerts = self.data.loc[self.data["pred"].isin([1, 2]) & (self.data["packet_no"] > start), "packet_no"]
        if alerts.empty:
            alerts = self.data.loc[self.data["pred"].isin([1, 2]), "packet_no"]
        if not alerts.empty:
            packet = int(alerts.iloc[0])
            self.page = max(0, (packet - 1) // 300)
            self.refresh()
            self.table.selectRow((packet - 1) % 300)

    def prev(self):
        self.page = max(0, self.page - 1)
        self.refresh()

    def next(self):
        self.page += 1
        self.refresh()


class DetectorMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mav link attck Detector")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(CYBER_DARK_THEME)
        self.loaded_csvs = []
        self.results = []
        self.active_result = None
        self.model_artifact = None
        self.model_path = ""
        self.load_worker = None
        self.detect_worker = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(self)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame(root)
        header.setObjectName("HeaderFrame")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 8, 18, 8)
        title = QLabel("mav link attck Detector", header)
        title.setObjectName("AppTitle")
        subtitle = QLabel("CSV-ONLY MAVLINK THREAT CLASSIFICATION", header)
        subtitle.setObjectName("AppSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        splitter = QSplitter(Qt.Horizontal, root)
        sidebar = QFrame(splitter)
        sidebar.setObjectName("SidebarFrame")
        sidebar.setMinimumWidth(265)
        sidebar.setMaximumWidth(320)
        left = QVBoxLayout(sidebar)
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(8)

        model_card = Card("MODEL", sidebar)
        self.model_summary = QLabel("Model: Not loaded", model_card)
        self.model_summary.setWordWrap(True)
        self.model_details = QLabel("Trained on: --\nMacro-F1: --\nTrain files: --\nTest files: --", model_card)
        self.model_details.setWordWrap(True)
        model_card.layout_box.addWidget(self.model_summary)
        model_card.layout_box.addWidget(self.model_details)
        load_model = QPushButton("Load model.joblib", model_card)
        load_model.clicked.connect(self._load_model)
        model_card.layout_box.addWidget(load_model)
        train = QPushButton("Train New Model", model_card)
        train.clicked.connect(self._train)
        model_card.layout_box.addWidget(train)
        left.addWidget(model_card)

        input_card = Card("CSV INPUT", sidebar)
        self.drop_zone = CsvDropZone(self._handle_drop, input_card)
        input_card.layout_box.addWidget(self.drop_zone)
        browse = QPushButton("Browse CSV Files", input_card)
        browse.clicked.connect(self._browse_files)
        folder = QPushButton("Open Folder of CSVs", input_card)
        folder.clicked.connect(self._browse_folder)
        input_card.layout_box.addWidget(browse)
        input_card.layout_box.addWidget(folder)
        self.input_files_label = QLabel("No CSV files loaded", input_card)
        self.input_files_label.setWordWrap(True)
        input_card.layout_box.addWidget(self.input_files_label)
        left.addWidget(input_card)
        left.addStretch()
        self.check_button = QPushButton("Check for Attacks", sidebar)
        self.check_button.setObjectName("PrimaryButton")
        self.check_button.setMinimumHeight(48)
        self.check_button.clicked.connect(self._detect)
        left.addWidget(self.check_button)
        self.progress = QProgressBar(sidebar)
        self.progress.setValue(0)
        left.addWidget(self.progress)

        content = QWidget(splitter)
        right = QVBoxLayout(content)
        right.setContentsMargins(10, 9, 10, 7)
        right.setSpacing(7)
        self.verdict_banner = QLabel("Load a Reader CSV and model to begin detection", content)
        self.verdict_banner.setAlignment(Qt.AlignCenter)
        self.verdict_banner.setMinimumHeight(34)
        self.verdict_banner.setWordWrap(True)
        self.verdict_banner.setStyleSheet("background:#10192e; color:#94a3b8; border:1px solid #1e293b; border-radius:6px; font-weight:700;")
        right.addWidget(self.verdict_banner)

        preview = Card("CSV PREVIEW", content)
        self.preview_label = QLabel("File: --    Rows: --    Columns: --    Duration: --    Streams: --    BAD_DATA: --    Label: --", preview)
        self.preview_label.setWordWrap(True)
        preview.layout_box.addWidget(self.preview_label)
        right.addWidget(preview)

        files_card = Card("FILES", content)
        self.files_table = _table(
            ["File", "Packets", "Verdict", "Attack types", "Events", "Attack packets", "First attack time", "Avg confidence"],
            files_card,
            24,
        )
        self.files_table.cellClicked.connect(self._select_file)
        files_card.layout_box.addWidget(self.files_table)
        right.addWidget(files_card)

        self.stats_label = QLabel("Packets: 0   |   Normal: 0   |   Command Injection: 0   |   Replay: 0   |   Attack events: 0", content)
        self.stats_label.setMinimumHeight(28)
        self.stats_label.setStyleSheet("color:#38bdf8; font-weight:700;")
        right.addWidget(self.stats_label)

        self.tabs = QTabWidget(content)
        self.tabs.setObjectName("DetectorTabs")
        self.events_table = _table(
            ["Event #", "Attack type", "Start (capture / UTC)", "End (capture / UTC)", "Packet range", "Packets", "Source", "Commands", "Max Δ Seq", "Avg conf.", "Reason"],
            self.tabs,
        )
        self.events_table.cellClicked.connect(self._select_event)
        self.packet_table = PacketTable(self.tabs)
        timeline_page = QWidget(self.tabs)
        timeline_layout = QVBoxLayout(timeline_page)
        self.timeline_figure = Figure(facecolor="#0d1322", tight_layout=True)
        self.timeline_canvas = FigureCanvasQTAgg(self.timeline_figure)
        timeline_layout.addWidget(self.timeline_canvas)
        self.performance_table = _table(["Class", "Precision", "Recall", "F1", "Support"], self.tabs)
        self.performance_note = QLabel("Model test metrics are displayed when no label column is present.", self.tabs)
        performance_page = QWidget(self.tabs)
        perf_layout = QVBoxLayout(performance_page)
        perf_layout.addWidget(self.performance_note)
        self.confusion_label = QLabel("Confusion matrix: --", performance_page)
        perf_layout.addWidget(self.confusion_label)
        perf_layout.addWidget(self.performance_table)
        self.tabs.addTab(self.events_table, "1  Attack Events")
        self.tabs.addTab(self.packet_table, "2  Packets")
        self.tabs.addTab(timeline_page, "3  Timeline")
        self.tabs.addTab(performance_page, "4  Model Performance")
        right.addWidget(self.tabs, 1)

        export_layout = QHBoxLayout()
        for text, handler in (
            ("Export Predictions CSV", self._export_predictions),
            ("Export Events CSV", self._export_events),
            ("Export HTML Report", self._export_report),
        ):
            button = QPushButton(text, content)
            button.clicked.connect(handler)
            export_layout.addWidget(button)
        right.addLayout(export_layout)
        self.console = QPlainConsole(content)
        right.addWidget(self.console, 0)
        splitter.addWidget(sidebar)
        splitter.addWidget(content)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)
        footer = QLabel("Simulated or synthetic datasets must be described as simulated in reports.", root)
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:#94a3b8; background:#0a0f1d; border-top:1px solid #1e293b; padding:5px;")
        outer.addWidget(footer)
        self.setCentralWidget(root)

    def log(self, message):
        self.console.append(message)

    def _handle_drop(self, paths):
        files = []
        for path in paths:
            if os.path.isdir(path):
                files.extend(
                    os.path.join(path, name)
                    for name in sorted(os.listdir(path))
                    if name.lower().endswith(".csv")
                )
            else:
                files.append(path)
        self._load_csvs(files)

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Reader CSV files", "", "CSV files (*.csv)"
        )
        if paths:
            self._load_csvs(paths)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open folder of Reader CSV files")
        if folder:
            self._handle_drop([folder])

    def _load_csvs(self, paths):
        if not paths:
            return
        invalid = [path for path in paths if os.path.splitext(path)[1].lower() != ".csv"]
        if invalid:
            message = BAD_DATA_MESSAGE if any(
                os.path.splitext(path)[1].lower() in {".pcap", ".pcapng", ".cap"} for path in invalid
            ) else "Only Reader-exported .csv files are accepted."
            self.log(f"[CSV ERROR] {message}")
            QMessageBox.warning(self, "Unsupported Input", message)
            return
        self.load_worker = CsvLoadWorker(paths, self)
        self.load_worker.loaded.connect(self._csv_loaded)
        self.load_worker.failed.connect(self._error)
        self.load_worker.log_emitted.connect(self.log)
        self.load_worker.start()

    def _csv_loaded(self, loaded):
        self.loaded_csvs = loaded
        names = [os.path.basename(item["path"]) for item in loaded]
        self.input_files_label.setText(f"{len(names)} CSV file(s): " + ", ".join(names))
        self._show_preview(loaded[0]["preview"])
        self.log(f"[CSV] Loaded {len(loaded)} Reader CSV file(s).")

    def _show_preview(self, summary):
        self.preview_label.setText(
            f"File: {summary['file']}    Rows: {summary['rows']:,}    Columns: {summary['columns']}    "
            f"Duration: {summary['duration_s']:.3f} s    Streams: {summary['streams']}    "
            f"BAD_DATA: {summary['bad_data']}    Label column: {'Yes (scoring only)' if summary['has_label'] else 'No'}"
        )

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load detector model", "", "Model files (*.joblib)")
        if not path:
            return
        try:
            artifact = load_model_artifact(path)
            self.set_model(artifact, path)
        except Exception as exc:
            self._error(str(exc))

    def set_model(self, artifact, path):
        self.model_artifact = artifact
        self.model_path = path
        classifier = artifact["model"].named_steps.get("classifier", artifact["model"])
        name = type(classifier).__name__
        metrics = artifact.get("metrics", {})
        self.model_summary.setText(f"Model: {name}")
        self.model_details.setText(
            f"Trained on: {artifact.get('trained_at', '--')}\n"
            f"Macro-F1: {metrics.get('macro_f1', 0.0):.3f}\n"
            f"Train files: {', '.join(artifact.get('train_files', [])) or '--'}\n"
            f"Test files: {', '.join(artifact.get('test_files', [])) or '--'}"
        )
        self.log(f"[MODEL] Loaded {path}.")
        self._render_metrics(metrics, "Saved held-out test metrics (label column was not used for prediction).")

    def _train(self):
        dialog = TrainingDialog(self)
        dialog.start_button.clicked.connect(dialog.begin_training)
        dialog.exec_()

    def _detect(self):
        if not self.model_artifact:
            self._error("No model loaded. Load model.joblib or train a model first.")
            return
        if not self.loaded_csvs:
            self._error("No CSV files loaded.")
            return
        self.check_button.setEnabled(False)
        self.detect_worker = DetectionWorker(
            [item["path"] for item in self.loaded_csvs], self.model_artifact, self
        )
        self.detect_worker.completed.connect(self._detection_completed)
        self.detect_worker.failed.connect(self._detection_failed)
        self.detect_worker.progress_changed.connect(self._progress)
        self.detect_worker.log_emitted.connect(self.log)
        self.detect_worker.start()

    def _progress(self, value, message):
        self.progress.setValue(value)
        self.progress.setFormat(message)

    def _detection_failed(self, message):
        self.check_button.setEnabled(True)
        self._error(message)

    def _detection_completed(self, results):
        self.check_button.setEnabled(True)
        self.results = results
        rows = []
        for result in results:
            frame = result["predictions"]
            attacks = frame[frame["pred"].isin([1, 2])]
            events = result["events"]
            rows.append([
                result["file"],
                len(frame),
                result["verdict"],
                ", ".join(dict.fromkeys(event["attack_type"] for event in events)) or "--",
                len(events),
                len(attacks),
                f"{events[0]['start_time_relative_s']:.3f}s" if events else "--",
                f"{attacks['confidence'].mean():.1%}" if len(attacks) else "--",
            ])
        _set_table_rows(self.files_table, rows)
        self.files_table.resizeColumnsToContents()
        self.files_table.setColumnWidth(2, 210)
        self._select_result(0)

    def _select_file(self, row, _column):
        if 0 <= row < len(self.results):
            self._select_result(row)

    def _select_result(self, index):
        self.active_result = self.results[index]
        result = self.active_result
        frame = result["predictions"]
        self.packet_table.set_data(frame)
        self._render_events(result["events"])
        self._render_timeline(result)
        self._render_stats(frame, result["events"])
        if result["events"]:
            self.verdict_banner.setText(result["verdict"])
            self.verdict_banner.setStyleSheet(
                "background:#451a12; color:#ff8b63; border:1px solid #eb6834; border-radius:6px; font-weight:800; padding:5px;"
            )
        else:
            self.verdict_banner.setText("Clean - no attack detected")
            self.verdict_banner.setStyleSheet(
                "background:#064e3b; color:#34d399; border:1px solid #10b981; border-radius:6px; font-weight:800; padding:5px;"
            )
        summary = next((item["preview"] for item in self.loaded_csvs if item["path"] == result["path"]), None)
        if summary:
            self._show_preview(summary)
        if result["score"] is not None:
            self._render_metrics(
                result["score"],
                "label column used only for scoring, not for prediction",
            )
        elif self.model_artifact:
            self._render_metrics(
                self.model_artifact.get("metrics", {}),
                "No label column available; showing saved held-out test metrics.",
            )

    def _render_stats(self, frame, events):
        counts = frame["pred"].value_counts()
        self.stats_label.setText(
            f"Packets: {len(frame):,}   |   Normal: {counts.get(0, 0):,}   |   "
            f"Command Injection: {counts.get(1, 0):,}   |   Replay: {counts.get(2, 0):,}   |   "
            f"Attack events: {len(events):,}"
        )

    def _render_events(self, events):
        rows = []
        for index, event in enumerate(events):
            rows.append([
                event["event_id"],
                event["attack_type"],
                f"{event['start_time_relative_s']:.3f}s / {event['start_time_utc']}",
                f"{event['end_time_relative_s']:.3f}s / {event['end_time_utc']}",
                f"{event['first_packet_no']}-{event['last_packet_no']}",
                event["packet_count"],
                f"{event['source_sys_id']}/{event['source_comp_id']}",
                ", ".join(event["commands_involved"]) or "--",
                event["max_seq_jump"],
                f"{event['avg_confidence']:.1%}",
                event["reason"],
            ])
        _set_table_rows(self.events_table, rows)
        self.events_table.resizeColumnsToContents()
        self.events_table.setColumnWidth(10, 300)

    def _select_event(self, row, _column):
        if not self.active_result or row >= len(self.active_result["events"]):
            return
        event = self.active_result["events"][row]
        self.packet_table.set_data(
            self.active_result["predictions"],
            (event["first_packet_no"], event["last_packet_no"]),
        )
        self.tabs.setCurrentIndex(1)

    def _render_timeline(self, result):
        self.timeline_figure.clear()
        axis = self.timeline_figure.add_subplot(111)
        axis.set_facecolor("#0d1322")
        frame = result["predictions"]
        start = float(frame["timestamp"].min())
        relative = frame["timestamp"].astype(float) - start
        for label in (0, 1, 2):
            selected = frame["pred"].astype(int).eq(label)
            axis.scatter(
                relative[selected],
                [label] * int(selected.sum()),
                s=11,
                color=CLASS_COLORS[label],
                label=CLASS_MAP[label],
            )
        for event in result["events"]:
            color = "#eb6834" if event["attack_type"] == "Command Injection" else "#6250d6"
            axis.axvspan(
                event["start_time_relative_s"],
                max(event["end_time_relative_s"], event["start_time_relative_s"] + 0.01),
                color=color,
                alpha=0.18,
            )
        axis.set_yticks([0, 1, 2], ["Normal", "Injection", "Replay"])
        axis.tick_params(colors="#cbd5e1")
        axis.set_xlabel("Seconds from capture start", color="#cbd5e1")
        axis.set_title("Packet prediction timeline", color="#f8fafc")
        axis.grid(color="#1e293b", alpha=0.7)
        axis.legend(facecolor="#0d1322", labelcolor="#e2e8f0", loc="upper right")
        self.timeline_canvas.draw_idle()

    def _render_metrics(self, metrics, note):
        self.performance_note.setText(note)
        per_class = metrics.get("per_class", {})
        rows = [
            [
                name,
                f"{values.get('precision', 0):.3f}",
                f"{values.get('recall', 0):.3f}",
                f"{values.get('f1', 0):.3f}",
                values.get("support", 0),
            ]
            for name, values in per_class.items()
        ]
        _set_table_rows(self.performance_table, rows)
        self.confusion_label.setText(
            f"Macro-F1: {metrics.get('macro_f1', 0.0):.3f}    "
            f"Confusion matrix [Normal, Injection, Replay]: {metrics.get('confusion_matrix', '--')}"
        )

    def _current_result(self):
        if self.active_result is None:
            self._error("Run detection before exporting results.")
            return None
        return self.active_result

    def _export_predictions(self):
        result = self._current_result()
        if not result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions CSV", "predictions.csv", "CSV files (*.csv)")
        if path:
            result["predictions"].to_csv(path, index=False)
            self.log(f"[EXPORT] Predictions: {path}")

    def _export_events(self):
        result = self._current_result()
        if not result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export events CSV", "events.csv", "CSV files (*.csv)")
        if path:
            pd.DataFrame(result["events"]).to_csv(path, index=False)
            self.log(f"[EXPORT] Events: {path}")

    def _export_report(self):
        result = self._current_result()
        if not result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export HTML report", "attack_report.html", "HTML files (*.html)")
        if not path:
            return
        image = BytesIO()
        canvas = FigureCanvasAgg(self.timeline_figure)
        canvas.print_png(image)
        timeline = base64.b64encode(image.getvalue()).decode("ascii")
        event_rows = "".join(
            "<tr>" + "".join(
                f"<td>{html.escape(str(event[key]))}</td>"
                for key in ("event_id", "attack_type", "start_time_utc", "end_time_utc", "packet_count", "reason")
            ) + "</tr>"
            for event in result["events"]
        )
        metrics = result.get("score") or (self.model_artifact or {}).get("metrics", {})
        document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Attack Detection Report</title>
<style>body{{font:14px Arial;background:#0b0f19;color:#e2e8f0;margin:28px}}h1{{color:#38bdf8}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #1e293b;padding:8px;text-align:left}}th{{color:#38bdf8}}img{{width:100%;max-height:440px;object-fit:contain;background:#0d1322}}.note{{color:#94a3b8}}</style></head><body>
<h1>MAVLink Attack Detection Report</h1><h2>{html.escape(result['verdict'])}</h2><p>File: {html.escape(result['file'])}</p>
<h2>Attack Events</h2><table><thead><tr><th>Event</th><th>Type</th><th>Start UTC</th><th>End UTC</th><th>Packets</th><th>Reason</th></tr></thead><tbody>{event_rows}</tbody></table>
<h2>Timeline</h2><img alt="Detection timeline" src="data:image/png;base64,{timeline}">
<h2>Model Metrics</h2><p>Macro-F1: {metrics.get('macro_f1', '--')}</p><pre>{html.escape(str(metrics))}</pre>
<p class="note">Simulated or synthetic datasets must be described as simulated in reports.</p></body></html>"""
        with open(path, "w", encoding="utf-8") as report:
            report.write(document)
        self.log(f"[EXPORT] HTML report: {path}")

    def _error(self, message):
        self.log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Detector Error", message)


class QPlainConsole(QFrame):
    def __init__(self, parent=None):
        from PyQt5.QtWidgets import QPlainTextEdit

        super().__init__(parent)
        self.setObjectName("ConsoleFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(3)
        title = QLabel("CONSOLE", self)
        title.setStyleSheet("color:#38bdf8; font-size:10px; font-weight:700;")
        self.text = QPlainTextEdit(self)
        self.text.setObjectName("ConsoleLog")
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(500)
        layout.addWidget(title)
        layout.addWidget(self.text)
        self.setMaximumHeight(116)

    def append(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.text.appendPlainText(f"[{stamp}] {message}")
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())
