"""
Master Application Window for SkyLink Analysis Pro / MAVLink Forensic Converter
Supports Packet-Level Ground-Truth Labeling, Per-Stream Dynamics, BAD_DATA Filtering,
and Real-Flight Attack Injection.
"""

import os
import time
import csv
import html
import tempfile
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QButtonGroup, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QFrame, QSplitter,
    QScrollArea, QStackedWidget, QTabBar, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QAbstractItemView, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .styles import CYBER_DARK_THEME
from .widgets import (
    DropZoneWidget, PaginatedTableWidget, ConsoleLogWidget, StatCard,
    CsvDropZoneWidget,
)
from mavlink_core.pcap_worker import PCAPParserWorker
from mavlink_core.parser import MAVLinkPCAPParser
from mavlink_core.generator import SyntheticPCAPGenerator, FlightAttackInjector
from ml.predictor import load_model_artifact
from ml.csv_predictor import CsvAttackCheckWorker
from ml.trainer import ModelTrainer


class MainWindow(QMainWindow):
    """Primary GUI Application Window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAV-Link Attack Detection")
        self.resize(1360, 900)
        self.setMinimumSize(1100, 720)

        self.current_filepath = None
        self.current_df = pd.DataFrame()
        self.worker: PCAPParserWorker = None
        self.ml_dataset_folder = None
        self.ml_model_path = None
        self.ml_model_artifact = None
        self.ml_train_worker = None
        self.ml_file_results = {}
        self.ml_current_result = None
        self.ml_current_event_id = None
        self.ml_csv_inputs = []
        self.ml_csv_worker = None

        self._init_ui()
        self.setStyleSheet(CYBER_DARK_THEME)
        self.console.log("System initialized. SkyLink Analysis Pro v2.5 ready.")
        self.console.log("Ready to ingest UAV network captures (.pcap, .pcapng).")

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_vbox = QVBoxLayout(central_widget)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)

        # 1. Header Bar
        main_vbox.addWidget(self._create_header())

        # 2. Page content
        self.page_stack = QStackedWidget(self)
        self.reader_page = self._create_reader_page()
        self.ml_page = self._create_ml_page()
        self.page_stack.addWidget(self.reader_page)
        self.page_stack.addWidget(self.ml_page)
        main_vbox.addWidget(self.page_stack, 1)

        # 3. Bottom Progress & Footer Bar
        main_vbox.addWidget(self._create_footer())

    def _create_reader_page(self) -> QWidget:
        """Build the original PCAP-to-CSV screen without changing its controls."""
        page = QWidget(self)
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        body_splitter = QSplitter(Qt.Horizontal, self)
        body_splitter.setHandleWidth(2)
        body_splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; }")

        # Left Sidebar (Inside ScrollArea for responsiveness)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(self._create_sidebar())
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setMaximumWidth(380)
        body_splitter.addWidget(sidebar_scroll)

        # Central Area (Summary Cards + Decoded Stream Table + Console Log)
        body_splitter.addWidget(self._create_center_panel())
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)

        page_layout.addWidget(body_splitter)
        return page

    def _create_header(self) -> QWidget:
        header = QFrame(self)
        header.setObjectName("HeaderFrame")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)
        header_layout.setSpacing(6)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(14)

        # Title & Subtitle
        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        app_title = QLabel("MAV-Link Attack Detection", header)
        app_title.setObjectName("AppTitle")

        app_sub = QLabel("MAVLINK INTRUSION DETECTION SYSTEM", header)
        app_sub.setObjectName("AppSubtitle")

        title_box.addWidget(app_title)
        title_box.addWidget(app_sub)
        top_layout.addLayout(title_box)

        # Status Badge Pill
        self.status_badge = QLabel("• UAV LINK STANDBY", header)
        self.status_badge.setObjectName("StatusBadge")
        top_layout.addWidget(self.status_badge)

        top_layout.addStretch()

        # Action Buttons
        self.inject_btn = QPushButton("🎯 Inject Attacks into Real Flight", header)
        self.inject_btn.setObjectName("SecondaryButton")
        self.inject_btn.setCursor(Qt.PointingHandCursor)
        self.inject_btn.setToolTip("Blend authentic flight telemetry with command injection & replay attacks into the real timeline")
        self.inject_btn.clicked.connect(self._inject_attacks_dialog)
        top_layout.addWidget(self.inject_btn)

        self.demo_btn = QPushButton("⚡ Generate Synthetic PCAP", header)
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.setCursor(Qt.PointingHandCursor)
        self.demo_btn.setToolTip("Synthesize multi-stream UAV & GCS capture with packet-level ground-truth tags")
        self.demo_btn.clicked.connect(self._generate_sample_pcap_dialog)
        top_layout.addWidget(self.demo_btn)

        self.help_btn = QPushButton("Docs & Help", header)
        self.help_btn.setObjectName("SecondaryButton")
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.clicked.connect(self._show_help_dialog)
        top_layout.addWidget(self.help_btn)

        self.page_tabs = QTabBar(header)
        self.page_tabs.setObjectName("PageTabs")
        self.page_tabs.addTab("📄 PCAP → CSV")
        self.page_tabs.addTab("🛡 ML Attack Detector")
        self.page_tabs.currentChanged.connect(self._on_page_tab_changed)

        header_layout.addLayout(top_layout)
        header_layout.addWidget(self.page_tabs)

        return header

    def _on_page_tab_changed(self, index: int):
        self.page_stack.setCurrentIndex(index)

    def _create_ml_page(self) -> QWidget:
        page = QWidget(self)
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        body_splitter = QSplitter(Qt.Horizontal, page)
        body_splitter.setHandleWidth(2)
        body_splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; }")

        sidebar_scroll = QScrollArea(page)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setMinimumWidth(300)
        sidebar_scroll.setMaximumWidth(380)
        sidebar_scroll.setWidget(self._create_ml_sidebar())
        body_splitter.addWidget(sidebar_scroll)
        body_splitter.addWidget(self._create_ml_center_panel())
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)

        page_layout.addWidget(body_splitter)
        return page

    def _create_ml_sidebar(self) -> QWidget:
        sidebar = QFrame(self)
        sidebar.setObjectName("SidebarFrame")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        brand_box = QFrame(sidebar)
        brand_box.setObjectName("BrandBox")
        brand_layout = QVBoxLayout(brand_box)
        brand_layout.setContentsMargins(8, 8, 8, 8)
        brand_layout.setSpacing(2)
        brand_title = QLabel("🛡 ML ATTACK DETECTOR", brand_box)
        brand_title.setObjectName("BrandTitle")
        brand_subtitle = QLabel("MAVLink threat classification", brand_box)
        brand_subtitle.setObjectName("BrandSubtitle")
        brand_layout.addWidget(brand_title)
        brand_layout.addWidget(brand_subtitle)
        layout.addWidget(brand_box)

        model_card = self._create_ml_card(sidebar, "MODEL")
        self.btn_load_model = QPushButton("Load model.joblib", model_card)
        self.btn_load_model.setObjectName("SecondaryButton")
        self.btn_load_model.setCursor(Qt.PointingHandCursor)
        self.btn_load_model.clicked.connect(self._load_ml_model_dialog)
        model_card.layout().addWidget(self.btn_load_model)
        self.model_name_label = QLabel("Model: Not loaded", model_card)
        self.model_trained_label = QLabel("Trained on: --", model_card)
        self.model_f1_label = QLabel("Macro-F1: --", model_card)
        for label in (self.model_name_label, self.model_trained_label, self.model_f1_label):
            label.setStyleSheet("color: #94a3b8; font-size: 11px;")
            label.setWordWrap(True)
            model_card.layout().addWidget(label)
        self.model_status = QLabel("MODEL NOT LOADED", model_card)
        self.model_status.setObjectName("StatusBadge")
        self.model_status.setAlignment(Qt.AlignCenter)
        self.model_status.setStyleSheet("""
            background-color: #422006;
            color: #fbbf24;
            border: 1px solid #d97706;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 700;
        """)
        model_card.layout().addWidget(self.model_status)
        layout.addWidget(model_card)

        csv_card = self._create_ml_card(sidebar, "CSV INPUT · UPLOAD & DETECT")
        self.csv_drop_zone = CsvDropZoneWidget(csv_card)
        self.csv_drop_zone.files_selected.connect(self._on_csv_sources_selected)
        csv_card.layout().addWidget(self.csv_drop_zone)
        csv_buttons = QHBoxLayout()
        self.btn_browse_csv = QPushButton("Browse CSVs", csv_card)
        self.btn_browse_csv.setObjectName("SecondaryButton")
        self.btn_browse_csv.clicked.connect(self._browse_detection_csvs)
        csv_buttons.addWidget(self.btn_browse_csv)
        self.btn_browse_csv_folder = QPushButton("Folder", csv_card)
        self.btn_browse_csv_folder.setObjectName("SecondaryButton")
        self.btn_browse_csv_folder.clicked.connect(self._browse_detection_csv_folder)
        csv_buttons.addWidget(self.btn_browse_csv_folder)
        csv_card.layout().addLayout(csv_buttons)
        self.btn_use_reader_csv = QPushButton("Use Reader CSV", csv_card)
        self.btn_use_reader_csv.setObjectName("SecondaryButton")
        self.btn_use_reader_csv.clicked.connect(self._use_reader_csv)
        csv_card.layout().addWidget(self.btn_use_reader_csv)
        self.btn_check_csv = QPushButton("✓ Check for Attacks", csv_card)
        self.btn_check_csv.setObjectName("PrimaryButton")
        self.btn_check_csv.setMinimumHeight(42)
        self.btn_check_csv.clicked.connect(self._check_csv_for_attacks)
        csv_card.layout().addWidget(self.btn_check_csv)
        layout.addWidget(csv_card)

        training_card = self._create_ml_card(sidebar, "TRAIN NEW MODEL")
        self.btn_select_csv_folder = QPushButton("Select CSV Folder", training_card)
        self.btn_select_csv_folder.setObjectName("SecondaryButton")
        self.btn_select_csv_folder.setCursor(Qt.PointingHandCursor)
        self.btn_select_csv_folder.clicked.connect(self._select_ml_dataset_folder)
        training_card.layout().addWidget(self.btn_select_csv_folder)
        self.dataset_folder_label = QLabel("No dataset folder selected", training_card)
        self.dataset_folder_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self.dataset_folder_label.setWordWrap(True)
        training_card.layout().addWidget(self.dataset_folder_label)
        self.model_type_combo = QComboBox(training_card)
        self.model_type_combo.addItems(["Random Forest", "XGBoost"])
        training_card.layout().addWidget(self.model_type_combo)
        self.smote_checkbox = QCheckBox("Apply SMOTE class balancing", training_card)
        training_card.layout().addWidget(self.smote_checkbox)
        self.btn_train_model = QPushButton("Train Model", training_card)
        self.btn_train_model.setObjectName("PrimaryButton")
        self.btn_train_model.clicked.connect(self._train_ml_model)
        training_card.layout().addWidget(self.btn_train_model)
        layout.addWidget(training_card)

        layout.addStretch()
        return sidebar

    @staticmethod
    def _create_ml_card(parent: QWidget, title: str) -> QFrame:
        card = QFrame(parent)
        card.setProperty("class", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(7)
        heading = QLabel(title, card)
        heading.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 12px;")
        card_layout.addWidget(heading)
        return card

    def _on_csv_sources_selected(self, paths):
        self.ml_csv_inputs = list(paths)
        self.ml_console.log(f"[CSV CHECK] {len(paths)} CSV file/folder source(s) selected.")

    def _browse_detection_csvs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Reader CSV files", "", "Reader CSV files (*.csv)"
        )
        if paths:
            self.csv_drop_zone.set_paths(paths)

    def _browse_detection_csv_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing Reader CSV files")
        if folder:
            self.csv_drop_zone.set_paths([folder])

    def _use_reader_csv(self):
        if self.current_df.empty:
            QMessageBox.warning(
                self, "No Reader CSV", "Load or parse a capture in the PCAP → CSV tab first."
            )
            return
        self.ml_csv_inputs = []
        self.csv_drop_zone.paths = []
        self.csv_drop_zone.label.setText("Using the CSV currently loaded in the Reader tab")
        self._start_csv_check([("Reader loaded CSV", self.current_df.copy())])

    def _check_csv_for_attacks(self):
        sources = []
        for path in self.ml_csv_inputs:
            name = os.path.basename(path.rstrip("\\/")) or path
            sources.append((name, path))
        if not sources:
            QMessageBox.warning(
                self, "No CSV Selected", "Drop or browse Reader CSV files/a folder, or use the Reader CSV."
            )
            return
        self._start_csv_check(sources)

    def _start_csv_check(self, sources):
        if not self.ml_model_path and self.ml_model_artifact is None:
            QMessageBox.warning(self, "No Model Loaded", "Load or train a model before checking CSV files.")
            return
        if self.ml_csv_worker and self.ml_csv_worker.isRunning():
            self.ml_console.log("[CSV CHECK] A CSV attack check is already running.")
            return
        model = self.ml_model_path or self.ml_model_artifact
        self.btn_check_csv.setEnabled(False)
        self.btn_train_model.setEnabled(False)
        self.ml_console.log(f"[CSV CHECK] Starting analysis of {len(sources)} source(s).")
        self.ml_csv_worker = CsvAttackCheckWorker(sources, model, parent=self)
        self.ml_csv_worker.progress_changed.connect(self._on_ml_progress)
        self.ml_csv_worker.log_emitted.connect(self.ml_console.log)
        self.ml_csv_worker.check_completed.connect(self._on_csv_check_completed)
        self.ml_csv_worker.check_failed.connect(self._on_csv_check_failed)
        self.ml_csv_worker.start()

    def _on_csv_check_completed(self, results):
        self.btn_check_csv.setEnabled(True)
        self.btn_train_model.setEnabled(True)
        normalized = []
        for index, item in enumerate(results):
            source = item["source"]
            path = source if isinstance(source, str) else f"reader-data-{index + 1}"
            result = {
                "path": path,
                "name": item["name"],
                "predictions": item["predictions"],
                "error": item["error"],
                "kind": "csv",
            }
            if isinstance(source, str) and os.path.isfile(source):
                try:
                    result["metadata"] = MAVLinkPCAPParser.get_file_metadata(source)
                except OSError as exc:
                    self.ml_console.log(f"[CSV METADATA ERROR] {item['name']}: {exc}")
            normalized.append(result)
        self._on_ml_batch_completed(normalized)
        if not any(item["predictions"] is not None for item in normalized):
            QMessageBox.warning(self, "CSV Check Failed", "No selected CSV source could be checked.")

    def _on_csv_check_failed(self, error):
        self.btn_check_csv.setEnabled(True)
        self.btn_train_model.setEnabled(True)
        QMessageBox.critical(self, "CSV Attack Check Failed", error)

    def _load_ml_model_dialog(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load SkyLink ML Model",
            self.ml_model_path or "model.joblib",
            "Joblib Models (*.joblib);;All Files (*)",
        )
        if not model_path:
            return
        try:
            artifact = load_model_artifact(model_path)
            self.ml_model_path = model_path
            self.ml_model_artifact = artifact
            model = artifact["model"]
            classifier = getattr(model, "named_steps", {}).get("classifier", model)
            model_name = type(classifier).__name__
            self.model_name_label.setText(f"Model: {model_name}")
            self.model_trained_label.setText(f"Trained on: {artifact.get('trained_at', '--')}")
            macro_f1 = artifact.get("metrics", {}).get("macro_f1")
            self.model_f1_label.setText(
                f"Macro-F1: {macro_f1:.3f}" if macro_f1 is not None else "Macro-F1: --"
            )
            self.model_status.setText("MODEL READY")
            self.model_status.setStyleSheet("""
                background-color: #064e3b;
                color: #34d399;
                border: 1px solid #059669;
                border-radius: 12px;
                padding: 4px 10px;
                font-size: 10px;
                font-weight: 700;
            """)
            self.ml_console.log(f"[MODEL] Loaded {os.path.basename(model_path)}.")
            self._populate_model_performance(artifact.get("metrics"), artifact)
        except Exception as exc:
            self.ml_console.log(f"[MODEL ERROR] {exc}")
            QMessageBox.critical(self, "Model Load Failed", str(exc))

    def _select_ml_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Reader CSV Dataset Folder")
        if folder:
            self.ml_dataset_folder = folder
            self.dataset_folder_label.setText(folder)
            self.ml_console.log(f"[DATASET] Selected folder: {folder}")

    def _train_ml_model(self):
        if not self.ml_dataset_folder:
            QMessageBox.warning(self, "No Dataset Folder", "Select the folder containing Reader CSV datasets first.")
            return
        if self.ml_train_worker and self.ml_train_worker.isRunning():
            self.ml_console.log("[TRAIN] Training is already in progress.")
            return

        model_path = os.path.join(self.ml_dataset_folder, "model.joblib")
        self.btn_train_model.setEnabled(False)
        self.ml_console.log(
            f"[TRAIN] Starting {self.model_type_combo.currentText()} training "
            f"(SMOTE: {'on' if self.smote_checkbox.isChecked() else 'off'})."
        )
        self.ml_train_worker = ModelTrainer(
            dataset_folder=self.ml_dataset_folder,
            model_type=self.model_type_combo.currentText(),
            use_smote=self.smote_checkbox.isChecked(),
            model_path=model_path,
            parent=self,
        )
        self.ml_train_worker.progress_changed.connect(self._on_ml_progress)
        self.ml_train_worker.log_emitted.connect(self.ml_console.log)
        self.ml_train_worker.training_completed.connect(self._on_ml_training_completed)
        self.ml_train_worker.training_failed.connect(self._on_ml_training_failed)
        self.ml_train_worker.start()

    def _on_ml_training_completed(self, artifact):
        self.ml_model_path = self.ml_train_worker.model_path
        self.ml_model_artifact = artifact
        self.btn_train_model.setEnabled(True)
        classifier = artifact["model"].named_steps["classifier"]
        self.model_name_label.setText(f"Model: {type(classifier).__name__}")
        self.model_trained_label.setText(f"Trained on: {artifact['trained_at']}")
        self.model_f1_label.setText(f"Macro-F1: {artifact['metrics']['macro_f1']:.3f}")
        self.model_status.setText("MODEL READY")
        self.model_status.setStyleSheet("""
            background-color: #064e3b;
            color: #34d399;
            border: 1px solid #059669;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 700;
        """)
        self._populate_model_performance(artifact.get("metrics"), artifact)
        QMessageBox.information(
            self,
            "Training Complete",
            f"Model saved to:\n{self.ml_model_path}\n\n"
            f"Held-out macro-F1: {artifact['metrics']['macro_f1']:.3f}",
        )

    def _on_ml_training_failed(self, error):
        self.btn_train_model.setEnabled(True)
        QMessageBox.critical(self, "Model Training Failed", error)

    def _on_ml_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.progress_percent.setText(f"{percent}%")
        self.progress_label.setText(f"Status: {status}")

    def _on_ml_batch_completed(self, results):
        self.btn_train_model.setEnabled(True)
        self.ml_file_results = {item["path"]: item for item in results}
        self._populate_files_table(results)
        successful = [item for item in results if item["predictions"] is not None]
        if successful:
            self._show_file_result(successful[0])
            self.ml_tabs.setCurrentIndex(0)
        elif results:
            self.verdict_banner.setText("⚠ Detection failed for all selected captures.")
            self.verdict_banner.setStyleSheet(
                "background-color: #2a1118; border: 1px solid #7f1d1d; "
                "border-left: 4px solid #ef4444; border-radius: 8px; "
                "color: #fca5a5; padding: 8px 12px; font-weight: 700;"
            )

    def _show_file_result(self, result):
        predictions = result["predictions"]
        if predictions is None:
            return
        self.ml_current_result = result
        self.ml_current_event_id = None
        display_df = predictions.copy()
        display_df["label"] = display_df["pred"]
        self.ml_table_widget.set_packet_range()
        self.ml_table_widget.set_attacks_only(self.attacks_only_toggle.isChecked())
        self.ml_table_widget.load_dataframe(display_df)
        predicted_counts = predictions["pred"].value_counts().to_dict()
        self.ml_stat_cards["Packets Analysed"].set_value(f"{len(predictions):,}")
        self.ml_stat_cards["Normal"].set_value(f"{predicted_counts.get(0, 0):,}")
        self.ml_stat_cards["Command Injection"].set_value(f"{predicted_counts.get(1, 0):,}")
        self.ml_stat_cards["Replay"].set_value(f"{predicted_counts.get(2, 0):,}")
        events = predictions.attrs.get("attack_events", [])
        self.ml_stat_cards["Attack Windows"].set_value(f"{len(events):,}")
        self._populate_event_table(events)
        self._update_file_verdict(result, events)
        self._draw_timeline(predictions, events)
        gt_metrics = predictions.attrs.get("ground_truth_metrics")
        if gt_metrics is not None:
            self.ml_tabs.setTabEnabled(3, True)
            if self.ml_model_artifact is None:
                self._populate_model_performance(gt_metrics, None, title="Ground-truth score")
        for event in events:
            self.ml_console.log(
                f"[{event['event_id']}] {event['attack_type']}: {event['reason']}"
            )
        ground_truth_metrics = gt_metrics
        if ground_truth_metrics and ground_truth_metrics["rows_scored"]:
            self.ml_console.log(
                f"[SCORE] Accuracy={ground_truth_metrics['accuracy']:.3f}; "
                f"macro-F1={ground_truth_metrics['macro_f1']:.3f} "
                "(ground truth was used only after prediction)."
            )
        self.progress_bar.setValue(100)
        self.progress_percent.setText("100%")
        self.progress_label.setText(
            f"Status: Analysed {result.get('name') or os.path.basename(result['path'])}"
        )

    def _populate_event_table(self, events):
        self.events_table.setSortingEnabled(False)
        self.events_table.setRowCount(len(events))
        for row_index, event in enumerate(events):
            start = (
                f"+{event['start_time_relative_s']:.3f}s / "
                f"{event['start_time_utc']}"
            )
            end = (
                f"+{event['end_time_relative_s']:.3f}s / "
                f"{event['end_time_utc']}"
            )
            values = [
                event["event_id"],
                event["attack_type"],
                start,
                end,
                f"{event['duration_s']:.3f}s",
                f"{event['first_packet_no']}-{event['last_packet_no']}",
                f"{event['source_sys_id']}/{event['source_comp_id']}",
                ", ".join(event["target_commands"]) or "—",
                f"{event['avg_confidence']:.1%} / {event['max_confidence']:.1%}",
                str(event.get("max_seq_jump", 0)),
                event["reason"],
                event["event_id"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    color = "#eb6834" if event["attack_type"] == "Command Injection" else "#a78bfa"
                    item.setForeground(QColor(color))
                    item.setBackground(QColor("#392116" if event["attack_type"] == "Command Injection" else "#252044"))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.events_table.setItem(row_index, column, item)
        self.events_table.setSortingEnabled(True)
        self.event_search.clear()

    def _filter_event_table(self, query):
        query = query.strip().lower()
        for row in range(self.events_table.rowCount()):
            visible = not query or any(
                query in (self.events_table.item(row, col).text().lower()
                          if self.events_table.item(row, col) else "")
                for col in range(self.events_table.columnCount() - 1)
            )
            self.events_table.setRowHidden(row, not visible)

    def _on_event_selected(self, row, _column):
        event_key = self.events_table.item(row, 11)
        if event_key is None or self.ml_current_result is None:
            return
        event_id = event_key.text()
        event = next(
            (
                item for item in self.ml_current_result["predictions"].attrs.get("attack_events", [])
                if item["event_id"] == event_id
            ),
            None,
        )
        if event is None:
            return
        self.ml_current_event_id = event_id
        self.ml_table_widget.set_packet_range(
            event["first_packet_no"], event["last_packet_no"]
        )
        self.ml_tabs.setCurrentIndex(1)

    def _on_attacks_only_toggled(self, checked):
        self.ml_table_widget.set_attacks_only(checked)

    def _jump_to_next_alert(self):
        predictions = (
            self.ml_current_result["predictions"]
            if self.ml_current_result
            else pd.DataFrame()
        )
        if predictions.empty:
            return
        current = self.ml_table_widget.table.currentItem()
        current_row = self.ml_table_widget.table.currentRow()
        current_packet_item = (
            self.ml_table_widget.table.item(current_row, 0)
            if current_row >= 0
            else None
        )
        current_packet = int(current_packet_item.text()) if current_packet_item else 0
        alerts = predictions[predictions["pred"].isin([1, 2])]
        alerts = alerts[alerts["packet_no"] > current_packet]
        if alerts.empty:
            alerts = predictions[predictions["pred"].isin([1, 2])]
        if alerts.empty:
            self.ml_console.log("[DETECT] No predicted attack packets to jump to.")
            return
        self.attacks_only_toggle.setChecked(True)
        self.ml_table_widget.set_packet_range()
        self.ml_table_widget.jump_to_packet(int(alerts.iloc[0]["packet_no"]))

    def _on_timeline_clicked(self, event):
        if event.xdata is None:
            return
        if not self.ml_current_result:
            return
        predictions = self.ml_current_result["predictions"]
        for row, attack_event in enumerate(predictions.attrs.get("attack_events", [])):
            start = attack_event["start_time_relative_s"]
            end = attack_event["end_time_relative_s"]
            if start <= event.xdata <= max(end, start + 0.01):
                event_key = attack_event["event_id"]
                for table_row in range(self.events_table.rowCount()):
                    key_item = self.events_table.item(table_row, 11)
                    if key_item and key_item.text() == event_key:
                        self.events_table.selectRow(table_row)
                        self._on_event_selected(table_row, 0)
                        break
                return

    def _draw_timeline(self, predictions, events):
        self.timeline_figure.clear()
        axis = self.timeline_figure.add_subplot(111)
        axis.set_facecolor("#0d1322")
        capture_start = float(predictions["timestamp"].min()) if not predictions.empty else 0.0
        lane_info = {
            0: ("Normal", "#10b981", 0),
            1: ("Command Injection", "#eb6834", 1),
            2: ("Replay", "#6250d6", 2),
        }
        for class_id, (name, color, lane) in lane_info.items():
            rows = predictions[predictions["pred"] == class_id]
            times = rows["timestamp"].astype(float) - capture_start
            axis.scatter(times, [lane] * len(rows), s=8, color=color, alpha=0.75, label=name)
        for event in events:
            color = "#eb6834" if event["attack_type"] == "Command Injection" else "#6250d6"
            start = event["start_time_relative_s"]
            end = event["end_time_relative_s"]
            axis.axvspan(start, max(end, start + 0.01), color=color, alpha=0.22, picker=True)
        axis.set_yticks([0, 1, 2], ["Normal", "Injection", "Replay"])
        axis.set_xlabel("Capture time (seconds)", color="#94a3b8")
        axis.tick_params(colors="#94a3b8")
        axis.grid(axis="x", color="#1e293b", alpha=0.7)
        axis.legend(facecolor="#131d31", edgecolor="#1e293b", labelcolor="#e2e8f0", loc="upper right")
        for spine in axis.spines.values():
            spine.set_color("#1e293b")
        self.timeline_canvas.draw_idle()

    def _update_file_verdict(self, result, events):
        predictions = result["predictions"]
        path = result["path"]
        display_name = result.get("name") or os.path.basename(path)
        counts = predictions["pred"].value_counts().to_dict()
        all_attacks = predictions[predictions["pred"].isin([1, 2])]
        event_types = sorted({event["attack_type"] for event in events})
        metadata = result.get("metadata")
        if metadata is None and os.path.isfile(path):
            metadata = MAVLinkPCAPParser.get_file_metadata(path)
        if events:
            relative_first = min(event["start_time_relative_s"] for event in events)
            relative_last = max(event["end_time_relative_s"] for event in events)
            first_event = min(events, key=lambda event: event["start_time_relative_s"])
            last_event = max(events, key=lambda event: event["end_time_relative_s"])
            duration = max(0.0, float(predictions["timestamp"].max() - predictions["timestamp"].min()))
            verdict = (
                f"ATTACK DETECTED: {', '.join(event_types)}  |  "
                f"File: {display_name}  |  "
                f"Types: {', '.join(event_types)}  |  Events: {len(events)}  |  "
                f"Attack packets: {len(all_attacks):,}/{len(predictions):,}  |  "
                f"First: +{relative_first:.3f}s ({first_event['start_time_utc']})  |  "
                f"Last: +{relative_last:.3f}s ({last_event['end_time_utc']})  |  "
                f"Duration: {duration:.2f}s"
            )
            color, border, foreground = "#2a1118", "#ef4444", "#fecaca"
        else:
            duration = max(0.0, float(predictions["timestamp"].max() - predictions["timestamp"].min()))
            verdict = (
                f"Clean - no attack detected  |  File: {display_name}  |  "
                f"Attack events: 0  |  Attack packets: {len(all_attacks):,}/{len(predictions):,}  |  "
                f"Duration: {duration:.2f}s"
            )
            color, border, foreground = "#092319", "#10b981", "#bbf7d0"
        model_name, macro_f1 = self._model_display_metrics()
        verdict += f"  |  Model: {model_name} (macro-F1 {macro_f1})"
        score = predictions.attrs.get("score_vs_label")
        if score is not None:
            verdict += (
                f"  |  Score vs label: true attack {score['true_attack_packets']}, "
                f"caught {score['caught_attack_packets']}, "
                f"false alarms {score['false_alarm_packets']}"
            )
        self.verdict_banner.setText(verdict)
        self.verdict_banner.setStyleSheet(
            f"background-color: {color}; border: 1px solid {border}; "
            f"border-left: 4px solid {border}; border-radius: 8px; "
            f"color: {foreground}; padding: 8px 12px; font-weight: 700;"
        )
        self.capture_metadata_label.setText(
            f"SHA-256: {metadata['sha256']}  |  Size: {metadata['filesize_str']} "
            f"({metadata['filesize_bytes']:,} bytes)"
            if metadata else "Reader in-memory CSV | SHA-256 / file size unavailable"
        )

    def _model_display_metrics(self):
        artifact = self.ml_model_artifact or {}
        model = artifact.get("model")
        if model is not None:
            classifier = getattr(model, "named_steps", {}).get("classifier", model)
            model_name = type(classifier).__name__
        else:
            model_name = "Not loaded"
        macro_f1 = artifact.get("metrics", {}).get("macro_f1")
        return model_name, f"{macro_f1:.3f}" if macro_f1 is not None else "--"

    def _export_predictions_csv(self):
        available = [
            (result.get("name") or os.path.basename(path), result["predictions"])
            for path, result in self.ml_file_results.items()
            if result.get("predictions") is not None
        ]
        if not available and self.ml_current_result and self.ml_current_result.get("predictions") is not None:
            available = [(self.ml_current_result.get("name", "predictions"), self.ml_current_result["predictions"])]
        if not available:
            QMessageBox.information(self, "No Predictions", "Run a CSV or PCAP attack check before exporting.")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Packet Predictions", "attack_predictions.csv", "CSV Files (*.csv)"
        )
        if not output_path:
            return
        try:
            frames = []
            for filename, predictions in available:
                frame = predictions.copy()
                frame.insert(0, "source_file", filename)
                frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
            self.ml_console.log(f"[EXPORT] Wrote packet predictions to {output_path}.")
            QMessageBox.information(self, "Predictions Exported", f"Saved predictions to:\n{output_path}")
        except Exception as exc:
            self.ml_console.log(f"[EXPORT ERROR] {exc}")
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _populate_files_table(self, results):
        csv_mode = bool(results and results[0].get("kind") == "csv")
        if csv_mode:
            self.files_table.setColumnCount(7)
            self.files_table.setHorizontalHeaderLabels([
                "File", "Packets", "Verdict", "Attack types", "Events",
                "Attack packets", "First attack time",
            ])
        else:
            self.files_table.setColumnCount(8)
            self.files_table.setHorizontalHeaderLabels([
                "File", "Packets", "Verdict", "Injection pkts", "Replay pkts",
                "First attack time", "Last attack time", "Confidence",
            ])
        self.files_table.setSortingEnabled(False)
        self.files_table.setRowCount(len(results))
        for row_index, result in enumerate(results):
            predictions = result["predictions"]
            name = result.get("name") or os.path.basename(result["path"])
            if predictions is None:
                values = (
                    [name, "0", f"ERROR: {result['error']}", "--", "0", "0", "--"]
                    if csv_mode
                    else [name, "0", f"ERROR: {result['error']}", "0", "0", "--", "--", "--"]
                )
            else:
                counts = predictions["pred"].value_counts().to_dict()
                events = predictions.attrs.get("attack_events", [])
                attacks = predictions[predictions["pred"].isin([1, 2])]
                attack_types = sorted({event["attack_type"] for event in events})
                first_time = min(
                    (event["start_time_utc"] for event in events),
                    default="--",
                )
                last_time = max(
                    (event["end_time_utc"] for event in events),
                    default="--",
                )
                confidence = (
                    float(attacks["confidence"].mean()) if not attacks.empty else 0.0
                )
                if csv_mode:
                    values = [
                        name,
                        f"{len(predictions):,}",
                        "ATTACK" if events else "CLEAN",
                        ", ".join(attack_types) or "--",
                        f"{len(events):,}",
                        f"{len(attacks):,}",
                        first_time,
                    ]
                else:
                    values = [
                        name,
                        f"{len(predictions):,}",
                        "ATTACK" if events else "CLEAN",
                        f"{counts.get(1, 0):,}",
                        f"{counts.get(2, 0):,}",
                        first_time,
                        last_time,
                        f"{confidence:.1%}",
                    ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, result["path"])
                if column == 2 and predictions is not None:
                    item.setForeground(QColor("#f87171" if result["predictions"].attrs.get("attack_events") else "#34d399"))
                self.files_table.setItem(row_index, column, item)
        self.files_table.setSortingEnabled(True)

    def _on_batch_file_selected(self, row, column):
        item = self.files_table.item(row, 0)
        if item is None:
            return
        filepath = item.data(Qt.UserRole)
        result = self.ml_file_results.get(filepath)
        if result and result["predictions"] is not None:
            self._show_file_result(result)
            self.ml_tabs.setCurrentIndex(0)

    def _populate_model_performance(self, metrics, artifact=None, title="Held-out Macro-F1"):
        self.ml_tabs.setTabEnabled(3, True)
        if not metrics:
            return
        macro_f1 = metrics.get("macro_f1")
        self.performance_macro_label.setText(
            f"{title}: {macro_f1:.3f}" if macro_f1 is not None else f"{title}: --"
        )
        while self.performance_class_layout.count():
            item = self.performance_class_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        classes = metrics.get("per_class", {})
        for name in ("Normal", "Command Injection", "Replay"):
            values = classes.get(name, {})
            card = StatCard(name, f"F1 {values.get('f1', 0.0):.3f}", "#10b981" if name == "Normal" else ("#eb6834" if name == "Command Injection" else "#6250d6"), self.ml_performance_page)
            card.setToolTip(
                f"Precision: {values.get('precision', 0.0):.3f}\n"
                f"Recall: {values.get('recall', 0.0):.3f}\n"
                f"F1: {values.get('f1', 0.0):.3f}"
            )
            card.value_label.setText(
                f"P {values.get('precision', 0.0):.2f} / R {values.get('recall', 0.0):.2f} / F1 {values.get('f1', 0.0):.2f}"
            )
            card.value_label.setStyleSheet("color: #f8fafc; font-size: 13px; font-weight: 700;")
            self.performance_class_layout.addWidget(card)

        confusion = metrics.get("confusion_matrix")
        self.confusion_figure.clear()
        axis = self.confusion_figure.add_subplot(111)
        axis.set_facecolor("#0d1322")
        if confusion:
            image = axis.imshow(confusion, cmap="magma")
            for y, row in enumerate(confusion):
                for x, value in enumerate(row):
                    axis.text(x, y, str(value), ha="center", va="center", color="#f8fafc")
            axis.set_xticks(range(3), ["Normal", "Injection", "Replay"], rotation=20)
            axis.set_yticks(range(3), ["Normal", "Injection", "Replay"])
            axis.set_title("Confusion Matrix", color="#e2e8f0")
            self.confusion_figure.colorbar(image, ax=axis, fraction=0.046)
        self.confusion_canvas.draw_idle()

        self.importance_figure.clear()
        axis = self.importance_figure.add_subplot(111)
        axis.set_facecolor("#0d1322")
        importances = metrics.get("feature_importances", {})
        top = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:12]
        if top:
            names, values = zip(*top)
            axis.barh(list(names)[::-1], list(values)[::-1], color="#38bdf8")
            axis.tick_params(colors="#94a3b8", labelsize=8)
            axis.set_title("Feature Importance", color="#e2e8f0")
        self.importance_canvas.draw_idle()
        artifact = artifact or self.ml_model_artifact or {}
        self.performance_files.setPlainText(
            "TRAIN FILES\n"
            + "\n".join(artifact.get("train_files", []))
            + "\n\nTEST FILES\n"
            + "\n".join(artifact.get("test_files", []))
        )

    def _export_detector_report(self):
        if not self.ml_current_result or self.ml_current_result["predictions"] is None:
            QMessageBox.information(self, "No Detection Results", "Run detection before exporting a report.")
            return
        html_path, _ = QFileDialog.getSaveFileName(
            self, "Export Detection Report", "attack_report.html", "HTML Report (*.html)"
        )
        if not html_path:
            return
        result = self.ml_current_result
        predictions = result["predictions"]
        events = predictions.attrs.get("attack_events", [])
        csv_path = os.path.splitext(html_path)[0] + "_events.csv"
        try:
            fields = list(events[0].keys()) if events else [
                "event_id", "attack_type", "first_packet_no", "last_packet_no",
                "start_time_relative_s", "end_time_relative_s", "start_time_utc",
                "end_time_utc", "duration_s", "packet_count", "avg_confidence",
                "max_confidence", "source_sys_id", "source_comp_id",
                "target_commands", "max_seq_jump", "src_ip_port", "dst_ip_port", "reason",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for event in events:
                    writer.writerow({
                        key: ", ".join(value) if isinstance(value, list) else value
                        for key, value in event.items()
                    })

            image_buffer = tempfile.SpooledTemporaryFile()
            self.timeline_figure.savefig(image_buffer, format="png", dpi=130, facecolor="#0d1322")
            image_buffer.seek(0)
            import base64
            timeline_image = base64.b64encode(image_buffer.read()).decode("ascii")
            image_buffer.close()
            event_rows = "".join(
                "<tr>" + "".join(
                    f"<td>{html.escape(str(value))}</td>"
                    for value in (
                        event["event_id"], event["attack_type"], event["start_time_utc"],
                        event["end_time_utc"], f"{event['duration_s']:.3f}s",
                        event["packet_count"], event["reason"],
                    )
                ) + "</tr>"
                for event in events
            )
            model_name, macro_f1 = self._model_display_metrics()
            document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SkyLink Detection Report</title><style>
body{{background:#0b0f19;color:#e2e8f0;font:14px Segoe UI,Arial;padding:24px}}
section{{background:#0d1322;border:1px solid #1e293b;border-radius:8px;padding:16px;margin:12px 0}}
th{{color:#38bdf8;text-align:left}}td,th{{padding:8px;border-bottom:1px solid #1e293b}}
img{{max-width:100%}} .verdict{{font-size:18px;font-weight:bold;color:#fca5a5}}
</style></head><body><h1>SkyLink Attack Detection Report</h1>
<section><div class="verdict">{html.escape(self.verdict_banner.text())}</div>
<p>Model: {html.escape(model_name)} | Macro-F1: {html.escape(macro_f1)}</p>
<p>{html.escape(self.capture_metadata_label.text())}</p></section>
<section><h2>Attack Events</h2><table><thead><tr><th>Event</th><th>Type</th>
<th>Start UTC</th><th>End UTC</th><th>Duration</th><th>Packets</th><th>Reason</th>
</tr></thead><tbody>{event_rows}</tbody></table></section>
<section><h2>Timeline</h2><img src="data:image/png;base64,{timeline_image}"></section>
<section><h2>Model Metrics</h2><pre>{html.escape(self.performance_files.toPlainText())}</pre>
<pre>{html.escape(str((self.ml_model_artifact or {}).get('metrics', {})))}</pre></section>
</body></html>"""
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write(document)
            self.ml_console.log(f"[REPORT] Wrote {html_path} and {csv_path}.")
            QMessageBox.information(
                self,
                "Report Exported",
                f"HTML report: {html_path}\nEvents CSV: {csv_path}",
            )
        except Exception as exc:
            self.ml_console.log(f"[REPORT ERROR] {exc}")
            QMessageBox.critical(self, "Report Export Failed", str(exc))

    def _create_ml_center_panel(self) -> QWidget:
        center = QWidget(self)
        layout = QVBoxLayout(center)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        self.verdict_banner = QLabel("Select a capture and run attack detection.", center)
        self.verdict_banner.setWordWrap(True)
        self.verdict_banner.setMinimumHeight(42)
        self.verdict_banner.setStyleSheet("""
            QLabel {
                background-color: #131d31;
                border: 1px solid #1e293b;
                border-left: 4px solid #38bdf8;
                border-radius: 8px;
                color: #e2e8f0;
                padding: 8px 12px;
                font-weight: 700;
            }
        """)
        layout.addWidget(self.verdict_banner)
        self.capture_metadata_label = QLabel("SHA-256: --  |  Size: --", center)
        self.capture_metadata_label.setWordWrap(True)
        self.capture_metadata_label.setStyleSheet(
            "color: #64748b; font-size: 10px; font-family: monospace; padding-left: 4px;"
        )
        layout.addWidget(self.capture_metadata_label)

        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(6)
        self.ml_stat_cards = {}
        stats = [
            ("Packets Analysed", "#f8fafc"),
            ("Normal", "#10b981"),
            ("Command Injection", "#f43f5e"),
            ("Replay", "#fbbf24"),
            ("Attack Windows", "#38bdf8"),
        ]
        for title, color in stats:
            card = StatCard(title, "0", color, center)
            self.ml_stat_cards[title] = card
            stats_layout.addWidget(card, 1)
        layout.addLayout(stats_layout)

        self.ml_tabs = QTabWidget(center)
        self.ml_tabs.setObjectName("DetectorTabs")
        self.ml_events_page = self._create_events_page()
        self.ml_packets_page = self._create_packets_page()
        self.ml_timeline_page = self._create_timeline_page()
        self.ml_performance_page = self._create_performance_page()
        self.ml_files_page = self._create_files_page()
        self.ml_tabs.addTab(self.ml_events_page, "Attack Events")
        self.ml_tabs.addTab(self.ml_packets_page, "Packets")
        self.ml_tabs.addTab(self.ml_timeline_page, "Timeline")
        self.ml_tabs.addTab(self.ml_performance_page, "Model Performance")
        self.ml_tabs.addTab(self.ml_files_page, "Files")
        self.ml_tabs.setTabEnabled(3, False)
        layout.addWidget(self.ml_tabs, 1)

        self.ml_console = ConsoleLogWidget(center)
        self.ml_console.title_label.setText("Attack Detection Console")
        self.ml_console.status_led.setText("● STANDBY")
        self.ml_console.status_led.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
        self.ml_console.setMaximumHeight(160)
        layout.addWidget(self.ml_console)
        return center

    def _create_events_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        title = QLabel("Attack Event Timeline", page)
        title.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: 700;")
        toolbar.addWidget(title)
        toolbar.addStretch()
        self.event_search = QLineEdit(page)
        self.event_search.setPlaceholderText("Search events, commands, sources...")
        self.event_search.textChanged.connect(self._filter_event_table)
        toolbar.addWidget(self.event_search)
        self.btn_export_report = QPushButton("Export Report", page)
        self.btn_export_report.setObjectName("ExportButton")
        self.btn_export_report.clicked.connect(self._export_detector_report)
        toolbar.addWidget(self.btn_export_report)
        self.btn_export_predictions = QPushButton("Export predictions CSV", page)
        self.btn_export_predictions.setObjectName("ExportButton")
        self.btn_export_predictions.clicked.connect(self._export_predictions_csv)
        toolbar.addWidget(self.btn_export_predictions)
        layout.addLayout(toolbar)

        self.events_table = QTableWidget(page)
        self.events_table.setColumnCount(12)
        self.events_table.setHorizontalHeaderLabels([
            "Event #", "Attack type", "Start (relative / UTC)", "End (relative / UTC)",
            "Duration", "Packets (first-last)", "Source sys/comp", "Commands involved",
            "Avg confidence", "Max seq jump", "Reason", "Event key",
        ])
        self.events_table.setColumnHidden(11, True)
        self.events_table.setSortingEnabled(True)
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.events_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.events_table.horizontalHeader().setStretchLastSection(True)
        for index, width in enumerate([88, 150, 185, 185, 85, 105, 105, 190, 100, 90, 330, 70]):
            self.events_table.setColumnWidth(index, width)
        self.events_table.cellClicked.connect(self._on_event_selected)
        layout.addWidget(self.events_table)
        return page

    def _create_packets_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.attacks_only_toggle = QCheckBox("Attacks only", page)
        self.attacks_only_toggle.toggled.connect(self._on_attacks_only_toggled)
        controls.addWidget(self.attacks_only_toggle)
        controls.addStretch()
        self.btn_next_alert = QPushButton("Jump to next alert", page)
        self.btn_next_alert.setObjectName("SecondaryButton")
        self.btn_next_alert.clicked.connect(self._jump_to_next_alert)
        controls.addWidget(self.btn_next_alert)
        layout.addLayout(controls)

        self.ml_table_widget = PaginatedTableWidget(page)
        self.ml_table_widget.table_title.setText("Packet Predictions")
        self.ml_table_widget.search_input.setPlaceholderText("Filter packets, messages, commands...")
        self.ml_table_widget.label_filter_combo.hide()
        self.ml_table_widget.export_btn.hide()
        self.ml_table_widget.table.setHorizontalHeaderLabels([
            "Packet no.", "Time", "Msg / Command", "Seq", "ΔSeq", "Δt (ms)",
            "Predicted class", "Confidence", "Sys/Comp",
        ])
        self.ml_table_widget.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.ml_table_widget.table.horizontalHeader().setStretchLastSection(True)
        for index, width in enumerate([90, 115, 230, 65, 70, 90, 160, 100, 100]):
            self.ml_table_widget.table.setColumnWidth(index, width)
        self.ml_table_widget.set_prediction_mode(True)
        layout.addWidget(self.ml_table_widget)
        return page

    def _create_timeline_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.timeline_figure = Figure(facecolor="#0d1322", tight_layout=True)
        self.timeline_canvas = FigureCanvas(self.timeline_figure)
        self.timeline_canvas.mpl_connect("button_press_event", self._on_timeline_clicked)
        layout.addWidget(self.timeline_canvas)
        return page

    def _create_performance_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self.performance_macro_label = QLabel("Held-out Macro-F1: --", page)
        self.performance_macro_label.setStyleSheet(
            "color: #38bdf8; font-size: 18px; font-weight: 800; padding: 4px;"
        )
        layout.addWidget(self.performance_macro_label)
        self.performance_class_layout = QHBoxLayout()
        layout.addLayout(self.performance_class_layout)

        visual_split = QHBoxLayout()
        self.confusion_figure = Figure(facecolor="#0d1322", tight_layout=True)
        self.confusion_canvas = FigureCanvas(self.confusion_figure)
        self.importance_figure = Figure(facecolor="#0d1322", tight_layout=True)
        self.importance_canvas = FigureCanvas(self.importance_figure)
        visual_split.addWidget(self.confusion_canvas, 1)
        visual_split.addWidget(self.importance_canvas, 1)
        layout.addLayout(visual_split, 1)
        self.performance_files = QPlainTextEdit(page)
        self.performance_files.setReadOnly(True)
        self.performance_files.setMaximumHeight(90)
        self.performance_files.setPlaceholderText("Training and held-out files appear here.")
        layout.addWidget(self.performance_files)
        return page

    def _create_files_page(self):
        page = QWidget(self)
        layout = QVBoxLayout(page)
        heading = QLabel("Capture Batch", page)
        heading.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: 700;")
        layout.addWidget(heading)
        self.files_table = QTableWidget(page)
        self.files_table.setColumnCount(8)
        self.files_table.setHorizontalHeaderLabels([
            "File", "Packets", "Verdict", "Injection pkts", "Replay pkts",
            "First attack time", "Last attack time", "Confidence",
        ])
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.files_table.setSortingEnabled(True)
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.files_table.horizontalHeader().setStretchLastSection(True)
        self.files_table.cellClicked.connect(self._on_batch_file_selected)
        layout.addWidget(self.files_table)
        return page

    def _create_sidebar(self) -> QWidget:
        sidebar = QFrame(self)
        sidebar.setObjectName("SidebarFrame")
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(14, 14, 14, 14)
        s_layout.setSpacing(12)

        # Brand Badge Box
        brand_box = QFrame(sidebar)
        brand_box.setObjectName("BrandBox")
        brand_layout = QVBoxLayout(brand_box)
        brand_layout.setContentsMargins(8, 8, 8, 8)
        brand_layout.setSpacing(2)

        b_title = QLabel("🛸 UAV OPERATOR CONSOLE", brand_box)
        b_title.setObjectName("BrandTitle")
        b_sub = QLabel("Per-Stream Tracking (sys/comp) | BAD_DATA Shield", brand_box)
        b_sub.setObjectName("BrandSubtitle")

        brand_layout.addWidget(b_title)
        brand_layout.addWidget(b_sub)
        s_layout.addWidget(brand_box)

        # File Ingestion Drop Zone Widget
        self.drop_zone = DropZoneWidget(sidebar)
        self.drop_zone.file_selected.connect(self._on_file_selected)
        s_layout.addWidget(self.drop_zone)

        # Target Dataset Label Selector Box (ML Ground Truth)
        lbl_box = QFrame(sidebar)
        lbl_box.setProperty("class", "card")
        lbl_layout = QVBoxLayout(lbl_box)
        lbl_layout.setContentsMargins(10, 10, 10, 10)
        lbl_layout.setSpacing(6)

        lbl_title = QLabel("🎯 ML Ground Truth Labeling Mode", lbl_box)
        lbl_title.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 12px;")
        lbl_layout.addWidget(lbl_title)

        self.label_group = QButtonGroup(self)
        self.radio_auto = QRadioButton("⚡ Auto / Packet-Level Ground Truth", lbl_box)
        self.radio_auto.setToolTip("Dynamic packet-level labeling from sidecar metadata, packet tags, or command forensics")
        self.radio_auto.setChecked(True)

        self.radio_benign = QRadioButton("🟢 Benign Flight Only (All Packets = 0)", lbl_box)
        self.radio_inject = QRadioButton("🔴 Injection Capture (Pkt-level: Cmds=1, Telemetry=0)", lbl_box)
        self.radio_replay = QRadioButton("🟡 Replay Capture (Pkt-level: Stale=2, Telemetry=0)", lbl_box)
        self.radio_force = QRadioButton("⚙️ Whole-File Override (Force all = 1)", lbl_box)

        self.ID_LABEL_AUTO = 10
        self.ID_LABEL_BENIGN = 0
        self.ID_LABEL_INJECT = 1
        self.ID_LABEL_REPLAY = 2
        self.ID_LABEL_FORCE = 99

        self.label_group.addButton(self.radio_auto, self.ID_LABEL_AUTO)
        self.label_group.addButton(self.radio_benign, self.ID_LABEL_BENIGN)
        self.label_group.addButton(self.radio_inject, self.ID_LABEL_INJECT)
        self.label_group.addButton(self.radio_replay, self.ID_LABEL_REPLAY)
        self.label_group.addButton(self.radio_force, self.ID_LABEL_FORCE)

        lbl_layout.addWidget(self.radio_auto)
        lbl_layout.addWidget(self.radio_benign)
        lbl_layout.addWidget(self.radio_inject)
        lbl_layout.addWidget(self.radio_replay)
        lbl_layout.addWidget(self.radio_force)
        s_layout.addWidget(lbl_box)

        # Forensic Filter Mode Options
        filter_box = QFrame(sidebar)
        filter_box.setProperty("class", "card")
        f_layout = QVBoxLayout(filter_box)
        f_layout.setContentsMargins(10, 10, 10, 10)
        f_layout.setSpacing(6)

        f_title = QLabel("⚙️ Ingestion & Cleaning Filters", filter_box)
        f_title.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 12px;")
        f_layout.addWidget(f_title)

        self.traffic_group = QButtonGroup(self)
        self.radio_all_traffic = QRadioButton("Show All MAVLink Traffic", filter_box)
        self.radio_all_traffic.setChecked(True)
        self.radio_commands_only = QRadioButton("Command Frames Only", filter_box)

        self.traffic_group.addButton(self.radio_all_traffic, 0)
        self.traffic_group.addButton(self.radio_commands_only, 1)

        f_layout.addWidget(self.radio_all_traffic)
        f_layout.addWidget(self.radio_commands_only)

        # Filter BAD_DATA frames checkbox
        self.chk_filter_bad_data = QCheckBox("Filter BAD_DATA / Corrupt (sys 0/0)", filter_box)
        self.chk_filter_bad_data.setChecked(True)
        self.chk_filter_bad_data.setToolTip("Eliminate framing errors, checksum failures, and 0/0 uninitialized frames (recommended)")
        f_layout.addWidget(self.chk_filter_bad_data)

        self.chk_highlight_anomalies = QCheckBox("Highlight Anomalies (Δ Seq ≠ 1)", filter_box)
        self.chk_highlight_anomalies.setChecked(True)
        self.chk_highlight_anomalies.toggled.connect(self._on_anomaly_toggle)
        f_layout.addWidget(self.chk_highlight_anomalies)

        s_layout.addWidget(filter_box)

        # Primary Action Buttons
        self.btn_execute = QPushButton("⚡ Execute Extraction & Parsing", sidebar)
        self.btn_execute.setObjectName("PrimaryButton")
        self.btn_execute.setCursor(Qt.PointingHandCursor)
        self.btn_execute.clicked.connect(self._execute_parsing)
        s_layout.addWidget(self.btn_execute)

        self.btn_export = QPushButton("📥 Export Structured CSV", sidebar)
        self.btn_export.setObjectName("ExportButton")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        s_layout.addWidget(self.btn_export)

        s_layout.addStretch()
        return sidebar

    def _create_center_panel(self) -> QWidget:
        center = QWidget(self)
        c_layout = QVBoxLayout(center)
        c_layout.setContentsMargins(12, 10, 12, 10)
        c_layout.setSpacing(0)

        # Central Splitter (Maximized Data Table on top, compact Console log on bottom)
        center_splitter = QSplitter(Qt.Vertical, center)
        center_splitter.setHandleWidth(3)
        center_splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; }")

        # Paginated Table Widget (occupying primary space)
        self.table_widget = PaginatedTableWidget(center_splitter)
        self.table_widget.export_requested.connect(self._export_csv)
        center_splitter.addWidget(self.table_widget)

        # Forensic Console Log Widget
        self.console = ConsoleLogWidget(center_splitter)
        self.console.setMaximumHeight(160)
        center_splitter.addWidget(self.console)

        center_splitter.setStretchFactor(0, 8)
        center_splitter.setStretchFactor(1, 1)
        center_splitter.setSizes([720, 110])
        c_layout.addWidget(center_splitter, 1)

        return center

    def _create_footer(self) -> QWidget:
        footer = QFrame(self)
        footer.setObjectName("FooterFrame")
        f_layout = QVBoxLayout(footer)
        f_layout.setContentsMargins(16, 6, 16, 6)
        f_layout.setSpacing(4)

        # Progress Status Line
        prog_top_layout = QHBoxLayout()
        self.progress_label = QLabel("Status: Standby", footer)
        self.progress_label.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        self.progress_percent = QLabel("0%", footer)
        self.progress_percent.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 700;")
        prog_top_layout.addWidget(self.progress_label)
        prog_top_layout.addStretch()
        prog_top_layout.addWidget(self.progress_percent)
        f_layout.addLayout(prog_top_layout)

        # Animated Progress Bar
        self.progress_bar = QProgressBar(footer)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        f_layout.addWidget(self.progress_bar)

        # Micro Footer
        micro_layout = QHBoxLayout()
        copy_label = QLabel("© 2026 SkyLink Data Systems. Mission Critical Forensics Interface.", footer)
        copy_label.setObjectName("FooterText")

        self.foot_stats = QLabel("0 Packets Decoded | 0 UDP Frames", footer)
        self.foot_stats.setObjectName("FooterText")

        micro_layout.addWidget(copy_label)
        micro_layout.addStretch()
        micro_layout.addWidget(self.foot_stats)
        f_layout.addLayout(micro_layout)

        return footer

    def _on_file_selected(self, filepath: str):
        self.current_filepath = filepath
        fname = os.path.basename(filepath)
        self.console.log(f"Capture selected: {fname}")
        self.status_badge.setText("• CAPTURE LOADED")
        self.status_badge.setStyleSheet("""
            background-color: #0c4a6e;
            color: #38bdf8;
            border: 1px solid #0284c7;
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: 700;
        """)

        try:
            meta = MAVLinkPCAPParser.get_file_metadata(filepath)
            self.drop_zone.set_file(filepath, meta, emit_signal=False)
            self.console.log(f"Integrity Check SHA-256: {meta['sha256']}")
        except Exception as e:
            self.console.log(f"Warning: could not calculate metadata: {str(e)}")

    def _on_anomaly_toggle(self, checked: bool):
        self.table_widget.set_anomalies_highlight(checked)

    def _execute_parsing(self):
        if not self.current_filepath or not os.path.exists(self.current_filepath):
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select or drop a valid .pcap or .pcapng network capture file first."
            )
            return

        commands_only = self.radio_commands_only.isChecked()
        filter_bad_data = self.chk_filter_bad_data.isChecked()

        # Robust label mode mapping (immune to Qt negative auto-assigned IDs)
        if self.radio_auto.isChecked():
            target_label = "auto"
        elif self.radio_benign.isChecked():
            target_label = 0
        elif self.radio_inject.isChecked():
            target_label = 1
        elif self.radio_replay.isChecked():
            target_label = 2
        elif self.radio_force.isChecked():
            target_label = 99
        else:
            cid = self.label_group.checkedId()
            if cid in (getattr(self, "ID_LABEL_AUTO", 10), -1, -2):
                target_label = "auto"
            elif cid in (0, 1, 2, 99):
                target_label = cid
            else:
                target_label = "auto"

        self.btn_execute.setEnabled(False)
        self.btn_export.setEnabled(False)

        self.status_badge.setText("• PROCESSING CAPTURE")
        self.status_badge.setStyleSheet("""
            background-color: #78350f;
            color: #fbbf24;
            border: 1px solid #d97706;
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: 700;
        """)

        # Launch background worker with per-stream dynamics and bad data filtering
        self.worker = PCAPParserWorker(
            filepath=self.current_filepath,
            target_ports=(14550, 14551),
            commands_only=commands_only,
            target_label=target_label,
            filter_bad_data=filter_bad_data,
            parent=self,
        )

        self.worker.progress_changed.connect(self._on_worker_progress)
        self.worker.stats_updated.connect(self._on_worker_stats)
        self.worker.log_emitted.connect(self.console.log)
        self.worker.parsing_completed.connect(self._on_worker_completed)
        self.worker.parsing_failed.connect(self._on_worker_failed)

        self.worker.start()

    def _on_worker_progress(self, percent: int, status_text: str):
        self.progress_bar.setValue(percent)
        self.progress_percent.setText(f"{percent}%")
        self.progress_label.setText(f"Status: {status_text}")

    def _on_worker_stats(self, stats: dict):
        valid = stats.get("mavlink_valid", 0)
        total = stats.get("total_packets", 0)
        self.foot_stats.setText(f"{valid:,} Valid MAVLink Frames | {total:,} Ingested Packets")

    def _on_worker_completed(self, df: pd.DataFrame):
        self.current_df = df
        self.btn_execute.setEnabled(True)
        self.btn_export.setEnabled(not df.empty)

        self.table_widget.load_dataframe(df)
        self.progress_bar.setValue(100)
        self.progress_percent.setText("100%")
        self.progress_label.setText(f"Status: Extraction complete ({len(df):,} records)")

        self.status_badge.setText("• UAV LINK ACTIVE")
        self.status_badge.setStyleSheet("""
            background-color: #064e3b;
            color: #34d399;
            border: 1px solid #059669;
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: 700;
        """)

        injections = int((df["label"] == 1).sum()) if not df.empty and "label" in df else 0
        replays = int((df["label"] == 2).sum()) if not df.empty and "label" in df else 0
        benign = int((df["label"] == 0).sum()) if not df.empty and "label" in df else 0

        self.console.log(
            f"Table rendered with {len(df):,} rows. Ground truth: {benign:,} Benign, "
            f"{injections:,} Injections, {replays:,} Replays."
        )

    def _on_worker_failed(self, error_msg: str):
        self.btn_execute.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.status_badge.setText("• ERROR")
        self.status_badge.setStyleSheet("""
            background-color: #7f1d1d;
            color: #f87171;
            border: 1px solid #dc2626;
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: 700;
        """)
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during PCAP decoding:\n\n{error_msg}")

    def _export_csv(self):
        if self.current_df.empty:
            QMessageBox.information(self, "No Data", "No parsed records available to export. Ingest and execute parsing first.")
            return

        default_fname = "mavlink_features_dataset.csv"
        if self.current_filepath:
            base = os.path.splitext(os.path.basename(self.current_filepath))[0]
            default_fname = f"{base}_ml_features.csv"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export ML Structured CSV",
            default_fname,
            "CSV Files (*.csv);;All Files (*)",
        )

        if save_path:
            try:
                self.current_df.to_csv(save_path, index=False)
                self.console.log(f"[EXPORT] Successfully exported {len(self.current_df):,} rows to: {save_path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Dataset successfully exported!\n\n"
                    f"File: {os.path.basename(save_path)}\n"
                    f"Rows: {len(self.current_df):,}\n"
                    f"Columns: {len(self.current_df.columns)}\n"
                    f"Path: {save_path}"
                )
            except Exception as e:
                self.console.log(f"[ERROR] Failed to save CSV: {str(e)}")
                QMessageBox.critical(self, "Export Failed", f"Failed to write CSV file:\n{str(e)}")

    def _inject_attacks_dialog(self):
        """Inject realistic attacks into an authentic flight PCAP."""
        input_pcap, _ = QFileDialog.getOpenFileName(
            self,
            "Select Base Authentic Flight PCAP",
            "",
            "PCAP Files (*.pcap *.pcapng);;All Files (*)",
        )
        if not input_pcap:
            return

        base = os.path.splitext(os.path.basename(input_pcap))[0]
        output_pcap, _ = QFileDialog.getSaveFileName(
            self,
            "Save Flight with Injected Attacks PCAP",
            f"{base}_with_attacks.pcap",
            "PCAP Files (*.pcap);;All Files (*)",
        )
        if not output_pcap:
            return

        self.console.log(f"[INJECT] Ingesting authentic flight: {input_pcap}...")
        self.console.log("[INJECT] Injecting realistic Command Injections (Disarm/Land) and Replay attacks into real flight timeline...")

        try:
            out_path, stats = FlightAttackInjector.inject_attacks_into_pcap(
                input_pcap_path=input_pcap,
                output_pcap_path=output_pcap,
                injection_count=50,
                replay_count=50,
            )
            self.console.log(
                f"[INJECT SUCCESS] Created blended capture: {out_path} "
                f"({stats['benign_packets']:,} authentic normal, {stats['injection_packets']} injections, {stats['replay_packets']} replays)"
            )
            self.drop_zone.set_file(out_path)
            self._on_file_selected(out_path)
            QMessageBox.information(
                self,
                "Attacks Injected Successfully",
                f"Realistic attacks blended into authentic flight capture!\n\n"
                f"Output: {os.path.basename(out_path)}\n"
                f"Authentic Normal Packets: {stats['benign_packets']:,} (label=0)\n"
                f"Injected Command Packets: {stats['injection_packets']} (label=1)\n"
                f"Replayed Telemetry Packets: {stats['replay_packets']} (label=2)\n\n"
                f"Timestamps and sensors match the authentic flight. Click 'Execute Extraction' to decode!"
            )
        except Exception as e:
            self.console.log(f"[ERROR] Attack injection failed: {str(e)}")
            QMessageBox.critical(self, "Injection Error", f"Failed to inject attacks into flight capture:\n{str(e)}")

    def _generate_sample_pcap_dialog(self):
        """Generate a realistic test UAV capture and immediately load it."""
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated UAV Sample PCAP",
            "synthetic_uav_flight_capture.pcap",
            "PCAP Files (*.pcap);;All Files (*)",
        )
        if not dest_path:
            return

        self.console.log(f"[GEN] Synthesizing multi-stream UAV flight capture with packet-level ground-truth tags...")
        try:
            SyntheticPCAPGenerator.create_uav_sample_pcap(dest_path, num_packets=1200, attack_scenario="mixed")
            self.console.log(f"[GEN] Sample PCAP created: {dest_path}")
            self.drop_zone.set_file(dest_path)
            self._on_file_selected(dest_path)
            QMessageBox.information(
                self,
                "Sample PCAP Created",
                f"Synthetic UAV flight capture generated successfully!\n\n"
                f"Location: {dest_path}\n"
                f"Packets: 1,200 multi-stream frames\n"
                f"Packet-level ground truth tags: Benign (0), Injections (1), Replays (2)\n\n"
                f"Click 'Execute Extraction & Parsing' to decode."
            )
        except Exception as e:
            self.console.log(f"[ERROR] Sample generation failed: {str(e)}")
            QMessageBox.critical(self, "Generation Error", f"Could not generate sample PCAP:\n{str(e)}")

    def _show_help_dialog(self):
        msg = (
            "<h3>SkyLink Analysis Pro — MAVLink Forensic Converter</h3>"
            "<p><b>Forensic Pipeline Features:</b></p>"
            "<ul>"
            "<li><b>Per-Stream Sequence Tracking:</b> Computes ΔSeq independently per <code>(sys_id, comp_id)</code> stream, eliminating false sequence anomalies between UAV and GCS.</li>"
            "<li><b>BAD_DATA Frame Shield:</b> Filters out corrupt, framing error, and <code>sys=0/0</code> junk frames.</li>"
            "<li><b>Packet-Level Ground Truth:</b> Individual packets are labeled accurately (0: Benign, 1: Injection, 2: Replay) instead of whole-file blanket labeling.</li>"
            "<li><b>Real-Flight Attack Injector:</b> Blends cyber-attacks directly into authentic flight captures to prevent synthetic distribution shifts.</li>"
            "</ul>"
            "<p><b>Feature Schema:</b><br>"
            "<code>timestamp, delta_time_ms, msg_id, msg_name, seq, delta_seq, sys_id, comp_id, command_id, command_name, param1..param7, label</code></p>"
        )
        QMessageBox.about(self, "SkyLink Analysis Pro — Guide", msg)
