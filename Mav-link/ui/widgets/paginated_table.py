"""
Paginated MAVLink Forensic Data Table Widget with Badges and Filtering
"""

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QComboBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QBrush


class PaginatedTableWidget(QFrame):
    """
    High-performance paginated table with cyber-forensics badge rendering
    and real-time in-memory search filtering.
    """

    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TableFrame")
        self.setStyleSheet("""
            #TableFrame {
                background-color: #0d1322;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)

        self.df_full: pd.DataFrame = pd.DataFrame()
        self.df_filtered: pd.DataFrame = pd.DataFrame()

        self.current_page = 1
        self.page_size = 100
        self.highlight_anomalies = True
        self.prediction_mode = False
        self.attacks_only = False
        self.packet_range = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Top Table Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        self.table_title = QLabel("Decoded MAVLink Stream", self)
        self.table_title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 800; letter-spacing: 0.3px;")
        
        self.records_badge = QLabel("0 Records", self)
        self.records_badge.setStyleSheet("""
            background-color: #1e293b;
            color: #38bdf8;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        """)

        toolbar_layout.addWidget(self.table_title)
        toolbar_layout.addWidget(self.records_badge)
        toolbar_layout.addStretch()

        # Search Bar
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Filter command or message name...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #0f172a;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 200px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #0ea5e9;
            }
        """)
        self.search_input.textChanged.connect(self._apply_search_filter)
        toolbar_layout.addWidget(self.search_input)

        # Label Filter Dropdown
        self.label_filter_combo = QComboBox(self)
        self.label_filter_combo.addItems([
            "All Labels (All Packets)",
            "🔴 Injections Only (Label = 1)",
            "🟡 Replays Only (Label = 2)",
            "⚠️ All Attacks (Label > 0)",
            "🟢 Benign Only (Label = 0)",
        ])
        self.label_filter_combo.setToolTip("Filter displayed packets by ground-truth classification")
        self.label_filter_combo.setStyleSheet("""
            QComboBox {
                background-color: #0f172a;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
                min-width: 175px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #f1f5f9;
                selection-background-color: #1e293b;
                selection-color: #38bdf8;
            }
        """)
        self.label_filter_combo.currentIndexChanged.connect(self._apply_search_filter)
        toolbar_layout.addWidget(self.label_filter_combo)

        # Export CSV Shortcut Button
        self.export_btn = QPushButton("📥 Export CSV", self)
        self.export_btn.setObjectName("SecondaryButton")
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self.export_requested.emit)
        toolbar_layout.addWidget(self.export_btn)

        layout.addLayout(toolbar_layout)

        # Main QTableWidget
        self.table = QTableWidget(self)
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Δt (ms)", "Msg Name", "Seq", "Δ Seq", "Sys/Comp", "Command", "Parameters (1-7)", "Label"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setShowGrid(True)

        # Proportional column width distribution
        self.table.setColumnWidth(0, 115)  # Timestamp
        self.table.setColumnWidth(1, 85)   # Δt (ms)
        self.table.setColumnWidth(2, 170)  # Msg Name
        self.table.setColumnWidth(3, 65)   # Seq
        self.table.setColumnWidth(4, 75)   # Δ Seq
        self.table.setColumnWidth(5, 95)   # Sys/Comp
        self.table.setColumnWidth(6, 175)  # Command
        self.table.setColumnWidth(8, 125)  # Label
        layout.addWidget(self.table)

        # Bottom Pagination Bar
        pagination_layout = QHBoxLayout()
        pagination_layout.setSpacing(10)

        # Rows per page selector
        rpp_label = QLabel("Rows per page:", self)
        rpp_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.page_size_combo = QComboBox(self)
        self.page_size_combo.addItems(["50", "100", "250", "500", "All"])
        self.page_size_combo.setCurrentText("100")
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)

        pagination_layout.addWidget(rpp_label)
        pagination_layout.addWidget(self.page_size_combo)
        pagination_layout.addStretch()

        # Nav Buttons
        self.btn_first = QPushButton("« First", self)
        self.btn_first.clicked.connect(self._go_first_page)
        self.btn_prev = QPushButton("‹ Prev", self)
        self.btn_prev.clicked.connect(self._go_prev_page)

        self.page_info_label = QLabel("Page 1 of 1", self)
        self.page_info_label.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 12px; padding: 0 8px;")

        self.btn_next = QPushButton("Next ›", self)
        self.btn_next.clicked.connect(self._go_next_page)
        self.btn_last = QPushButton("Last »", self)
        self.btn_last.clicked.connect(self._go_last_page)

        pagination_layout.addWidget(self.btn_first)
        pagination_layout.addWidget(self.btn_prev)
        pagination_layout.addWidget(self.page_info_label)
        pagination_layout.addWidget(self.btn_next)
        pagination_layout.addWidget(self.btn_last)

        layout.addLayout(pagination_layout)

    def load_dataframe(self, df: pd.DataFrame):
        """Set master DataFrame and refresh rendering."""
        self.df_full = df.copy()
        self.current_page = 1
        if hasattr(self, "label_filter_combo"):
            self.label_filter_combo.blockSignals(True)
            self.label_filter_combo.setCurrentIndex(0)
            self.label_filter_combo.blockSignals(False)
        self._apply_search_filter()

    def set_prediction_mode(self, enabled=True):
        self.prediction_mode = enabled
        self._render_current_page()

    def set_attacks_only(self, enabled):
        self.attacks_only = enabled
        self._apply_search_filter()

    def set_packet_range(self, first_packet=None, last_packet=None):
        self.packet_range = (
            (int(first_packet), int(last_packet))
            if first_packet is not None and last_packet is not None
            else None
        )
        self._apply_search_filter()

    def jump_to_packet(self, packet_no):
        if "packet_no" not in self.df_filtered:
            return
        matches = self.df_filtered.index[self.df_filtered["packet_no"] == int(packet_no)]
        if len(matches):
            position = self.df_filtered.index.get_loc(matches[0])
            self.current_page = position // self.page_size + 1
            self._render_current_page()
            self.table.selectRow(position % self.page_size)

    def set_anomalies_highlight(self, enabled: bool):
        self.highlight_anomalies = enabled
        self._render_current_page()

    def set_label_filter_mode(self, mode: str):
        """Programmatically switch label filter (e.g. 'injections', 'replays', 'attacks', 'benign', 'all')."""
        if not hasattr(self, "label_filter_combo"):
            return
        if mode == "injections":
            self.label_filter_combo.setCurrentIndex(1)
        elif mode == "replays":
            self.label_filter_combo.setCurrentIndex(2)
        elif mode == "attacks":
            self.label_filter_combo.setCurrentIndex(3)
        elif mode == "benign":
            self.label_filter_combo.setCurrentIndex(4)
        else:
            self.label_filter_combo.setCurrentIndex(0)

    def _apply_search_filter(self):
        if self.df_full.empty:
            self.df_filtered = pd.DataFrame()
            self.current_page = 1
            self._render_current_page()
            return

        filtered = self.df_full
        label_idx = self.label_filter_combo.currentIndex() if hasattr(self, "label_filter_combo") else 0

        # 0: All, 1: Injections (1), 2: Replays (2), 3: All Attacks (>0), 4: Benign (0)
        if self.prediction_mode and "pred" in filtered.columns:
            if self.attacks_only:
                filtered = filtered[filtered["pred"].isin([1, 2])]
            if self.packet_range and "packet_no" in filtered:
                first_packet, last_packet = self.packet_range
                filtered = filtered[
                    filtered["packet_no"].between(first_packet, last_packet)
                ]
        elif "label" in filtered.columns:
            if label_idx == 1:
                filtered = filtered[filtered["label"] == 1]
            elif label_idx == 2:
                filtered = filtered[filtered["label"] == 2]
            elif label_idx == 3:
                filtered = filtered[filtered["label"] > 0]
            elif label_idx == 4:
                filtered = filtered[filtered["label"] == 0]

        query = self.search_input.text().strip().lower()
        if query and not filtered.empty:
            searchable = [
                column for column in filtered.columns
                if column not in ("label", "confidence")
            ]
            mask = filtered[searchable].astype(str).apply(
                lambda column: column.str.lower().str.contains(query, regex=False)
            ).any(axis=1)
            filtered = filtered[mask]

        self.df_filtered = filtered
        self.current_page = 1
        self._render_current_page()

    def _get_total_pages(self) -> int:
        if self.df_filtered.empty or self.page_size == 0:
            return 1
        return max(1, (len(self.df_filtered) + self.page_size - 1) // self.page_size)

    def _on_page_size_changed(self, text: str):
        if text == "All":
            self.page_size = len(self.df_filtered) if not self.df_filtered.empty else 1000
        else:
            self.page_size = int(text)
        self.current_page = 1
        self._render_current_page()

    def _go_first_page(self):
        self.current_page = 1
        self._render_current_page()

    def _go_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._render_current_page()

    def _go_next_page(self):
        if self.current_page < self._get_total_pages():
            self.current_page += 1
            self._render_current_page()

    def _go_last_page(self):
        self.current_page = self._get_total_pages()
        self._render_current_page()

    def _render_current_page(self):
        total_rows = len(self.df_filtered)
        total_pages = self._get_total_pages()
        self.records_badge.setText(f"{total_rows:,} Records")
        self.page_info_label.setText(f"Page {self.current_page} of {total_pages}")
        self.export_btn.setEnabled(total_rows > 0)

        self.btn_prev.setEnabled(self.current_page > 1)
        self.btn_first.setEnabled(self.current_page > 1)
        self.btn_next.setEnabled(self.current_page < total_pages)
        self.btn_last.setEnabled(self.current_page < total_pages)

        self.table.setRowCount(0)
        if self.df_filtered.empty:
            return

        start_idx = (self.current_page - 1) * self.page_size
        end_idx = min(start_idx + self.page_size, total_rows)
        page_df = self.df_filtered.iloc[start_idx:end_idx]

        self.table.setRowCount(len(page_df))
        bold_font = QFont()
        bold_font.setBold(True)

        if self.prediction_mode:
            self._render_prediction_rows(page_df, bold_font)
            return

        for r_idx, (_, row) in enumerate(page_df.iterrows()):
            # 0. Timestamp
            ts_str = f"{row['timestamp']:.3f}"
            item_ts = QTableWidgetItem(ts_str)
            item_ts.setForeground(QBrush(QColor("#94a3b8")))

            # 1. Delta Time (ms)
            dt_val = row["delta_time_ms"]
            item_dt = QTableWidgetItem(f"{dt_val:.1f}")
            if self.highlight_anomalies and dt_val > 150.0:
                item_dt.setForeground(QBrush(QColor("#f43f5e")))  # Latency spike
                item_dt.setFont(bold_font)
            else:
                item_dt.setForeground(QBrush(QColor("#38bdf8")))

            # 2. Message Name
            msg_name = str(row["msg_name"])
            item_msg = QTableWidgetItem(msg_name)
            is_cmd = "COMMAND" in msg_name or "MISSION" in msg_name or row["command_id"] > 0
            if is_cmd:
                item_msg.setForeground(QBrush(QColor("#fbbf24")))  # Amber Command Badge
                item_msg.setFont(bold_font)
            else:
                item_msg.setForeground(QBrush(QColor("#e2e8f0")))

            # 3. Sequence
            seq_val = int(row["seq"])
            item_seq = QTableWidgetItem(str(seq_val))
            item_seq.setForeground(QBrush(QColor("#cbd5e1")))

            # 4. Delta Sequence (Δ Seq)
            dseq_val = int(row["delta_seq"])
            item_dseq = QTableWidgetItem(str(dseq_val))
            if self.highlight_anomalies and dseq_val != 1 and dseq_val != 0:
                item_dseq.setForeground(QBrush(QColor("#ef4444")))  # Red alert for sequence skip/drop
                item_dseq.setFont(bold_font)
                item_dseq.setBackground(QBrush(QColor(60, 20, 20)))
            else:
                item_dseq.setForeground(QBrush(QColor("#34d399")))

            # 5. Sys/Comp ID
            sys_comp = f"{int(row['sys_id'])} / {int(row['comp_id'])}"
            item_sys = QTableWidgetItem(sys_comp)
            item_sys.setForeground(QBrush(QColor("#94a3b8")))

            # 6. Command Name / ID
            cmd_name = str(row["command_name"]) if row["command_name"] else ("-" if row["command_id"] == 0 else f"CMD_{int(row['command_id'])}")
            item_cmd = QTableWidgetItem(cmd_name)
            if row["command_id"] > 0:
                item_cmd.setForeground(QBrush(QColor("#38bdf8")))
                item_cmd.setFont(bold_font)
            else:
                item_cmd.setForeground(QBrush(QColor("#64748b")))

            # 7. Operational Parameters (1-7)
            if is_cmd or row["command_id"] > 0:
                params_str = (
                    f"p1:{row['param1']}, p2:{row['param2']}, p3:{row['param3']}, "
                    f"p4:{row['param4']}, p5:{row['param5']}, p6:{row['param6']}, p7:{row['param7']}"
                )
            else:
                params_str = f"p1:{row['param1']}, p2:{row['param2']}" if row['param1'] != 0 or row['param2'] != 0 else "-"
            item_params = QTableWidgetItem(params_str)
            item_params.setForeground(QBrush(QColor("#cbd5e1")))

            # 8. Label
            lbl_val = int(row["label"])
            lbl_text = {0: "0 (Benign)", 1: "1 (Injection)", 2: "2 (Replay)"}.get(lbl_val, f"{lbl_val} (Unknown)")
            item_lbl = QTableWidgetItem(lbl_text)
            if lbl_val == 1:
                item_lbl.setForeground(QBrush(QColor("#f43f5e")))  # Ruby Injection
                item_lbl.setFont(bold_font)
            elif lbl_val == 2:
                item_lbl.setForeground(QBrush(QColor("#f59e0b")))  # Amber Replay
                item_lbl.setFont(bold_font)
            elif lbl_val == 0:
                item_lbl.setForeground(QBrush(QColor("#10b981")))  # Green Benign
            else:
                item_lbl.setForeground(QBrush(QColor("#94a3b8")))

            # Populate row
            self.table.setItem(r_idx, 0, item_ts)
            self.table.setItem(r_idx, 1, item_dt)
            self.table.setItem(r_idx, 2, item_msg)
            self.table.setItem(r_idx, 3, item_seq)
            self.table.setItem(r_idx, 4, item_dseq)
            self.table.setItem(r_idx, 5, item_sys)
            self.table.setItem(r_idx, 6, item_cmd)
            self.table.setItem(r_idx, 7, item_params)
            self.table.setItem(r_idx, 8, item_lbl)

    def _render_prediction_rows(self, page_df, bold_font):
        for row_index, (_, row) in enumerate(page_df.iterrows()):
            timestamp = float(row.get("timestamp", 0.0) or 0.0)
            command = str(row.get("command_name", "") or "")
            message = str(row.get("msg_name", "") or "")
            msg_command = command or message or str(row.get("msg_id", ""))
            prediction = int(row.get("pred", 0))
            class_name = str(row.get("pred_name", "Normal"))
            values = [
                str(int(row.get("packet_no", 0))),
                f"{timestamp:.3f}",
                msg_command,
                str(int(row.get("seq", 0) or 0)),
                str(int(row.get("delta_seq", 0) or 0)),
                f"{float(row.get('delta_time_ms', 0.0) or 0.0):.1f}",
                class_name,
                f"{float(row.get('confidence', 0.0) or 0.0):.1%}",
                str(int(row.get("sys_id", 0) or 0)) + "/" + str(int(row.get("comp_id", 0) or 0)),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if prediction in (1, 2):
                    color = "#eb6834" if prediction == 1 else "#6250d6"
                    item.setForeground(QBrush(QColor(color)))
                    item.setBackground(QBrush(QColor("#2a1820" if prediction == 1 else "#211d35")))
                    item.setFont(bold_font)
                elif column == 6:
                    item.setForeground(QBrush(QColor("#34d399")))
                self.table.setItem(row_index, column, item)
