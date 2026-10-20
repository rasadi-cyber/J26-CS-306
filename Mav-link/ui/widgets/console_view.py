"""
Forensic Console Log Pane Widget
"""

import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QFrame, QApplication
)
from PyQt5.QtCore import Qt


class ConsoleLogWidget(QFrame):
    """Real-time scrolling terminal output pane for forensics and extraction diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ConsoleFrame")
        self.setStyleSheet("""
            #ConsoleFrame {
                background-color: #080c14;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Header Bar
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self.title_label = QLabel("Forensic Diagnostic Terminal", self)
        self.title_label.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 700; font-family: monospace;")

        self.status_led = QLabel("● LIVE", self)
        self.status_led.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.status_led)
        header_layout.addStretch()

        self.btn_copy = QPushButton("Copy", self)
        self.btn_copy.setObjectName("ConsoleActionButton")
        self.btn_copy.setFixedHeight(26)
        self.btn_copy.clicked.connect(self._copy_logs)

        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setObjectName("ConsoleActionButton")
        self.btn_clear.setFixedHeight(26)
        self.btn_clear.clicked.connect(self.clear)

        header_layout.addWidget(self.btn_copy)
        header_layout.addWidget(self.btn_clear)
        layout.addLayout(header_layout)

        # PlainTextEdit Terminal Log
        self.log_edit = QPlainTextEdit(self)
        self.log_edit.setObjectName("ConsoleLog")
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(2000)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit#ConsoleLog {
                background-color: #06090e;
                color: #38bdf8;
                border: 1px solid #162036;
                border-radius: 4px;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.log_edit)

    def log(self, message: str):
        """Append log message with timestamp."""
        ts = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{ts}] {message}")
        # Auto scroll to bottom
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self.log_edit.clear()

    def _copy_logs(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_edit.toPlainText())
