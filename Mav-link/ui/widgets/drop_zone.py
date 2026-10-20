"""
File Ingestion & Drag-and-Drop Zone Widget with Evidence Integrity Display
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class DropZoneWidget(QFrame):
    """Interactive drag & drop ingestion card with metadata and SHA-256 hash."""

    file_selected = pyqtSignal(str)
    files_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.current_filepath = None
        self.setObjectName("DropZoneCard")
        self.setStyleSheet("""
            #DropZoneCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #131d31, stop:1 #0c1424);
                border: 2px dashed #334155;
                border-radius: 10px;
                padding: 16px;
            }
            #DropZoneCard:hover {
                border: 2px dashed #0284c7;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Icon and Title
        self.icon_label = QLabel("📥", self)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 28px;")

        self.title_label = QLabel("Select UAV Packet Log", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")

        self.sub_label = QLabel("Drag & drop PCAP / PCAPNG files here", self)
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setStyleSheet("font-size: 11px; color: #94a3b8;")

        # Browse Button
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        self.browse_btn = QPushButton("Browse Files", self)
        self.browse_btn.setObjectName("SecondaryButton")
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.clicked.connect(self._open_file_dialog)
        btn_layout.addWidget(self.browse_btn)

        # File Metadata Container (Hidden until file selected)
        self.meta_frame = QFrame(self)
        self.meta_frame.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        meta_layout = QVBoxLayout(self.meta_frame)
        meta_layout.setContentsMargins(6, 6, 6, 6)
        meta_layout.setSpacing(4)

        self.filename_label = QLabel("No file loaded", self.meta_frame)
        self.filename_label.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 12px;")
        self.filesize_label = QLabel("Size: --", self.meta_frame)
        self.filesize_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        
        self.hash_label = QLabel("SHA-256: --", self.meta_frame)
        self.hash_label.setWordWrap(True)
        self.hash_label.setStyleSheet("color: #64748b; font-size: 10px; font-family: monospace;")

        meta_layout.addWidget(self.filename_label)
        meta_layout.addWidget(self.filesize_label)
        meta_layout.addWidget(self.hash_label)
        self.meta_frame.setVisible(False)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.sub_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.meta_frame)

    def _open_file_dialog(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select UAV Capture Log(s) (.pcap, .pcapng)",
            "",
            "Network Captures (*.pcap *.pcapng *.cap);;All Files (*)",
        )
        if filepaths:
            self.set_files(filepaths)

    def set_files(self, filepaths, emit_signal=True):
        valid_paths = [
            filepath for filepath in filepaths
            if filepath.lower().endswith((".pcap", ".pcapng", ".cap"))
        ]
        if not valid_paths:
            return
        self.set_file(valid_paths[0], emit_signal=False)
        if emit_signal:
            self.files_selected.emit(valid_paths)
            self.file_selected.emit(valid_paths[0])

    def set_file(self, filepath: str, metadata: dict = None, emit_signal: bool = True):
        """Update widget with loaded file and emit signal."""
        self.current_filepath = filepath
        fname = os.path.basename(filepath)
        fsize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        fsize_str = f"{fsize / (1024*1024):.2f} MB" if fsize > 1024*1024 else f"{fsize / 1024:.2f} KB"

        self.filename_label.setText(f"📄 {fname}")
        self.filesize_label.setText(f"Size: {fsize_str} ({fsize:,} bytes)")
        
        if metadata and "sha256" in metadata:
            self.hash_label.setText(f"SHA-256: {metadata['sha256']}")
        else:
            self.hash_label.setText(f"Path: {filepath}")

        self.meta_frame.setVisible(True)
        self.setStyleSheet("""
            #DropZoneCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #131d31, stop:1 #0c1424);
                border: 2px solid #10b981;
                border-radius: 10px;
                padding: 16px;
            }
        """)
        if emit_signal:
            self.file_selected.emit(filepath)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and any(
                url.toLocalFile().lower().endswith((".pcap", ".pcapng", ".cap"))
                for url in urls
            ):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    #DropZoneCard {
                        background-color: #1e3a8a;
                        border: 2px solid #38bdf8;
                        border-radius: 10px;
                        padding: 16px;
                    }
                """)

    def dragLeaveEvent(self, event):
        if self.current_filepath:
            self.setStyleSheet("""
                #DropZoneCard {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #131d31, stop:1 #0c1424);
                    border: 2px solid #10b981;
                    border-radius: 10px;
                    padding: 16px;
                }
            """)
        else:
            self.setStyleSheet("""
                #DropZoneCard {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #131d31, stop:1 #0c1424);
                    border: 2px dashed #334155;
                    border-radius: 10px;
                    padding: 16px;
                }
            """)

    def dropEvent(self, event: QDropEvent):
        filepaths = [url.toLocalFile() for url in event.mimeData().urls()]
        valid = [path for path in filepaths if path.lower().endswith((".pcap", ".pcapng", ".cap"))]
        if valid:
            self.set_files(valid)
            event.acceptProposedAction()
