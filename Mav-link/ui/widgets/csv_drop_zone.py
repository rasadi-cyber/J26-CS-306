"""Dark-themed drag-and-drop selector for Reader-exported CSV files."""

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout


class CsvDropZoneWidget(QFrame):
    files_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths = []
        self.setAcceptDrops(True)
        self.setObjectName("CsvDropZone")
        self.setStyleSheet(
            "#CsvDropZone { background:#0c1424; border:1px dashed #334155; "
            "border-radius:8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.label = QLabel("Drop Reader CSV files or a folder here", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color:#94a3b8; background:transparent;")
        layout.addWidget(self.label)

    def set_paths(self, paths):
        valid = [
            os.path.abspath(path)
            for path in paths
            if os.path.isfile(path) and path.lower().endswith(".csv")
            or os.path.isdir(path)
        ]
        self.paths = list(dict.fromkeys(valid))
        if self.paths:
            names = [os.path.basename(path.rstrip("\\/")) or path for path in self.paths]
            self.label.setText(f"{len(self.paths)} CSV source(s): " + ", ".join(names))
        else:
            self.label.setText("Drop Reader CSV files or a folder here")
        self.files_selected.emit(self.paths.copy())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if any(
                os.path.isdir(path) or path.lower().endswith(".csv")
                for path in paths
            ):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.set_paths(paths)
        event.acceptProposedAction()
