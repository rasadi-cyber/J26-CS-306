"""Launch the standalone MAVLink CSV attack detector."""

import os
import sys

from PyQt5.QtWidgets import QApplication

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector_app.main_window import DetectorMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("mav link attck Detector")
    window = DetectorMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
