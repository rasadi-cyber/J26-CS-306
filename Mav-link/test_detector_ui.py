"""Headless integration coverage for the investigator-facing detector views."""

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PyQt5.QtWidgets import QApplication

from mavlink_core.parser import MAVLinkPCAPParser
from ui.main_window import MainWindow


class DetectorViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_ml_tab_uses_csv_upload_only(self):
        window = MainWindow()
        self.addCleanup(window.close)

        window.page_tabs.setCurrentIndex(1)
        self.assertFalse(hasattr(window, "ml_drop_zone"))
        self.assertFalse(hasattr(window, "btn_open_csv_converter"))
        self.assertTrue(hasattr(window, "csv_drop_zone"))
        self.assertTrue(hasattr(window, "btn_check_csv"))
        self.assertEqual(window.csv_drop_zone.label.text(), "Drop Reader CSV files or a folder here")

    def test_events_packets_batch_verdict_and_compact_layout(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.resize(1360, 900)
        window.show()
        window.page_tabs.setCurrentIndex(1)
        self.app.processEvents()
        capture = os.path.abspath("sample_uav_flight.pcap")
        metadata = MAVLinkPCAPParser.get_file_metadata(capture)
        predictions = pd.DataFrame(
            [
                {
                    "timestamp": 1000.0 + index * 0.1,
                    "delta_time_ms": 100.0,
                    "msg_id": 76,
                    "msg_name": "COMMAND_LONG",
                    "seq": index,
                    "delta_seq": 12,
                    "sys_id": 255,
                    "comp_id": 190,
                    "command_id": 21,
                    "command_name": "MAV_CMD_NAV_LAND",
                    "pred": 1 if index in (2, 3, 4) else 0,
                    "pred_name": "Command Injection" if index in (2, 3, 4) else "Normal",
                    "confidence": 0.92,
                    "packet_no": index + 1,
                    "label": pd.NA,
                }
                for index in range(8)
            ]
        )
        predictions.attrs["attack_events"] = [
            {
                "event_id": "EVT-0001",
                "attack_type": "Command Injection",
                "first_packet_no": 3,
                "last_packet_no": 5,
                "start_time_relative_s": 0.2,
                "end_time_relative_s": 0.4,
                "start_time_utc": "1970-01-01T00:16:40.200000+00:00",
                "end_time_utc": "1970-01-01T00:16:40.400000+00:00",
                "duration_s": 0.2,
                "packet_count": 3,
                "avg_confidence": 0.92,
                "max_confidence": 0.92,
                "source_sys_id": 255,
                "source_comp_id": 190,
                "target_commands": ["MAV_CMD_NAV_LAND x3"],
                "src_ip_port": "192.0.2.1:14550",
                "dst_ip_port": "192.0.2.2:14550",
                "reason": "3 COMMAND_LONG packets with seq jumps up to 12 within 0.2 s",
            }
        ]
        predictions.attrs["ground_truth_metrics"] = None
        result = {
            "path": capture,
            "metadata": metadata,
            "predictions": predictions,
            "error": None,
        }
        window.ml_model_artifact = {
            "model": None,
            "metrics": {
                "macro_f1": 0.875,
                "per_class": {
                    "Normal": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
                    "Command Injection": {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                    "Replay": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                },
                "confusion_matrix": [[9, 1, 0], [1, 7, 2], [0, 2, 8]],
                "feature_importances": {"delta_seq": 0.6, "command_id": 0.4},
            },
            "train_files": ["train_a.csv"],
            "test_files": ["test_b.csv"],
        }
        window._populate_model_performance(
            window.ml_model_artifact["metrics"], window.ml_model_artifact
        )
        window.ml_file_results[capture] = result
        window._populate_files_table([result])
        window._show_file_result(result)
        self.app.processEvents()

        self.assertEqual(window.ml_page.size().width(), 1360)
        self.assertGreater(window.ml_tabs.currentWidget().height(), 300)
        self.assertIn("ATTACK DETECTED", window.verdict_banner.text())
        self.assertIn("SHA-256:", window.capture_metadata_label.text())
        self.assertEqual(window.events_table.rowCount(), 1)
        self.assertEqual(window.files_table.rowCount(), 1)
        self.assertEqual(window.ml_stat_cards["Attack Windows"].value_label.text(), "1")
        self.assertEqual(window.ml_table_widget.table.rowCount(), 8)

        window._on_event_selected(0, 0)
        self.assertEqual(window.ml_tabs.currentIndex(), 1)
        self.assertEqual(window.ml_table_widget.df_filtered["packet_no"].tolist(), [3, 4, 5])
        window.ml_table_widget.set_packet_range()
        window.attacks_only_toggle.setChecked(True)
        self.assertEqual(window.ml_table_widget.df_filtered["packet_no"].tolist(), [3, 4, 5])
        window.attacks_only_toggle.setChecked(False)

        window.ml_tabs.setCurrentIndex(2)
        window.ml_tabs.setCurrentIndex(3)
        window.ml_tabs.setCurrentIndex(4)
        self.app.processEvents()
        for widget in (
            window.verdict_banner,
            window.events_table,
            window.ml_table_widget,
            window.timeline_canvas,
            window.files_table,
            window.btn_export_report,
        ):
            self.assertGreater(widget.width(), 0)
            self.assertGreater(widget.height(), 0)

        with tempfile.TemporaryDirectory() as directory:
            report_path = os.path.join(directory, "report.html")
            with patch(
                "ui.main_window.QFileDialog.getSaveFileName",
                return_value=(report_path, "HTML Report (*.html)"),
            ), patch("ui.main_window.QMessageBox.information"):
                window._export_detector_report()
            csv_path = os.path.join(directory, "report_events.csv")
            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(os.path.isfile(csv_path))
            with open(report_path, encoding="utf-8") as report:
                contents = report.read()
            self.assertIn("data:image/png;base64,", contents)
            self.assertIn("EVT-0001", contents)


if __name__ == "__main__":
    unittest.main()
