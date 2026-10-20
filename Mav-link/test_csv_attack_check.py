"""Headless coverage for CSV-based attack detection."""

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PyQt5.QtWidgets import QApplication

from ml.csv_predictor import CsvAttackCheckWorker
from ml.features import FEATURE_COLUMNS
from ui.main_window import MainWindow


class _CommandRuleClassifier:
    classes_ = [0, 1, 2]

    def predict_proba(self, features):
        self.feature_columns = list(features.columns)
        probabilities = []
        for command_id in features["command_id"]:
            probabilities.append([0.02, 0.96, 0.02] if command_id == 21 else [0.96, 0.02, 0.02])
        return probabilities


class CsvAttackCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _csv_frame(command_id, label):
        return pd.DataFrame(
            [
                {
                    "timestamp": 1_700_000_000.0 + index * 0.1,
                    "msg_id": 76 if command_id else 0,
                    "msg_name": "COMMAND_LONG" if command_id else "HEARTBEAT",
                    "seq": index if not command_id else (index * 17) % 256,
                    "sys_id": 255,
                    "comp_id": 190,
                    "command_id": command_id,
                    "command_name": "MAV_CMD_NAV_LAND" if command_id else "",
                    "param1": 0.0,
                    "param2": 0.0,
                    "param3": 0.0,
                    "param4": 0.0,
                    "param5": 0.0,
                    "param6": 0.0,
                    "param7": 0.0,
                    "label": label,
                }
                for index in range(6)
            ]
        )

    def test_clean_and_attack_csv_verdicts_and_events(self):
        classifier = _CommandRuleClassifier()
        model = {"model": classifier, "features": FEATURE_COLUMNS}
        inputs = [
            ("clean.csv", self._csv_frame(0, 0)),
            ("attack.csv", self._csv_frame(21, 1)),
        ]
        worker = CsvAttackCheckWorker(inputs, model)
        completed = []
        failed = []
        worker.check_completed.connect(completed.append)
        worker.check_failed.connect(failed.append)
        worker.start()
        self.assertTrue(worker.wait(30000))
        self.app.processEvents()
        self.assertFalse(failed, failed)
        self.assertEqual(len(completed), 1)
        results = completed[0]
        clean, attack = [item["predictions"] for item in results]

        self.assertEqual(clean.attrs["verdict"], "Clean - no attack detected")
        self.assertEqual(clean.attrs["attack_events"], [])
        self.assertTrue(attack.attrs["verdict"].startswith("ATTACK DETECTED: Command Injection"))
        self.assertGreaterEqual(len(attack.attrs["attack_events"]), 1)
        self.assertEqual(attack.attrs["attack_events"][0]["packet_count"], 6)
        self.assertEqual(attack.attrs["attack_events"][0]["max_seq_jump"], 17)
        self.assertEqual(
            attack.attrs["score_vs_label"],
            {
                "true_attack_packets": 6,
                "caught_attack_packets": 6,
                "false_alarm_packets": 0,
            },
        )
        self.assertEqual(classifier.feature_columns, FEATURE_COLUMNS)
        self.assertFalse({"timestamp", "seq", "sys_id", "comp_id", "label"} & set(classifier.feature_columns))

        window = MainWindow()
        self.addCleanup(window.close)
        window.ml_model_artifact = {"model": classifier, "features": FEATURE_COLUMNS, "metrics": {}}
        window._on_csv_check_completed(results)
        attack_result = next(
            result for result in window.ml_file_results.values()
            if result["name"] == "attack.csv"
        )
        window._show_file_result(attack_result)
        self.assertIn("ATTACK DETECTED: Command Injection", window.verdict_banner.text())
        self.assertIn("Score vs label", window.verdict_banner.text())
        self.assertEqual(window.events_table.rowCount(), 1)
        self.assertEqual(window.files_table.rowCount(), 2)
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = os.path.join(directory, "predictions.csv")
            with patch(
                "ui.main_window.QFileDialog.getSaveFileName",
                return_value=(predictions_path, "CSV Files (*.csv)"),
            ), patch("ui.main_window.QMessageBox.information"):
                window._export_predictions_csv()
            exported = pd.read_csv(predictions_path)
            self.assertTrue(
                {"source_file", "label", "pred", "confidence"}.issubset(exported.columns)
            )
            self.assertEqual(len(exported), 12)
        clean_result = next(
            result for result in window.ml_file_results.values()
            if result["name"] == "clean.csv"
        )
        window._show_file_result(clean_result)
        self.assertIn("Clean - no attack detected", window.verdict_banner.text())

    def test_missing_reader_columns_are_reported(self):
        from ml.csv_predictor import predict_csv

        model = {"model": _CommandRuleClassifier(), "features": FEATURE_COLUMNS}
        with self.assertRaisesRegex(ValueError, "missing required Reader columns"):
            predict_csv(pd.DataFrame([{"timestamp": 1.0}]), model)


if __name__ == "__main__":
    unittest.main()
