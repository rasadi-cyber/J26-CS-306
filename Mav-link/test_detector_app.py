"""Headless tests for the independent CSV-only detector."""

import os
import tempfile
import unittest

import pandas as pd

from detector_app.features import FEATURE_COLUMNS
from detector_app.pipeline import (
    BAD_DATA_MESSAGE,
    predict_csv_frame,
    read_reader_csv,
    validate_csv_path,
)


class CommandBasedClassifier:
    classes_ = [0, 1, 2]

    def predict_proba(self, features):
        probabilities = []
        for command in features["command_id"]:
            if command > 0:
                probabilities.append([0.02, 0.96, 0.02])
            else:
                probabilities.append([0.96, 0.02, 0.02])
        return probabilities


def make_csv_rows(commands):
    rows = []
    for index, command_id in enumerate(commands):
        rows.append(
            {
                "timestamp": 1_700_000_000.0 + index * 0.25,
                "delta_time_ms": 250.0,
                "delta_seq": 1 if index == 0 else 8,
                "msg_id": 76 if command_id else 30,
                "msg_name": "COMMAND_LONG" if command_id else "ATTITUDE",
                "command_id": command_id,
                "command_name": "MAV_CMD_NAV_LAND" if command_id else "",
                "sys_id": 255 if command_id else 1,
                "comp_id": 190 if command_id else 1,
                "seq": index,
                **{f"param{parameter}": 0.0 for parameter in range(1, 8)},
            }
        )
    return pd.DataFrame(rows)


class DetectorCsvTests(unittest.TestCase):
    def setUp(self):
        self.artifact = {
            "model": CommandBasedClassifier(),
            "features": FEATURE_COLUMNS,
            "metrics": {"macro_f1": 0.8},
        }

    def test_clean_csv_is_clean(self):
        result = predict_csv_frame(make_csv_rows([0, 0, 0, 0]), self.artifact)
        self.assertEqual(result["verdict"], "Clean - no attack detected")
        self.assertEqual(result["events"], [])

    def test_attack_csv_has_event(self):
        result = predict_csv_frame(make_csv_rows([0, 21, 21, 21, 21]), self.artifact)
        self.assertTrue(result["events"])
        self.assertIn("ATTACK DETECTED", result["verdict"])
        self.assertGreaterEqual(result["events"][0]["packet_count"], 3)

    def test_pcap_is_rejected_with_reader_conversion_message(self):
        with self.assertRaisesRegex(ValueError, BAD_DATA_MESSAGE):
            validate_csv_path("capture.pcap")
        with self.assertRaisesRegex(ValueError, BAD_DATA_MESSAGE):
            validate_csv_path("capture.pcapng")

    def test_missing_columns_are_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "incomplete.csv")
            pd.DataFrame([{"timestamp": 1.0, "label": 0}]).to_csv(path, index=False)
            with self.assertRaises(ValueError) as context:
                read_reader_csv(path)
            self.assertIn("Missing required columns:", str(context.exception))
            self.assertIn("delta_time_ms", str(context.exception))
            self.assertIn("param7", str(context.exception))

    def test_bad_data_is_dropped_before_prediction(self):
        rows = make_csv_rows([0, 0, 0, 0])
        rows.loc[0, "msg_name"] = "BAD_DATA"
        result = predict_csv_frame(rows, self.artifact)
        self.assertEqual(result["bad_data"], 1)
        self.assertEqual(len(result["predictions"]), 3)

    def test_detector_modules_do_not_import_pcap_libraries(self):
        root = os.path.dirname(__file__)
        for filename in os.listdir(os.path.join(root, "detector_app")):
            if filename.endswith(".py"):
                with open(os.path.join(root, "detector_app", filename), encoding="utf-8") as source:
                    text = source.read()
                self.assertNotIn("import scapy", text)
                self.assertNotIn("import pymavlink", text)


if __name__ == "__main__":
    unittest.main()
