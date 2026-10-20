"""Headless tests for file-isolated ML training and label-blind PCAP detection."""

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PyQt5.QtWidgets import QApplication
from scapy.all import Ether, IP, UDP, Raw, PcapWriter, wrpcap

from mavlink_core.generator import SyntheticPCAPGenerator
from ml.events import build_attack_events
from ml.features import FEATURE_COLUMNS, build_features
from ml.predictor import predict_pcap
from ml.trainer import ModelTrainer


class _DummyClassifier:
    classes_ = [0, 1, 2]

    def predict_proba(self, features):
        return [[1.0, 0.0, 0.0] for _ in range(len(features))]


class MLPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _write_training_csvs(self, folder):
        for file_index in range(4):
            rows = []
            for index in range(60):
                label = 0 if index < 36 else (1 if index < 50 else 2)
                rows.append(
                    {
                        "timestamp": 1000.0 + file_index * 10 + index * 0.02,
                        "msg_id": 76 if label == 1 else (30 if label == 2 else 0),
                        "msg_name": "COMMAND_LONG" if label == 1 else "ATTITUDE",
                        "seq": (index * (17 if label else 1)) % 256,
                        "sys_id": 255 if label == 1 else 1,
                        "comp_id": 190 if label == 1 else 1,
                        "command_id": 21 if label == 1 else 0,
                        "command_name": "MAV_CMD_NAV_LAND" if label == 1 else "",
                        "param1": float(label),
                        "param2": 0.0,
                        "param3": 0.0,
                        "param4": 0.0,
                        "param5": 0.0,
                        "param6": 0.0,
                        "param7": 0.0,
                        "label": label,
                    }
                )
            pd.DataFrame(rows).to_csv(
                os.path.join(folder, f"reader_capture_{file_index}.csv"),
                index=False,
            )

    def test_build_features_sorts_and_excludes_identifiers(self):
        records = pd.DataFrame(
            [
                {"timestamp": 2.0, "seq": 2, "sys_id": 1, "comp_id": 1, "msg_id": 5, "label": 1, "file": "a"},
                {"timestamp": 1.0, "seq": 250, "sys_id": 1, "comp_id": 1, "msg_id": 5, "label": 0, "file": "a"},
                {"timestamp": 3.0, "seq": 1, "sys_id": 1, "comp_id": 1, "msg_id": 5, "label": 0, "file": "a"},
            ]
        )
        features = build_features(records)
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features["delta_seq"].tolist(), [1.0, 8.0, 255.0])
        self.assertEqual(features["delta_time_ms"].tolist(), [0.0, 1000.0, 1000.0])
        self.assertFalse({"timestamp", "seq", "sys_id", "comp_id", "label", "file"} & set(features.columns))

    def test_train_smote_split_by_file_and_predict_without_ground_truth_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_folder = os.path.join(directory, "dataset")
            os.makedirs(dataset_folder)
            self._write_training_csvs(dataset_folder)

            trainer = ModelTrainer(
                dataset_folder,
                model_type="Random Forest",
                use_smote=True,
                model_path=os.path.join(directory, "model.joblib"),
            )
            completed = []
            failed = []
            trainer.training_completed.connect(completed.append)
            trainer.training_failed.connect(failed.append)
            trainer.start()
            self.assertTrue(trainer.wait(120000))
            self.app.processEvents()
            self.assertFalse(failed, failed)
            self.assertEqual(len(completed), 1)
            artifact = completed[0]
            self.assertTrue(os.path.isfile(trainer.model_path))
            self.assertEqual(
                set(artifact),
                {"model", "features", "metrics", "train_files", "test_files", "trained_at"},
            )
            self.assertFalse(set(artifact["train_files"]) & set(artifact["test_files"]))
            self.assertTrue(artifact["train_files"])
            self.assertTrue(artifact["test_files"])
            for side_files in (artifact["train_files"], artifact["test_files"]):
                self.assertTrue(
                    any(
                        pd.read_csv(os.path.join(dataset_folder, filename))["label"].isin([1, 2]).any()
                        for filename in side_files
                    )
                )
            self.assertTrue(artifact["metrics"]["macro_f1"] >= 0.0)
            self.assertEqual(len(artifact["metrics"]["confusion_matrix"]), 3)

            pcap_path = os.path.join(directory, "mixed_capture.pcap")
            SyntheticPCAPGenerator.create_uav_sample_pcap(
                pcap_path,
                num_packets=120,
                attack_scenario="mixed",
                base_time=1_617_778_618.0,
            )
            with patch(
                "mavlink_core.parser.load_sidecar_ground_truth",
                side_effect=AssertionError("Detector must not read labels during parsing."),
            ) as sidecar_loader:
                first = predict_pcap(pcap_path, trainer.model_path)
                self.assertFalse(sidecar_loader.called)
            self.assertGreater(len(first), 0)
            self.assertTrue(
                {"packet_no", "pred", "pred_name", "confidence"}.issubset(first.columns)
            )
            self.assertEqual(first["packet_no"].tolist(), list(range(1, len(first) + 1)))
            self.assertTrue(first["label"].isna().all())
            self.assertFalse(first.attrs["ground_truth_used_for_prediction"])
            self.assertIsNotNone(first.attrs["ground_truth_metrics"])

            ground_truth_path = os.path.splitext(pcap_path)[0] + "_ground_truth.csv"
            ground_truth = pd.read_csv(ground_truth_path)
            ground_truth["label"] = 2
            ground_truth.to_csv(ground_truth_path, index=False)
            with patch(
                "mavlink_core.parser.load_sidecar_ground_truth",
                side_effect=AssertionError("Detector must not read labels during parsing."),
            ) as sidecar_loader:
                second = predict_pcap(pcap_path, trainer.model_path)
                self.assertFalse(sidecar_loader.called)
            pd.testing.assert_series_equal(first["pred"], second["pred"])
            self.assertEqual(first["packet_no"].tolist(), second["packet_no"].tolist())

    def test_training_rejects_single_class_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                pd.DataFrame(
                    [{"timestamp": float(index), "msg_id": 0, "label": 0}]
                ).to_csv(os.path.join(directory, f"normal_{index}.csv"), index=False)
            worker = ModelTrainer(directory, model_path=os.path.join(directory, "model.joblib"))
            failed = []
            worker.training_failed.connect(failed.append)
            worker.start()
            self.assertTrue(worker.wait(120000))
            self.app.processEvents()
            self.assertEqual(len(failed), 1)
            self.assertIn("single-class", failed[0])

    def test_events_require_three_packets_and_group_by_attack_type(self):
        rows = pd.DataFrame(
            [
                {"timestamp": 100.0, "pred": 1, "packet_no": 1, "confidence": 0.8, "sys_id": 255, "comp_id": 190, "command_id": 21, "command_name": "MAV_CMD_NAV_LAND", "msg_name": "COMMAND_LONG", "seq_jump": 8},
                {"timestamp": 100.2, "pred": 1, "packet_no": 2, "confidence": 0.9, "sys_id": 255, "comp_id": 190, "command_id": 21, "command_name": "MAV_CMD_NAV_LAND", "msg_name": "COMMAND_LONG", "seq_jump": 9},
                {"timestamp": 100.4, "pred": 1, "packet_no": 3, "confidence": 0.95, "sys_id": 255, "comp_id": 190, "command_id": 400, "command_name": "MAV_CMD_COMPONENT_ARM_DISARM", "msg_name": "COMMAND_LONG", "seq_jump": 10},
                {"timestamp": 101.0, "pred": 2, "packet_no": 4, "confidence": 0.7, "sys_id": 1, "comp_id": 1, "command_id": 0, "command_name": "", "msg_name": "ATTITUDE", "seq_jump": 2},
                {"timestamp": 101.2, "pred": 2, "packet_no": 5, "confidence": 0.75, "sys_id": 1, "comp_id": 1, "command_id": 0, "command_name": "", "msg_name": "ATTITUDE", "seq_jump": 2},
            ]
        )
        events = build_attack_events(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["attack_type"], "Command Injection")
        self.assertEqual(events[0]["packet_count"], 3)
        self.assertEqual(events[0]["first_packet_no"], 1)
        self.assertEqual(events[0]["last_packet_no"], 3)
        self.assertEqual(events[0]["target_commands"], ["MAV_CMD_NAV_LAND x2", "MAV_CMD_COMPONENT_ARM_DISARM x1"])
        self.assertIn("sys 255/comp 190", events[0]["reason"])

    def test_predict_rejects_empty_and_non_mavlink_captures(self):
        artifact = {"model": _DummyClassifier(), "features": FEATURE_COLUMNS}
        with tempfile.TemporaryDirectory() as directory:
            empty_path = os.path.join(directory, "empty.pcap")
            with open(empty_path, "wb"):
                pass
            with self.assertRaisesRegex(ValueError, "empty"):
                predict_pcap(empty_path, artifact)

            zero_packet_path = os.path.join(directory, "zero_packet.pcap")
            PcapWriter(zero_packet_path, linktype=1, sync=True).close()
            with self.assertRaisesRegex(ValueError, "no packets"):
                predict_pcap(zero_packet_path, artifact)

            no_mavlink_path = os.path.join(directory, "no_mavlink.pcap")
            wrpcap(
                no_mavlink_path,
                [Ether() / IP(src="192.0.2.1", dst="192.0.2.2") / UDP(sport=14550, dport=14550) / Raw(load=b"not mavlink")],
            )
            with self.assertRaisesRegex(ValueError, "no valid MAVLink packets"):
                predict_pcap(no_mavlink_path, artifact)


if __name__ == "__main__":
    unittest.main()
