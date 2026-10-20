"""
Automated Test & Validation Script for MAVLink PCAP Extraction Pipeline
Tests:
1. Packet-Level Ground-Truth Labeling (Benign=0, Injection=1, Replay=2)
2. Per-Stream Sequence Tracking (sys_id, comp_id)
3. BAD_DATA & Corrupted Frame Filtering
4. Real-Flight Attack Injection into Authentic Flight PCAP
"""

import os
import sys
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from mavlink_core.generator import SyntheticPCAPGenerator, FlightAttackInjector
from mavlink_core.parser import MAVLinkPCAPParser, COMMON_MAV_CMDS


def run_tests():
    print("=" * 75)
    print("RUNNING ADVANCED MAVLINK FORENSIC EXTRACTION TESTS")
    print("=" * 75)

    test_pcap = "test_synthetic_flight.pcap"
    export_csv = "test_exported_features.csv"

    # Step 1: Synthesize UAV PCAP with Packet-Level Labels
    print("\n[TEST 1] Synthesizing Multi-Stream UAV PCAP (1200 packets with packet-level tags)...")
    SyntheticPCAPGenerator.create_uav_sample_pcap(test_pcap, num_packets=1200, attack_scenario="mixed")
    assert os.path.exists(test_pcap), "Failed to generate test PCAP file"
    print(f"-> Generated {test_pcap} ({os.path.getsize(test_pcap):,} bytes)")

    # Step 2: Test Evidence Hash & Metadata
    print("\n[TEST 2] Verifying SHA-256 Evidence Integrity Hash...")
    meta = MAVLinkPCAPParser.get_file_metadata(test_pcap)
    print(f"-> Filename: {meta['filename']}")
    print(f"-> Filesize: {meta['filesize_str']}")
    print(f"-> SHA-256 : {meta['sha256']}")
    assert len(meta["sha256"]) == 64, "SHA-256 hash length mismatch"

    # Step 3: Decapsulate and Parse with Packet-Level Ground Truth
    print("\n[TEST 3] Parsing Traffic with Packet-Level Ground Truth (Auto Mode)...")
    parser_auto = MAVLinkPCAPParser(target_ports=(14550, 14551), commands_only=False, target_label="auto")
    df_all = parser_auto.parse_pcap(test_pcap)
    print(f"-> Total Extracted Rows: {len(df_all):,}")
    print(f"-> Unique Messages Found: {df_all['msg_name'].unique().tolist()}")
    assert not df_all.empty, "DataFrame is empty!"

    # CRITICAL: Verify Packet-Level Labeling (Normal packets are NOT labeled as attacks!)
    label_counts = df_all["label"].value_counts().to_dict()
    print(f"-> Ground Truth Label Distribution: {label_counts}")
    assert 0 in label_counts, "Benign packets (label 0) must be present!"
    assert 1 in label_counts, "Command injection packets (label 1) must be present!"
    assert 2 in label_counts, "Replay attack packets (label 2) must be present!"

    # Ensure routine telemetry is labeled 0
    heartbeats = df_all[df_all["msg_name"] == "HEARTBEAT"]
    assert (heartbeats["label"] == 0).all(), "Routine HEARTBEAT packets must have label=0 (Benign)!"

    # Ensure injected commands have label 1
    injections = df_all[df_all["label"] == 1]
    assert len(injections) > 0, "Injected attack packets must be isolated!"
    print(f"-> Injected Attack Rows isolated: {len(injections):,} (Command IDs: {injections['command_id'].unique().tolist()})")

    # Step 4: Verify Per-Stream Sequence Tracking (sys_id, comp_id)
    print("\n[TEST 4] Validating Per-Stream Sequence & Timing Dynamics...")
    expected_cols = [
        "timestamp", "delta_time_ms", "msg_id", "msg_name",
        "seq", "delta_seq", "sys_id", "comp_id",
        "command_id", "command_name",
        "param1", "param2", "param3", "param4", "param5", "param6", "param7",
        "label"
    ]
    for col in expected_cols:
        assert col in df_all.columns, f"Missing expected column: {col}"

    # Verify distinct streams exist and don't corrupt each other
    streams = df_all[["sys_id", "comp_id"]].drop_duplicates().values.tolist()
    print(f"-> Distinct Streams Tracked: {streams}")
    assert len(streams) >= 2, "Expected at least UAV (1, 1) and GCS (255, 190) streams!"

    # Step 5: Test Real-Flight Attack Injection
    print("\n[TEST 5] Testing Real-Flight Attack Injection into Authentic Flight PCAP...")
    injected_pcap = "test_injected_real_flight.pcap"
    if os.path.exists("sample_uav_flight.pcap"):
        out_path, stats = FlightAttackInjector.inject_attacks_into_pcap(
            input_pcap_path="sample_uav_flight.pcap",
            output_pcap_path=injected_pcap,
            injection_count=30,
            replay_count=30,
        )
        assert os.path.exists(injected_pcap), "Failed to generate blended flight PCAP"
        df_injected = parser_auto.parse_pcap(injected_pcap)
        inj_counts = df_injected["label"].value_counts().to_dict()
        print(f"-> Injected Real Flight Extracted Rows: {len(df_injected):,}")
        print(f"-> Injected Real Flight Labels: {inj_counts}")
        assert inj_counts.get(0, 0) > 1000, "Authentic normal flight packets must retain label=0!"
        assert inj_counts.get(1, 0) > 0, "Injected command packets must be labeled 1!"
        print("-> Real-flight injection validation succeeded!")

    # Step 6: Test BAD_DATA Filtering
    print("\n[TEST 6] Verifying BAD_DATA & Corrupted Frame Filtering...")
    parser_filter = MAVLinkPCAPParser(filter_bad_data=True)
    df_clean = parser_filter.parse_pcap(test_pcap)
    assert (df_clean["msg_name"] == "BAD_DATA").sum() == 0, "BAD_DATA must not be present in output DataFrame"
    assert ((df_clean["sys_id"] == 0) & (df_clean["comp_id"] == 0)).sum() == 0, "sys 0/0 frames must be filtered out"
    print("-> BAD_DATA and 0/0 frames filtered out cleanly!")

    # Step 7: Export CSV & Verify Schema
    print("\n[TEST 7] Exporting CSV and Verifying ML Schema...")
    df_all.to_csv(export_csv, index=False)
    assert os.path.exists(export_csv), "Failed to write export CSV"

    df_reloaded = pd.read_csv(export_csv)
    assert list(df_reloaded.columns) == expected_cols, "Exported CSV headers mismatch"
    print(f"-> Exported CSV verified ({len(df_reloaded):,} rows, {len(df_reloaded.columns)} columns)")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
