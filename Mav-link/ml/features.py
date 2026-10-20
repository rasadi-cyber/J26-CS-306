"""Shared, leakage-safe MAVLink feature engineering for training and detection."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "delta_time_ms",
    "delta_seq",
    "msg_id",
    "command_id",
    "is_command",
    "seq_jump",
    "fast",
    "param1",
    "param2",
    "param3",
    "param4",
    "param5",
    "param6",
    "param7",
]

CLASS_MAP = {
    0: "Normal",
    1: "Command Injection",
    2: "Replay",
}


def compute_stream_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Sort records chronologically and recalculate per-(sys, comp) dynamics."""
    result = df.copy()
    if "timestamp" not in result:
        result["timestamp"] = np.arange(len(result), dtype=float)
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="coerce").fillna(0.0)

    for column, default in (("seq", 0), ("sys_id", 1), ("comp_id", 1)):
        if column not in result:
            result[column] = default
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(default).astype(int)
    result["seq"] %= 256
    result.sort_values("timestamp", kind="mergesort", inplace=True)

    stream_state = {}
    delta_times = []
    delta_seqs = []
    columns = ["timestamp", "seq", "sys_id", "comp_id"]
    if "file" in result:
        columns.append("file")
    for values in result[columns].itertuples(index=False, name=None):
        timestamp, seq, sys_id, comp_id = values[:4]
        file_name = values[4] if len(values) == 5 else None
        stream = (file_name, int(sys_id), int(comp_id))
        timestamp = float(timestamp)
        seq = int(seq) % 256
        if stream in stream_state:
            previous_time, previous_seq = stream_state[stream]
            delta_times.append(round(max(0.0, timestamp - previous_time) * 1000.0, 3))
            delta_seqs.append((seq - previous_seq) % 256)
        else:
            delta_times.append(0.0)
            delta_seqs.append(1)
        stream_state[stream] = (timestamp, seq)

    result["delta_time_ms"] = delta_times
    result["delta_seq"] = delta_seqs
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed 14-feature matrix; identifiers, labels, and filenames are excluded."""
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=np.float32)

    records = compute_stream_dynamics(df)
    features = pd.DataFrame(index=records.index)
    for column in ("delta_time_ms", "delta_seq", "msg_id", "command_id"):
        if column in records:
            features[column] = pd.to_numeric(records[column], errors="coerce").fillna(0).astype(np.float32)
        else:
            features[column] = np.float32(0)

    features["is_command"] = (features["command_id"] > 0).astype(np.float32)
    signed_jump = ((features["delta_seq"].round().astype(int) + 128) % 256) - 128
    features["seq_jump"] = signed_jump.abs().astype(np.float32)
    features["fast"] = (features["delta_time_ms"] < 50.0).astype(np.float32)
    for parameter in range(1, 8):
        name = f"param{parameter}"
        if name in records:
            features[name] = pd.to_numeric(records[name], errors="coerce").fillna(0).astype(np.float32)
        else:
            features[name] = np.float32(0)

    return (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def extract_features(
    df: pd.DataFrame,
    is_training: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Backward-compatible wrapper around :func:`build_features`."""
    features = build_features(df)
    if not is_training:
        return features
    records = compute_stream_dynamics(df)
    if "label" in records:
        labels = pd.to_numeric(records.loc[features.index, "label"], errors="coerce").fillna(0).astype(int)
    else:
        labels = pd.Series(0, index=features.index, dtype=int)
    return features, labels
