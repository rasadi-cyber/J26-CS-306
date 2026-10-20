"""Feature engineering shared by detector training and CSV inference."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
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

CLASS_COLORS = {
    0: "#2a78d6",
    1: "#eb6834",
    2: "#6250d6",
}

REQUIRED_COLUMNS = [
    "timestamp",
    "delta_time_ms",
    "delta_seq",
    "msg_id",
    "msg_name",
    "command_id",
    "command_name",
    "sys_id",
    "comp_id",
    "seq",
    *(f"param{index}" for index in range(1, 8)),
]


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed feature matrix; identifiers and labels are never features."""
    features = pd.DataFrame(index=records.index)
    for column in ("delta_time_ms", "delta_seq", "msg_id", "command_id"):
        features[column] = pd.to_numeric(
            records[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    features["is_command"] = (features["command_id"] > 0).astype(np.float32)
    signed_delta = ((features["delta_seq"].round().astype(int) + 128) % 256) - 128
    features["seq_jump"] = signed_delta.abs().astype(np.float32)
    features["fast"] = (features["delta_time_ms"] < 50.0).astype(np.float32)
    for index in range(1, 8):
        column = f"param{index}"
        features[column] = pd.to_numeric(
            records[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return features[FEATURE_COLUMNS].astype(np.float32)
