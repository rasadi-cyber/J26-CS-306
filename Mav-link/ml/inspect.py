"""
Dataset Ingestion and Forensic Quality Inspection Module.
Performs pre-flight dataset validation, class balance profiling, timestamp sorting checks,
and corrupted BAD_DATA frame auditing for MAVLink network traces.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FileProfile:
    filename: str
    rows: int
    labels_present: List[int]
    is_sorted_timestamp: bool
    bad_data_count: int
    missing_count: int
    inf_count: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetInspectionResult:
    total_files: int
    total_rows: int
    total_cols: int
    class_counts: Dict[int, int]
    class_percentages: Dict[int, float]
    per_class_median_delta_seq: Dict[int, float]
    per_class_median_delta_time_ms: Dict[int, float]
    total_missing: int
    total_inf: int
    total_duplicates: int
    total_bad_data: int
    file_profiles: List[FileProfile]
    global_warnings: List[str]
    formatted_report: str


def load_csv_folder(folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all .csv files from the target directory and attaches a 'file' column.
    Returns a dictionary mapping {filename: DataFrame}.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dataset folder does not exist: {folder_path}")

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {folder_path}")

    loaded_dfs: Dict[str, pd.DataFrame] = {}
    for fname in sorted(csv_files):
        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df["file"] = fname
            loaded_dfs[fname] = df
        except Exception as e:
            # Skip unreadable or corrupted files with notice
            print(f"[WARN] Failed to read {fname}: {e}")

    return loaded_dfs


def inspect_dataset(dfs: Dict[str, pd.DataFrame]) -> DatasetInspectionResult:
    """
    Analyzes loaded CSVs and produces a forensic dataset inspection report.
    Checks:
    - Rows, columns
    - Class balance (Normal 0, Injection 1, Replay 2)
    - Labels per file (warns if file is single-class)
    - Missing / inf / duplicate / BAD_DATA counts
    - Per-class median delta_seq and delta_time_ms
    - Timestamp sorting verification
    """
    if not dfs:
        raise ValueError("No DataFrames provided for dataset inspection.")

    combined_df = pd.concat(dfs.values(), ignore_index=True)
    total_files = len(dfs)
    total_rows = len(combined_df)
    total_cols = combined_df.shape[1]

    # Global data cleanliness
    total_missing = int(combined_df.isna().sum().sum())
    
    # Check numeric infs
    num_cols = combined_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        total_inf = int(np.isinf(combined_df[num_cols].to_numpy()).sum())
    else:
        total_inf = 0

    # Duplicates across all non-file columns
    content_cols = [c for c in combined_df.columns if c != "file"]
    total_duplicates = int(combined_df.duplicated(subset=content_cols).sum())

    # BAD_DATA count: sys_id==0 & comp_id==0 or msg_name=='BAD_DATA'
    bad_data_mask = pd.Series(False, index=combined_df.index)
    if "sys_id" in combined_df.columns and "comp_id" in combined_df.columns:
        sys_0 = pd.to_numeric(combined_df["sys_id"], errors="coerce").fillna(0) == 0
        comp_0 = pd.to_numeric(combined_df["comp_id"], errors="coerce").fillna(0) == 0
        bad_data_mask = bad_data_mask | (sys_0 & comp_0)
    if "msg_name" in combined_df.columns:
        bad_data_mask = bad_data_mask | (combined_df["msg_name"].astype(str) == "BAD_DATA")
    total_bad_data = int(bad_data_mask.sum())

    # Class balance
    class_counts = {0: 0, 1: 0, 2: 0}
    if "label" in combined_df.columns:
        counts = pd.to_numeric(combined_df["label"], errors="coerce").fillna(0).astype(int).value_counts().to_dict()
        for k, v in counts.items():
            if k in class_counts:
                class_counts[k] = v
            else:
                class_counts[k] = v

    class_percentages = {}
    for k, v in class_counts.items():
        class_percentages[k] = round((v / total_rows * 100.0) if total_rows > 0 else 0.0, 2)

    # Per-class median delta_seq and delta_time_ms
    per_class_median_delta_seq = {}
    per_class_median_delta_time_ms = {}

    for k in (0, 1, 2):
        if "label" in combined_df.columns:
            subset = combined_df[pd.to_numeric(combined_df["label"], errors="coerce").fillna(0).astype(int) == k]
        else:
            subset = combined_df

        if not subset.empty:
            if "delta_seq" in subset.columns:
                per_class_median_delta_seq[k] = round(float(pd.to_numeric(subset["delta_seq"], errors="coerce").median()), 2)
            else:
                per_class_median_delta_seq[k] = 1.0

            if "delta_time_ms" in subset.columns:
                per_class_median_delta_time_ms[k] = round(float(pd.to_numeric(subset["delta_time_ms"], errors="coerce").median()), 2)
            else:
                per_class_median_delta_time_ms[k] = 0.0
        else:
            per_class_median_delta_seq[k] = 0.0
            per_class_median_delta_time_ms[k] = 0.0

    # Per-file profiles and single-class checks
    file_profiles: List[FileProfile] = []
    global_warnings: List[str] = []

    for fname, df in dfs.items():
        f_rows = len(df)
        f_missing = int(df.isna().sum().sum())
        f_num_cols = df.select_dtypes(include=[np.number]).columns
        f_inf = int(np.isinf(df[f_num_cols].to_numpy()).sum()) if len(f_num_cols) > 0 else 0

        # Bad data in this file
        f_bad = 0
        if "sys_id" in df.columns and "comp_id" in df.columns:
            s0 = pd.to_numeric(df["sys_id"], errors="coerce").fillna(0) == 0
            c0 = pd.to_numeric(df["comp_id"], errors="coerce").fillna(0) == 0
            f_bad += int((s0 & c0).sum())
        if "msg_name" in df.columns:
            f_bad += int((df["msg_name"].astype(str) == "BAD_DATA").sum())

        # Check timestamp monotonicity
        is_sorted = True
        if "timestamp" in df.columns:
            ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            is_sorted = bool(ts.is_monotonic_increasing)

        # Labels present
        f_labels = []
        if "label" in df.columns:
            f_labels = sorted(pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).unique().tolist())

        f_warnings = []
        if len(f_labels) <= 1:
            lbl_name = {0: "Normal", 1: "Command Injection", 2: "Replay"}.get(f_labels[0] if f_labels else 0, "Unknown")
            warn_msg = f"Single-class file: '{fname}' only contains class {f_labels} ({lbl_name}). Split by flight carefully."
            f_warnings.append(warn_msg)
            global_warnings.append(warn_msg)

        if not is_sorted:
            f_warnings.append(f"File '{fname}' timestamps not monotonically increasing. Will sort by timestamp prior to feature extraction.")

        if f_bad > 0:
            f_warnings.append(f"File '{fname}' contains {f_bad} BAD_DATA / uninitialized (0/0) frames.")

        file_profiles.append(FileProfile(
            filename=fname,
            rows=f_rows,
            labels_present=f_labels,
            is_sorted_timestamp=is_sorted,
            bad_data_count=f_bad,
            missing_count=f_missing,
            inf_count=f_inf,
            warnings=f_warnings,
        ))

    # Build ASCII formatted report
    report_lines = [
        "═════════════════════════════════════════════════════════════════════════",
        "                DATASET FORENSIC QUALITY INSPECTION REPORT               ",
        "═════════════════════════════════════════════════════════════════════════",
        f"  Total Ingested Files  : {total_files}",
        f"  Total Extracted Rows  : {total_rows:,}",
        f"  Feature Columns       : {total_cols}",
        "",
        "── CLASS BALANCE & DISTRIBUTION ──────────────────────────────────────────",
        f"  [0] Normal (Benign)   : {class_counts.get(0, 0):>8,} pkts ({class_percentages.get(0, 0.0):>6.2f}%)",
        f"  [1] Command Injection : {class_counts.get(1, 0):>8,} pkts ({class_percentages.get(1, 0.0):>6.2f}%)",
        f"  [2] Replay Attack     : {class_counts.get(2, 0):>8,} pkts ({class_percentages.get(2, 0.0):>6.2f}%)",
        "",
        "── STATEFUL DYNAMICS MEDIANS (Per Class) ──────────────────────────────────",
        f"  Class 0 (Normal)      : Median ΔSeq = {per_class_median_delta_seq.get(0, 0.0):<5} | Median Δt = {per_class_median_delta_time_ms.get(0, 0.0):<7} ms",
        f"  Class 1 (Injection)   : Median ΔSeq = {per_class_median_delta_seq.get(1, 0.0):<5} | Median Δt = {per_class_median_delta_time_ms.get(1, 0.0):<7} ms",
        f"  Class 2 (Replay)      : Median ΔSeq = {per_class_median_delta_seq.get(2, 0.0):<5} | Median Δt = {per_class_median_delta_time_ms.get(2, 0.0):<7} ms",
        "",
        "── DATA QUALITY & CLEANLINESS ───────────────────────────────────────────",
        f"  Missing / NaN Cells   : {total_missing:<8,} (Imputed during feature extraction)",
        f"  Infinity (±Inf) Cells : {total_inf:<8,}",
        f"  Duplicate Rows        : {total_duplicates:<8,}",
        f"  BAD_DATA / Sys 0/0    : {total_bad_data:<8,} (Corrupted frames)",
        "",
        "── FILE-BY-FILE AUDIT ───────────────────────────────────────────────────",
    ]

    for p in file_profiles:
        labels_str = ",".join(str(l) for l in p.labels_present) if p.labels_present else "None"
        sort_status = "OK (Sorted)" if p.is_sorted_timestamp else "UNSORTED (Auto-sorted)"
        report_lines.append(
            f"  • {p.filename:<30} | {p.rows:>6,} rows | Labels: [{labels_str}] | Time: {sort_status}"
        )

    if global_warnings:
        report_lines.append("")
        report_lines.append("── AUDIT WARNINGS & RECOMMENDATIONS ────────────────────────────────────")
        for w in global_warnings:
            report_lines.append(f"  ⚠️  {w}")

    report_lines.append("═════════════════════════════════════════════════════════════════════════")

    return DatasetInspectionResult(
        total_files=total_files,
        total_rows=total_rows,
        total_cols=total_cols,
        class_counts=class_counts,
        class_percentages=class_percentages,
        per_class_median_delta_seq=per_class_median_delta_seq,
        per_class_median_delta_time_ms=per_class_median_delta_time_ms,
        total_missing=total_missing,
        total_inf=total_inf,
        total_duplicates=total_duplicates,
        total_bad_data=total_bad_data,
        file_profiles=file_profiles,
        global_warnings=global_warnings,
        formatted_report="\n".join(report_lines),
    )
