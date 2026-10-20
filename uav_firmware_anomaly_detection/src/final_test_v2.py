"""One-time final evaluation on the reserved version 2 test split.

Uses the frozen saved models (thresholds already fixed on validation data).
Nothing is tuned here. Run once; it refuses to overwrite its report.
Outputs reports/v2/final_test_summary.csv.
"""
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed_v2"
REPORTS = ROOT / "reports" / "v2"
MODELS = ROOT / "models" / "v2"
OUT = REPORTS / "final_test_summary.csv"

# (label, family, vehicle scope used for the model, test rows used)
SCOPES = [
    ("ArduPilot/multirotor", "ArduPilot", "multirotor"),
    ("ArduPilot/fixed_wing", "ArduPilot", "fixed_wing"),
    ("ArduPilot/helicopter (general ArduPilot model)", "ArduPilot", "helicopter"),
    ("PX4/all", "PX4", "all"),
]
MODELS_USED = [
    ("UAV-only One-Class SVM (main)", "uav", "one_class_svm"),
    ("UAV-only Isolation Forest (baseline)", "uav", "isolation_forest"),
    ("Generic bytes One-Class SVM", "generic", "one_class_svm"),
]


def main():
    if OUT.exists():
        raise SystemExit("Final test already run; keep the result.")

    manifest = pd.read_csv(REPORTS / "split_manifest.csv")
    test_path = PROCESSED / "test_features.csv"
    if not test_path.exists():
        full = pd.read_csv(PROCESSED / "firmware_features.csv")
        keys = manifest.loc[manifest["split"] == "test", "meta_payload_sha256"]
        full[full["meta_payload_sha256"].isin(keys)].to_csv(
            test_path, index=False)
    if not (PROCESSED / "test_stress_features.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "build_stress_v2.py"),
                        "--split", "test"], check=True)

    stress = pd.read_csv(PROCESSED / "test_stress_features.csv")
    rows = []
    for label, family, vehicle in SCOPES:
        mask = stress["meta_family"].eq(family)
        if vehicle != "all":
            mask &= stress["meta_vehicle_class"].eq(vehicle)
        data = stress[mask].reset_index(drop=True)
        # Helicopters use the general ArduPilot model; others their own.
        scope_file = "all" if vehicle == "helicopter" else vehicle
        clean = data["meta_kind"].eq("clean").to_numpy()
        kinds = data["meta_kind"].to_numpy()
        uav_mask = np.array([k.startswith("uav_remove_") for k in kinds])
        byte_mask = ~clean & ~uav_mask
        parents = data["meta_parent_payload_sha256"]

        for name, rep, model in MODELS_USED:
            path = MODELS / f"{family}_{scope_file}_{rep}_{model}.joblib"
            bundle = joblib.load(path)
            cols = bundle["feature_names"]
            scores = -bundle["pipeline"].decision_function(data[cols])
            flags = scores > bundle["threshold"]
            parent_flag = parents.map(pd.Series(
                flags[clean], index=parents[clean])).to_numpy(dtype=bool)
            eligible = ~clean & ~parent_flag

            def rate(m):
                m = m & eligible
                return float(flags[m].sum() / m.sum()) if m.any() else np.nan

            rows.append({
                "scope": label, "model": name,
                "clean_test_samples": int(clean.sum()),
                "clean_flag_rate": float(flags[clean].mean()),
                "detect_byte_level": rate(byte_mask),
                "detect_uav_targeted": rate(uav_mask),
                "roc_auc_all_modified": float(
                    roc_auc_score((~clean).astype(int), scores)),
            })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
