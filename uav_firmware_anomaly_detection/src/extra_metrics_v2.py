"""Additional evaluation metrics for the frozen version 2 models.

Reads the test stress samples that final_test_v2.py already built and the
saved models. Nothing is retrained or re-tuned; thresholds stay as fixed on
validation data. Adds the metrics promised in the proposal (Table 2):
precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC, inference time.

Positives are the modified samples, negatives are the clean parent samples.
The test set contains several modified samples per clean sample, so
precision is affected by that class balance; read it with the clean counts.
Outputs reports/v2/extra_metrics_summary.csv (refuses to overwrite).
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed_v2"
REPORTS = ROOT / "reports" / "v2"
MODELS = ROOT / "models" / "v2"
OUT = REPORTS / "extra_metrics_summary.csv"

SCOPES = [
    ("ArduPilot/multirotor", "ArduPilot", "multirotor", "multirotor"),
    ("ArduPilot/fixed_wing", "ArduPilot", "fixed_wing", "fixed_wing"),
    ("ArduPilot/helicopter (general model)", "ArduPilot", "helicopter", "all"),
    ("PX4/all", "PX4", "all", "all"),
]
MODELS_USED = [
    ("UAV-only One-Class SVM (main)", "uav", "one_class_svm"),
    ("UAV-only Isolation Forest (baseline)", "uav", "isolation_forest"),
    ("Generic bytes One-Class SVM", "generic", "one_class_svm"),
]


def main():
    if OUT.exists():
        raise SystemExit("Extra metrics already computed; keep the result.")
    stress = pd.read_csv(PROCESSED / "test_stress_features.csv")
    rows = []
    for label, family, vehicle, model_scope in SCOPES:
        mask = stress["meta_family"].eq(family)
        if vehicle != "all":
            mask &= stress["meta_vehicle_class"].eq(vehicle)
        data = stress[mask].reset_index(drop=True)
        kinds = data["meta_kind"].to_numpy()
        clean = kinds == "clean"
        uav_mod = np.array([k.startswith("uav_remove_") for k in kinds])
        byte_mod = ~clean & ~uav_mod

        for name, rep, model in MODELS_USED:
            bundle = joblib.load(
                MODELS / f"{family}_{model_scope}_{rep}_{model}.joblib")
            cols = bundle["feature_names"]
            started = time.perf_counter()
            scores = -bundle["pipeline"].decision_function(data[cols])
            ms_per_file = 1000 * (time.perf_counter() - started) / len(data)
            flags = scores > bundle["threshold"]

            for positives, positive_name in ((uav_mod, "safety-content removal"),
                                             (byte_mod, "byte-level edits")):
                keep = clean | positives
                y = positives[keep].astype(int)
                s = scores[keep]
                f = flags[keep].astype(int)
                rows.append({
                    "scope": label, "model": name,
                    "modification_type": positive_name,
                    "clean_samples": int(clean.sum()),
                    "modified_samples": int(positives.sum()),
                    "precision": precision_score(y, f, zero_division=0),
                    "recall": recall_score(y, f, zero_division=0),
                    "f1": f1_score(y, f, zero_division=0),
                    "false_positive_rate": float(flags[clean].mean()),
                    "roc_auc": roc_auc_score(y, s),
                    "pr_auc": average_precision_score(y, s),
                    "scoring_ms_per_file": ms_per_file,
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(result.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
