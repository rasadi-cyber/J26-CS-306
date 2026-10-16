from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    output_path = REPORT_DIR / "feature_group_comparison.csv"

    if output_path.exists():
        raise SystemExit("Comparison already exists. Preserve this experiment.")

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )

    source = PROCESSED_DIR / "firmware_features.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Feature dataset changed after splitting.")

    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_features.csv")
    stress = pd.read_csv(PROCESSED_DIR / "validation_stress_features.csv")

    validation_hashes = set(validation["meta_payload_sha256"])

    if set(stress["meta_parent_payload_sha256"]) != validation_hashes:
        raise ValueError("Stress samples do not match validation parents.")

    if set(train["meta_payload_sha256"]) & validation_hashes:
        raise ValueError("Training/validation overlap.")

    all_names = config["feature_names"]
    groups = {
        "generic": [
            name for name in all_names if not name.startswith("uav_")
        ],
        "uav": [
            name for name in all_names if name.startswith("uav_")
        ],
        "combined": all_names,
    }

    labels = stress["meta_kind"].ne("clean").astype(int).to_numpy()
    clean_mask = labels == 0

    summaries = []
    details = []

    for group_name, names in groups.items():
        estimators = {
            "one_class_svm": OneClassSVM(
                kernel="rbf", nu=0.10, gamma="scale"
            ),
            "isolation_forest": IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=42,
                n_jobs=-1,
            ),
        }

        for model_name, estimator in estimators.items():
            pipeline = Pipeline([
                ("remove_constants", VarianceThreshold()),
                ("scale", StandardScaler()),
                ("model", estimator),
            ])

            pipeline.fit(train[names])

            clean_scores = -pipeline.decision_function(validation[names])
            threshold = float(np.quantile(clean_scores, 0.95))

            scores = -pipeline.decision_function(stress[names])
            flagged = scores > threshold

            retained = int(
                pipeline.named_steps["remove_constants"]
                .get_support().sum()
            )

            summaries.append({
                "feature_group": group_name,
                "model": model_name,
                "candidate_features": len(names),
                "retained_features": retained,
                "threshold": threshold,
                "precision": float(
                    precision_score(labels, flagged, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, flagged, zero_division=0)
                ),
                "f1": float(f1_score(labels, flagged, zero_division=0)),
                "clean_flag_rate": float(flagged[clean_mask].mean()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "pr_average_precision": float(
                    average_precision_score(labels, scores)
                ),
            })

            for (family, kind), subset in stress.groupby(
                ["meta_family", "meta_kind"]
            ):
                positions = subset.index.to_numpy()

                details.append({
                    "feature_group": group_name,
                    "model": model_name,
                    "family": family,
                    "kind": kind,
                    "samples": len(positions),
                    "flagged": int(flagged[positions].sum()),
                    "flag_rate": float(flagged[positions].mean()),
                })

            print(f"Completed: {group_name} / {model_name}", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_path, index=False)

    pd.DataFrame(details).to_csv(
        REPORT_DIR / "feature_group_comparison_details.csv",
        index=False,
    )

    print("\nVALIDATION FEATURE-GROUP COMPARISON")
    print(summary.drop(columns=["threshold"]).to_string(index=False))
    print("\nReserved test set NOT used.")


if __name__ == "__main__":
    main()