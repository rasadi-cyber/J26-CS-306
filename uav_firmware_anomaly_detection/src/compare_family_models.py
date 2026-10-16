from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"


def make_pipeline(model_name):
    if model_name == "one_class_svm":
        estimator = OneClassSVM(
            kernel="rbf", nu=0.10, gamma="scale"
        )
    else:
        estimator = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("remove_constants", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def main():
    output_path = REPORT_DIR / "family_model_comparison.csv"

    if output_path.exists():
        raise SystemExit("Comparison already exists. Preserve these results.")

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )
    names = config["feature_names"]

    source = PROCESSED_DIR / "firmware_features.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Feature dataset changed after splitting.")

    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_features.csv")
    stress = pd.read_csv(PROCESSED_DIR / "validation_stress_features.csv")

    if set(stress["meta_parent_payload_sha256"]) != set(
        validation["meta_payload_sha256"]
    ):
        raise ValueError("Stress samples do not match validation parents.")

    labels = stress["meta_kind"].ne("clean").astype(int).to_numpy()
    clean_mask = labels == 0

    summaries = []
    family_details = []
    modification_details = []

    for strategy in ("pooled", "separate_families"):
        for model_name in ("one_class_svm", "isolation_forest"):
            flagged = np.zeros(len(stress), dtype=bool)

            pooled_pipeline = None
            if strategy == "pooled":
                pooled_pipeline = make_pipeline(model_name)
                pooled_pipeline.fit(train[names])

            for family in sorted(validation["meta_family"].unique()):
                family_train = train[train["meta_family"] == family]
                family_validation = validation[
                    validation["meta_family"] == family
                ]
                positions = stress.index[
                    stress["meta_family"] == family
                ].to_numpy()

                if strategy == "pooled":
                    pipeline = pooled_pipeline
                else:
                    pipeline = make_pipeline(model_name)
                    pipeline.fit(family_train[names])

                clean_scores = -pipeline.decision_function(
                    family_validation[names]
                )
                threshold = float(np.quantile(clean_scores, 0.95))

                scores = -pipeline.decision_function(
                    stress.loc[positions, names]
                )
                family_flags = scores > threshold
                flagged[positions] = family_flags

                family_labels = labels[positions]
                family_clean = family_labels == 0

                family_details.append({
                    "strategy": strategy,
                    "model": model_name,
                    "family": family,
                    "training_samples": (
                        len(train) if strategy == "pooled"
                        else len(family_train)
                    ),
                    "retained_features": int(
                        pipeline.named_steps["remove_constants"]
                        .get_support().sum()
                    ),
                    "recall": float(
                        recall_score(
                            family_labels, family_flags, zero_division=0
                        )
                    ),
                    "clean_flag_rate": float(
                        family_flags[family_clean].mean()
                    ),
                    "roc_auc": float(
                        roc_auc_score(family_labels, scores)
                    ),
                })

            summaries.append({
                "strategy": strategy,
                "model": model_name,
                "precision": float(
                    precision_score(labels, flagged, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, flagged, zero_division=0)
                ),
                "f1": float(f1_score(labels, flagged, zero_division=0)),
                "clean_flag_rate": float(flagged[clean_mask].mean()),
            })

            for (family, kind), subset in stress.groupby(
                ["meta_family", "meta_kind"]
            ):
                positions = subset.index.to_numpy()
                modification_details.append({
                    "strategy": strategy,
                    "model": model_name,
                    "family": family,
                    "kind": kind,
                    "samples": len(positions),
                    "flagged": int(flagged[positions].sum()),
                })

            print(f"Completed: {strategy} / {model_name}", flush=True)

    summary = pd.DataFrame(summaries)
    details = pd.DataFrame(family_details)

    summary.to_csv(output_path, index=False)
    details.to_csv(
        REPORT_DIR / "family_model_comparison_details.csv", index=False
    )
    pd.DataFrame(modification_details).to_csv(
        REPORT_DIR / "family_model_modification_details.csv", index=False
    )

    print("\nOVERALL VALIDATION RESULTS")
    print(summary.to_string(index=False))
    print("\nRESULTS BY FAMILY")
    print(details.to_string(index=False))
    print("\nBoth strategies used family-specific validation thresholds.")
    print("Reserved test set NOT used.")


if __name__ == "__main__":
    main()