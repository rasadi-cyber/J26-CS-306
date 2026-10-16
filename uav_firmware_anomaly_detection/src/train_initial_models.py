from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    summary_path = REPORT_DIR / "initial_training_summary.json"

    if summary_path.exists():
        raise SystemExit(
            "Initial training already exists. Preserve this experiment."
        )

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )

    source = PROCESSED_DIR / "firmware_features.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    if source_hash != config["dataset_sha256"]:
        raise ValueError("The feature dataset changed after splitting.")

    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_features.csv")
    names = config["feature_names"]

    train_hashes = set(train["meta_payload_sha256"])
    validation_hashes = set(validation["meta_payload_sha256"])

    if train_hashes & validation_hashes:
        raise ValueError("Training/validation payload overlap detected.")

    x_train = train[names]
    x_validation = validation[names]

    for matrix in (x_train, x_validation):
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite input features.")

    estimators = {
        "one_class_svm": OneClassSVM(
            kernel="rbf",
            nu=0.10,
            gamma="scale",
        ),
        "isolation_forest": IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("remove_constants", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

        pipeline.fit(x_train)

        # Larger scores mean more anomalous for both models.
        scores = -pipeline.decision_function(x_validation)

        # Initial calibration choice; not a guaranteed false-positive rate.
        threshold = float(np.quantile(scores, 0.95))
        flagged = scores > threshold

        retained = pipeline.named_steps[
            "remove_constants"
        ].get_support()

        bundle = {
            "pipeline": pipeline,
            "feature_names": names,
            "threshold": threshold,
            "score_definition": "negative decision_function",
            "dataset_sha256": source_hash,
            "sklearn_version": sklearn.__version__,
            "model_name": name,
            "experiment": "initial_pooled",
            "training_families": sorted(train["meta_family"].unique()),
            "training_medians": x_train.median().to_dict(),
            "training_standard_deviations": x_train.std(
                ddof=0
            ).to_dict(),
            "validation_scores": scores.tolist(),
        }

        joblib.dump(bundle, MODEL_DIR / f"{name}_initial.joblib")

        score_report = validation[
            ["meta_relative_path", "meta_family", "meta_payload_sha256"]
        ].copy()
        score_report["anomaly_score"] = scores
        score_report["threshold"] = threshold
        score_report["flagged"] = flagged

        score_report.to_csv(
            REPORT_DIR / f"{name}_initial_validation.csv",
            index=False,
        )

        family_results = {}
        for family in sorted(validation["meta_family"].unique()):
            mask = validation["meta_family"].eq(family).to_numpy()
            family_results[family] = {
                "samples": int(mask.sum()),
                "flagged": int(flagged[mask].sum()),
            }

        summaries[name] = {
            "training_samples": len(train),
            "validation_samples": len(validation),
            "candidate_features": len(names),
            "retained_features": int(retained.sum()),
            "threshold": threshold,
            "validation_flagged": int(flagged.sum()),
            "validation_by_family": family_results,
        }

        print(f"\nMODEL: {name}")
        print("Training samples:", len(train))
        print("Retained features:", int(retained.sum()), "of", len(names))
        print("Validation threshold:", threshold)
        print("Candidate-clean validation flags:", int(flagged.sum()))
        print("By family:", family_results)

    summary_path.write_text(
        json.dumps({
            "dataset_sha256": source_hash,
            "sklearn_version": sklearn.__version__,
            "test_set_used": False,
            "threshold_policy": "95th percentile of validation scores",
            "models": summaries,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nBoth initial models saved. Reserved test set NOT used.")


if __name__ == "__main__":
    main()