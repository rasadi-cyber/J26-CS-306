from pathlib import Path
from itertools import product
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    summary_path = REPORT_DIR / "iforest_tuning_selected.csv"

    if summary_path.exists():
        raise SystemExit("Tuning already exists. Preserve this experiment.")

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )

    source = PROCESSED_DIR / "firmware_features.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Dataset changed after splitting.")

    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_features.csv")
    stress = pd.read_csv(PROCESSED_DIR / "validation_stress_features.csv")

    validation_hashes = set(validation["meta_payload_sha256"])

    if set(stress["meta_parent_payload_sha256"]) != validation_hashes:
        raise ValueError("Stress samples do not match validation parents.")

    if set(train["meta_payload_sha256"]) & validation_hashes:
        raise ValueError("Training/validation overlap.")

    all_names = config["feature_names"]
    representations = {
        "generic": [
            name for name in all_names if not name.startswith("uav_")
        ],
        "combined": all_names,
    }

    candidates = []
    selected = []
    details = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for family in sorted(train["meta_family"].unique()):
        family_train = train[train["meta_family"] == family]
        family_validation = validation[
            validation["meta_family"] == family
        ]
        family_stress = stress[
            stress["meta_family"] == family
        ].reset_index(drop=True)

        labels = family_stress["meta_kind"].ne("clean").astype(int).to_numpy()
        clean_mask = labels == 0

        for representation, names in representations.items():
            best_key = None
            best_bundle = None
            best_summary = None
            best_flags = None
            best_new_flags = None
            best_eligible = None

            for trees, sample_fraction, feature_fraction in product(
                (100, 200),
                (0.50, 0.75, 1.0),
                (0.50, 1.0),
            ):
                pipeline = Pipeline([
                    ("remove_constants", VarianceThreshold()),
                    ("scale", StandardScaler()),
                    ("model", IsolationForest(
                        n_estimators=trees,
                        max_samples=sample_fraction,
                        max_features=feature_fraction,
                        contamination="auto",
                        random_state=42,
                        n_jobs=-1,
                    )),
                ])

                pipeline.fit(family_train[names])

                clean_scores = -pipeline.decision_function(
                    family_validation[names]
                )
                threshold = float(np.quantile(clean_scores, 0.95))

                scores = -pipeline.decision_function(family_stress[names])
                flags = scores > threshold

                clean_parent_flags = pd.Series(
                    flags[clean_mask],
                    index=family_stress.loc[
                        clean_mask, "meta_parent_payload_sha256"
                    ],
                )

                parent_flags = family_stress[
                    "meta_parent_payload_sha256"
                ].map(clean_parent_flags).to_numpy(dtype=bool)

                eligible = (labels == 1) & ~parent_flags
                new_flags = flags & eligible

                new_detection_rate = (
                    float(new_flags.sum() / eligible.sum())
                    if eligible.any() else 0.0
                )

                row = {
                    "family": family,
                    "representation": representation,
                    "n_estimators": trees,
                    "max_samples": sample_fraction,
                    "max_features": feature_fraction,
                    "retained_features": int(
                        pipeline.named_steps["remove_constants"]
                        .get_support().sum()
                    ),
                    "threshold": threshold,
                    "recall": float(
                        recall_score(labels, flags, zero_division=0)
                    ),
                    "f1": float(f1_score(labels, flags, zero_division=0)),
                    "clean_flag_rate": float(flags[clean_mask].mean()),
                    "new_detection_rate": new_detection_rate,
                    "roc_auc": float(roc_auc_score(labels, scores)),
                }
                candidates.append(row)

                key = (new_detection_rate, row["roc_auc"])

                if best_key is None or key > best_key:
                    best_key = key
                    best_summary = row
                    best_flags = flags.copy()
                    best_new_flags = new_flags.copy()
                    best_eligible = eligible.copy()

                    best_bundle = {
                        "pipeline": pipeline,
                        "feature_names": names,
                        "threshold": threshold,
                        "model_name": "isolation_forest",
                        "family": family,
                        "representation": representation,
                        "experiment": "bounded_validation_tuning",
                        "dataset_sha256": config["dataset_sha256"],
                        "training_medians": family_train[
                            names
                        ].median().to_dict(),
                        "training_standard_deviations": family_train[
                            names
                        ].std(ddof=0).to_dict(),
                        "validation_scores": clean_scores.tolist(),
                        "parameters": {
                            "n_estimators": trees,
                            "max_samples": sample_fraction,
                            "max_features": feature_fraction,
                        },
                    }

            selected.append(best_summary)

            joblib.dump(
                best_bundle,
                MODEL_DIR / f"iforest_tuned_{family}_{representation}.joblib",
            )

            for kind, subset in family_stress.groupby("meta_kind"):
                positions = subset.index.to_numpy()

                details.append({
                    "family": family,
                    "representation": representation,
                    "kind": kind,
                    "samples": len(positions),
                    "flagged": int(best_flags[positions].sum()),
                    "unflagged_parent_modified_samples": int(
                        best_eligible[positions].sum()
                    ),
                    "new_flags": int(best_new_flags[positions].sum()),
                })

            print(
                f"Completed: {family} / {representation}",
                flush=True,
            )

    pd.DataFrame(candidates).to_csv(
        REPORT_DIR / "iforest_tuning_candidates.csv", index=False
    )
    pd.DataFrame(details).to_csv(
        REPORT_DIR / "iforest_tuning_modification_details.csv",
        index=False,
    )

    summary = pd.DataFrame(selected)
    summary.to_csv(summary_path, index=False)

    print("\nSELECTED VALIDATION CONFIGURATIONS")
    print(summary.to_string(index=False))
    print("\nSelection: new detection rate, then ROC-AUC.")
    print("Reserved test set NOT used.")


if __name__ == "__main__":
    main()