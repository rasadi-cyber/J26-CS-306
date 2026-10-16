from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from binary_features import extract_binary_features
from content_features import extract_content_features
from controlled_mutations import generate_mutations
from firmware_loader import load_firmware
from uav_features import extract_uav_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"


def extract_all_features(main_image, external_image):
    uav, _ = extract_uav_features(main_image, external_image)

    return {
        **extract_binary_features(main_image, external_image),
        **extract_content_features(main_image, external_image),
        **uav,
    }


def main():
    summary_path = REPORT_DIR / "initial_validation_evaluation.json"

    if summary_path.exists():
        raise SystemExit(
            "Evaluation already exists. Preserve the initial results."
        )

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )
    names = config["feature_names"]

    source = PROCESSED_DIR / "firmware_features.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["dataset_sha256"]:
        raise ValueError("Feature dataset changed after splitting.")

    validation = pd.read_csv(
        PROCESSED_DIR / "validation_features.csv"
    )

    records = []

    for index, (_, parent) in enumerate(validation.iterrows(), start=1):
        path = PROJECT_ROOT / "data" / "raw" / parent["meta_relative_path"]
        firmware = load_firmware(path)

        if firmware["container_sha256"] != parent["meta_container_sha256"]:
            raise ValueError(f"Original file changed: {path.name}")

        if firmware["payload_sha256"] != parent["meta_payload_sha256"]:
            raise ValueError(f"Original payload changed: {path.name}")

        original_features = extract_all_features(
            firmware["main_image"], firmware["external_image"]
        )

        if set(original_features) != set(names):
            raise ValueError("Feature schema mismatch.")

        if not np.allclose(
            [original_features[name] for name in names],
            parent[names].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(f"Feature extraction changed: {path.name}")

        common = {
            "meta_family": parent["meta_family"],
            "meta_parent_payload_sha256": parent["meta_payload_sha256"],
        }

        records.append({
            **common,
            "meta_kind": "clean",
            "meta_changed_bytes": 0,
            "meta_modified_payload_sha256": parent["meta_payload_sha256"],
            **original_features,
        })

        for mutation in generate_mutations(
            firmware["main_image"], firmware["external_image"]
        ):
            features = extract_all_features(
                mutation["main_image"], mutation["external_image"]
            )

            records.append({
                **common,
                "meta_kind": mutation["mutation_type"],
                "meta_changed_bytes": mutation["changed_bytes"],
                "meta_modified_payload_sha256": mutation[
                    "modified_payload_sha256"
                ],
                **features,
            })

        if index % 10 == 0:
            print(f"Processed {index} validation parents", flush=True)

    samples = pd.DataFrame(records)

    if not np.isfinite(samples[names].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite modified features.")

    samples.to_csv(
        PROCESSED_DIR / "validation_stress_features.csv",
        index=False,
    )

    labels = samples["meta_kind"].ne("clean").astype(int).to_numpy()
    clean_mask = labels == 0
    summaries = {}

    for model_name in ("one_class_svm", "isolation_forest"):
        bundle = joblib.load(
            MODEL_DIR / f"{model_name}_initial.joblib"
        )

        if bundle["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Model/dataset mismatch.")

        if bundle["feature_names"] != names:
            raise ValueError("Model feature schema mismatch.")

        scores = -bundle["pipeline"].decision_function(samples[names])
        flagged = scores > bundle["threshold"]

        result = samples[
            [name for name in samples if name.startswith("meta_")]
        ].copy()
        result["anomaly_score"] = scores
        result["flagged"] = flagged

        parent_scores = result.loc[clean_mask].set_index(
            "meta_parent_payload_sha256"
        )["anomaly_score"]

        result["parent_anomaly_score"] = result[
            "meta_parent_payload_sha256"
        ].map(parent_scores)

        result["score_change_from_parent"] = (
            result["anomaly_score"] - result["parent_anomaly_score"]
        )

        result.to_csv(
            REPORT_DIR / f"{model_name}_validation_stress_scores.csv",
            index=False,
        )

        metrics = {
            "precision": float(
                precision_score(labels, flagged, zero_division=0)
            ),
            "recall": float(
                recall_score(labels, flagged, zero_division=0)
            ),
            "f1": float(f1_score(labels, flagged, zero_division=0)),
            "candidate_clean_flag_rate": float(flagged[clean_mask].mean()),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_average_precision": float(
                average_precision_score(labels, scores)
            ),
        }

        summaries[model_name] = metrics

        print(f"\nMODEL: {model_name}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("Flags by family and modification:")
        print(
            result.groupby(["meta_family", "meta_kind"])["flagged"]
            .agg(["count", "sum", "mean"])
            .to_string()
        )

        print("Mean score change from parent:")
        print(
            result.groupby("meta_kind")["score_change_from_parent"]
            .mean()
            .to_string()
        )

    summary_path.write_text(
        json.dumps({
            "validation_parents": len(validation),
            "stress_samples": len(samples),
            "test_set_used": False,
            "evaluation_type": "exploratory synthetic validation",
            "metrics": summaries,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nValidation evaluation saved. Reserved test set NOT used.")


if __name__ == "__main__":
    main()