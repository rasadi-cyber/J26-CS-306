"""Train and compare One-Class SVM and Isolation Forest on version 2 data.

For every scope (ArduPilot overall, ArduPilot multirotor, ArduPilot
fixed-wing, PX4) and every feature representation (generic, UAV-only,
combined) a small, fixed grid of settings is tried per model. The setting is
chosen on the VALIDATION stress samples only; the test split is not used.

Threshold: 95th percentile of the clean validation scores, so the clean
flag rate is about 5% by construction (it is not an independent false
positive estimate).
"""
from pathlib import Path
from itertools import product
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "v2"
MODEL_DIR = PROJECT_ROOT / "models" / "v2"

SCOPES = [
    ("ArduPilot", "all"),
    ("ArduPilot", "multirotor"),
    ("ArduPilot", "fixed_wing"),
    ("PX4", "all"),
]
MIN_TRAIN = 15
MIN_CLEAN_VALIDATION = 5


def candidate_models():
    for nu, gamma in product((0.05, 0.10, 0.20),
                             ("scale", 0.0001, 0.001, 0.01)):
        yield "one_class_svm", f"nu={nu};gamma={gamma}", lambda nu=nu, gamma=gamma: OneClassSVM(
            kernel="rbf", nu=nu, gamma=gamma
        )
    for max_samples, max_features in product(("auto", 0.5), (1.0, 0.5)):
        yield "isolation_forest", f"max_samples={max_samples};max_features={max_features}", lambda ms=max_samples, mf=max_features: IsolationForest(
            n_estimators=200, max_samples=ms, max_features=mf,
            contamination="auto", random_state=42, n_jobs=-1,
        )


def subset(frame, family, vehicle):
    mask = frame["meta_family"].eq(family)
    if vehicle != "all":
        mask &= frame["meta_vehicle_class"].eq(vehicle)
    return frame[mask].reset_index(drop=True)


def rate(flags, mask):
    return float(flags[mask].sum() / mask.sum()) if mask.any() else float("nan")


def main():
    summary_path = REPORT_DIR / "train_eval_summary.csv"
    if summary_path.exists():
        raise SystemExit("Version 2 training results exist; keep them.")

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )
    source = PROCESSED_DIR / "firmware_features.csv"
    dataset_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if dataset_hash != config["dataset_sha256"]:
        raise ValueError("Dataset changed after splitting.")

    names = config["feature_names"]
    representations = {
        "generic": [n for n in names if not n.startswith("uav_")],
        "uav": [n for n in names if n.startswith("uav_")],
        "combined": names,
    }

    train_all = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    stress_all = pd.read_csv(PROCESSED_DIR / "validation_stress_features.csv")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    selected, candidates = [], []

    for family, vehicle in SCOPES:
        train = subset(train_all, family, vehicle)
        stress = subset(stress_all, family, vehicle)
        label = f"{family}/{vehicle}"

        clean_mask = stress["meta_kind"].eq("clean").to_numpy()
        if len(train) < MIN_TRAIN or clean_mask.sum() < MIN_CLEAN_VALIDATION:
            print(f"Skipping {label}: train={len(train)}, "
                  f"clean validation={int(clean_mask.sum())}")
            continue

        labels = (~clean_mask).astype(int)
        kinds = stress["meta_kind"].to_numpy()
        uav_mask = np.array([k.startswith("uav_remove_") for k in kinds])
        byte_mask = labels.astype(bool) & ~uav_mask
        parents = stress["meta_parent_payload_sha256"]

        for representation, columns in representations.items():
            try:
                VarianceThreshold().fit(train[columns])
            except ValueError:
                print(f"Skipping {label} / {representation}: no feature "
                      "varies in the training data", flush=True)
                continue
            best = {}
            for model_name, setting, factory in candidate_models():
                pipeline = Pipeline([
                    ("remove_constants", VarianceThreshold()),
                    ("scale", StandardScaler()),
                    ("model", factory()),
                ])
                pipeline.fit(train[columns])
                scores = -pipeline.decision_function(stress[columns])
                clean_scores = scores[clean_mask]
                threshold = float(np.quantile(clean_scores, 0.95))
                flags = scores > threshold

                parent_flag = parents.map(
                    pd.Series(flags[clean_mask],
                              index=parents[clean_mask])
                ).to_numpy(dtype=bool)
                eligible = labels.astype(bool) & ~parent_flag

                detect_byte = rate(flags, eligible & byte_mask)
                detect_uav = rate(flags, eligible & uav_mask)
                parts = [v for v in (detect_byte, detect_uav)
                         if not np.isnan(v)]
                detect_mean = float(np.mean(parts)) if parts else 0.0
                auc = float(roc_auc_score(labels, scores))

                row = {
                    "scope": label, "representation": representation,
                    "model": model_name, "setting": setting,
                    "train_samples": len(train),
                    "clean_validation_samples": int(clean_mask.sum()),
                    "retained_features": int(
                        pipeline.named_steps["remove_constants"]
                        .get_support().sum()
                    ),
                    "threshold": threshold,
                    "clean_flag_rate": float(flags[clean_mask].mean()),
                    "detect_byte_level": detect_byte,
                    "detect_uav_targeted": detect_uav,
                    "detect_mean": detect_mean,
                    "roc_auc": auc,
                }
                for kind in sorted(set(kinds) - {"clean"}):
                    row[f"det_{kind}"] = rate(
                        flags, eligible & (kinds == kind)
                    )
                candidates.append(row)

                key = (detect_mean, auc)
                if model_name not in best or key > best[model_name][0]:
                    medians = train[columns].median().to_dict()
                    spreads = train[columns].std(ddof=0).to_dict()
                    best[model_name] = (key, row, {
                        "pipeline": pipeline,
                        "feature_names": columns,
                        "threshold": threshold,
                        "score_definition": "negative decision_function",
                        "model_name": model_name,
                        "family": family,
                        "vehicle_scope": vehicle,
                        "representation": representation,
                        "experiment": "v2_validation_selection",
                        "setting": setting,
                        "dataset_sha256": dataset_hash,
                        "sklearn_version": sklearn.__version__,
                        "training_medians": medians,
                        "training_standard_deviations": spreads,
                        "validation_scores": clean_scores.tolist(),
                    })

            for model_name, (_, row, bundle) in best.items():
                selected.append(row)
                joblib.dump(
                    bundle,
                    MODEL_DIR / f"{family}_{vehicle}_{representation}_"
                                f"{model_name}.joblib",
                )
            print(f"Completed: {label} / {representation}", flush=True)

    pd.DataFrame(candidates).to_csv(
        REPORT_DIR / "train_eval_candidates.csv", index=False
    )
    summary = pd.DataFrame(selected)
    summary.to_csv(summary_path, index=False)

    show = ["scope", "representation", "model", "train_samples",
            "clean_flag_rate", "detect_byte_level", "detect_uav_targeted",
            "roc_auc"]
    pd.set_option("display.width", 200)
    print("\nSELECTED VALIDATION RESULTS")
    print(summary[show].round(3).to_string(index=False))
    print("\nTest split NOT used. Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
