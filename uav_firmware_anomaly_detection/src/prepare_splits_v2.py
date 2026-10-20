"""Version 2 train / validation / test split.

Hardware is the unit of separation: every sample of the same board (any
release, any vehicle type) goes to exactly one split, so the test set
contains boards the model has never seen. Writes only to data/processed_v2
and reports/v2.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "v2"
SEED = 42


def main():
    source = PROCESSED_DIR / "firmware_features.csv"
    config_path = REPORT_DIR / "split_config.json"

    if config_path.exists():
        raise SystemExit(
            "Version 2 splits already exist. Keep them fixed; do not overwrite."
        )

    data = pd.read_csv(source, dtype={"meta_board_id": "string"})

    feature_names = [
        name for name in data.columns if not name.startswith("meta_")
    ]

    if data["meta_payload_sha256"].duplicated().any():
        raise ValueError("Duplicate payload hashes found.")

    required = ["meta_family", "meta_board_id", "meta_vehicle_class",
                "meta_release"]
    if data[required].isna().any().any():
        raise ValueError("Missing family, board, vehicle or release values.")

    values = data[feature_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite numeric features.")

    assignments = pd.Series("", index=data.index, dtype="string")

    for family, subset in data.groupby("meta_family", sort=True):
        groups = subset["meta_board_id"]

        if groups.nunique() < 5:
            raise ValueError(f"Too few board groups for {family}.")

        first_split = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=SEED
        )
        remaining_positions, test_positions = next(
            first_split.split(subset, groups=groups)
        )
        remaining = subset.iloc[remaining_positions]

        second_split = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=SEED + 1
        )
        train_positions, validation_positions = next(
            second_split.split(remaining, groups=remaining["meta_board_id"])
        )

        assignments.loc[remaining.iloc[train_positions].index] = "train"
        assignments.loc[
            remaining.iloc[validation_positions].index
        ] = "validation"
        assignments.loc[subset.iloc[test_positions].index] = "test"

    if assignments.eq("").any():
        raise ValueError("Some samples were not assigned.")

    group_keys = (
        data["meta_family"].astype(str)
        + ":"
        + data["meta_board_id"].astype(str)
    )
    group_sets = {
        role: set(group_keys[assignments == role])
        for role in ("train", "validation", "test")
    }
    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test"),
    ):
        if group_sets[first] & group_sets[second]:
            raise ValueError("Board-group leakage detected.")

    payloads = {
        role: set(data.loc[assignments == role, "meta_payload_sha256"])
        for role in ("train", "validation", "test")
    }
    for first, second in (
        ("train", "validation"), ("train", "test"), ("validation", "test"),
    ):
        if payloads[first] & payloads[second]:
            raise ValueError("Payload overlap between splits.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_names = [
        name for name in data.columns if name.startswith("meta_")
    ]
    manifest = data[metadata_names].copy()
    manifest["split"] = assignments
    manifest.to_csv(REPORT_DIR / "split_manifest.csv", index=False)

    for role in ("train", "validation", "test"):
        data.loc[assignments == role].to_csv(
            PROCESSED_DIR / f"{role}_features.csv", index=False
        )

    config = {
        "version": "v2",
        "seed": SEED,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_names": feature_names,
        "grouping": "firmware family plus board ID (all releases and "
                    "vehicle types of a board stay together)",
        "split_sizes": {k: int(v) for k, v in
                        assignments.value_counts().items()},
        "training_population": "candidate legitimate firmware",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Samples per family and split:")
    print(pd.crosstab(data["meta_family"], assignments).to_string())
    print("\nSamples per vehicle class and split:")
    print(pd.crosstab(data["meta_vehicle_class"], assignments).to_string())
    print("\nSamples per release and split:")
    print(pd.crosstab(data["meta_release"], assignments).to_string())
    print("\nBoard-group overlap: NONE")
    print("Payload overlap: NONE")
    print("Numeric features:", len(feature_names))
    print("Split manifest and configuration saved.")


if __name__ == "__main__":
    main()
