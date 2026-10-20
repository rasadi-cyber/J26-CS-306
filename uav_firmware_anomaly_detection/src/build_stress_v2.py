"""Build modified-firmware stress samples for the version 2 split.

Two kinds of controlled modification are generated from each legitimate
sample of the chosen split:

  * byte-level edits (the four version 1 types), and
  * UAV-targeted edits, which blank out flight-stack content groups
    (geofence, arming/failsafe, navigation/RTL, flight control, state
    estimation) with same-length zero bytes.

These are synthetic stress tests, not confirmed attacks. UAV-targeted edits
remove flight-safety parameter names and messages; they show whether a
representation can see such changes at all, not how real attackers behave.

Defaults to the validation split. The test split is only built when you
pass --split test, which you should do once, at the very end.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from controlled_mutations import generate_mutations
from feature_pipeline import extract_firmware_features
from firmware_loader import calculate_payload_hash, load_firmware
from uav_features import GROUP_PATTERNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "v2"

SAFETY_GROUPS = ("geofence", "arming_failsafe", "navigation_rtl")


def blank_markers(image, groups):
    """Zero every flight-stack marker of the given groups, same length."""
    data = bytearray(image)
    changed = 0
    for group in groups:
        for pattern in GROUP_PATTERNS[group].values():
            expression = re.compile(pattern.encode("ascii"))
            for match in expression.finditer(bytes(data)):
                span = range(match.start(), match.end())
                changed += sum(1 for i in span if data[i] != 0)
                data[match.start():match.end()] = bytes(len(span))
    return bytes(data), changed


def generate_uav_mutations(main_image, external_image):
    parent_hash = calculate_payload_hash(main_image, external_image)
    plans = {f"uav_remove_{group}": (group,) for group in GROUP_PATTERNS}
    plans["uav_remove_safety_groups"] = SAFETY_GROUPS

    mutations = []
    for kind, groups in plans.items():
        new_main, changed_main = blank_markers(main_image, groups)
        new_external, changed_external = blank_markers(
            external_image, groups
        )
        changed = changed_main + changed_external
        if changed == 0:
            continue  # the marker group was not present in this firmware
        mutations.append({
            "mutation_type": kind,
            "changed_bytes": changed,
            "parent_payload_sha256": parent_hash,
            "modified_payload_sha256": calculate_payload_hash(
                new_main, new_external
            ),
            "main_image": new_main,
            "external_image": new_external,
        })
    return mutations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation")
    args = parser.parse_args()

    output_path = PROCESSED_DIR / f"{args.split}_stress_features.csv"
    if output_path.exists():
        raise SystemExit(f"{output_path.name} already exists; keep it.")

    config = json.loads(
        (REPORT_DIR / "split_config.json").read_text(encoding="utf-8")
    )
    names = config["feature_names"]
    parents = pd.read_csv(PROCESSED_DIR / f"{args.split}_features.csv")

    if args.split == "test":
        print("NOTE: building test-split stress samples. Do this once, "
              "after the configuration is frozen.")

    records = []
    for index, (_, parent) in enumerate(parents.iterrows(), start=1):
        path = PROJECT_ROOT / "data" / "raw" / parent["meta_relative_path"]
        firmware = load_firmware(path)

        if firmware["payload_sha256"] != parent["meta_payload_sha256"]:
            raise ValueError(f"Original payload changed: {path.name}")

        original, _ = extract_firmware_features(
            firmware["main_image"], firmware["external_image"]
        )
        if set(original) != set(names):
            raise ValueError("Feature schema mismatch.")
        if not np.allclose(
            [original[n] for n in names],
            parent[names].to_numpy(dtype=float),
            rtol=1e-9, atol=1e-12,
        ):
            raise ValueError(f"Feature extraction changed: {path.name}")

        common = {
            "meta_family": parent["meta_family"],
            "meta_vehicle_class": parent["meta_vehicle_class"],
            "meta_release": parent["meta_release"],
            "meta_parent_payload_sha256": parent["meta_payload_sha256"],
        }
        records.append({
            **common, "meta_kind": "clean", "meta_changed_bytes": 0,
            "meta_modified_payload_sha256": parent["meta_payload_sha256"],
            **original,
        })

        mutations = (
            generate_mutations(
                firmware["main_image"], firmware["external_image"]
            )
            + generate_uav_mutations(
                firmware["main_image"], firmware["external_image"]
            )
        )
        for mutation in mutations:
            features, _ = extract_firmware_features(
                mutation["main_image"], mutation["external_image"]
            )
            records.append({
                **common,
                "meta_kind": mutation["mutation_type"],
                "meta_changed_bytes": mutation["changed_bytes"],
                "meta_modified_payload_sha256":
                    mutation["modified_payload_sha256"],
                **features,
            })

        if index % 10 == 0:
            print(f"Processed {index}/{len(parents)} parents", flush=True)

    samples = pd.DataFrame(records)
    if not np.isfinite(samples[names].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite modified features.")
    samples.to_csv(output_path, index=False)

    print(f"\nParents: {len(parents)}  Total stress rows: {len(samples)}")
    print("\nRows by modification kind:")
    print(samples["meta_kind"].value_counts().to_string())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
