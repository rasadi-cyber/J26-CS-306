from pathlib import Path
import csv
import json
import math

from binary_features import extract_binary_features
from firmware_loader import load_firmware
from uav_features import extract_uav_features
from content_features import extract_content_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".apj", ".px4"}
    )

    if not files:
        raise SystemExit("No firmware files found.")

    rows = []
    evidence_records = []
    seen_payloads = set()
    expected_feature_names = None
    duplicate_count = 0

    for path in files:
        firmware = load_firmware(path)
        payload_hash = firmware["payload_sha256"]

        if payload_hash in seen_payloads:
            duplicate_count += 1
            continue

        main_image = firmware["main_image"]
        external_image = firmware["external_image"]

        binary_features = extract_binary_features(
            main_image, external_image
        )
        uav_features, evidence = extract_uav_features(
            main_image, external_image
        )

        content_features = extract_content_features(
            main_image, external_image
        )

        feature_groups = [
            binary_features,
            content_features,
            uav_features,
        ]

        features = {}

        for group in feature_groups:
            if features.keys() & group.keys():
                raise ValueError("Feature names overlap between extractors.")

            features.update(group)

        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError(f"Non-finite feature found in {path.name}")

        if expected_feature_names is None:
            expected_feature_names = list(features)
        elif list(features) != expected_feature_names:
            raise ValueError(f"Inconsistent feature schema: {path.name}")

        relative_path = path.relative_to(RAW_DIR).as_posix()
        metadata = firmware["metadata"]

        row = {
            "meta_relative_path": relative_path,
            "meta_family": (
                "ArduPilot" if path.suffix.lower() == ".apj" else "PX4"
            ),
            "meta_board_id": metadata.get("board_id"),
            "meta_git_identity": metadata.get("git_identity"),
            "meta_container_sha256": firmware["container_sha256"],
            "meta_payload_sha256": payload_hash,
            **features,
        }

        rows.append(row)
        evidence_records.append({
            "relative_path": relative_path,
            "payload_sha256": payload_hash,
            "uav_evidence": evidence,
        })
        seen_payloads.add(payload_hash)

        if len(rows) % 25 == 0:
            print(f"Extracted {len(rows)} unique payloads", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / "firmware_features.csv"
    with features_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evidence_path = OUTPUT_DIR / "uav_evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8") as file:
        for record in evidence_records:
            file.write(json.dumps(record) + "\n")

    print(f"\nFiles scanned: {len(files)}")
    print(f"Duplicate payloads skipped: {duplicate_count}")
    print(f"Unique feature rows: {len(rows)}")
    print(f"Numeric features per row: {len(expected_feature_names)}")
    print(f"Features saved to: {features_path}")
    print(f"Evidence saved to: {evidence_path}")


if __name__ == "__main__":
    main()