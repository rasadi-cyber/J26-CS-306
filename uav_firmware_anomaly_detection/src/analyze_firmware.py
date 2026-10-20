from pathlib import Path
from datetime import datetime, timezone
from xml.etree.ElementTree import ParseError
import argparse
import hashlib
import json
import time
import zlib

import joblib
import numpy as np
import pandas as pd

from feature_pipeline import extract_firmware_features
from firmware_loader import load_firmware
from px4_consistency import check_px4_consistency


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def analyze_firmware(file_path, model_directory=MODEL_DIR):
    started = time.perf_counter()
    path = Path(file_path)

    firmware = load_firmware(path)
    features, uav_evidence = extract_firmware_features(
        firmware["main_image"],
        firmware["external_image"],
    )

    container = json.loads(path.read_text(encoding="utf-8-sig"))

    try:
        consistency = check_px4_consistency(
            container,
            firmware["main_image"],
            firmware["external_image"],
        )
    except (ValueError, TypeError, zlib.error, ParseError) as error:
        consistency = {
            "status": "invalid_metadata",
            "reason": str(error),
            "groups": {},
        }

    family = {
        "APJFWv1": "ArduPilot",
        "PX4FWv1": "PX4",
    }[firmware["metadata"]["magic"]]

    model_path = (
        Path(model_directory)
        / f"ocsvm_tuned_{family}_combined.joblib"
    )

    if not model_path.is_file():
        raise ValueError(
            f"No saved combined model available for {family}"
        )

    # Load only trusted model files produced by this project.
    bundle = joblib.load(model_path)

    if bundle["family"] != family:
        raise ValueError("Firmware/model family mismatch")

    names = bundle["feature_names"]

    if set(features) != set(names):
        raise ValueError(
            "Extracted features do not match the model schema"
        )

    frame = pd.DataFrame(
        [[features[name] for name in names]],
        columns=names,
    )

    score = float(
        -bundle["pipeline"].decision_function(frame)[0]
    )
    threshold = float(bundle["threshold"])
    verdict = "anomalous" if score > threshold else "normal"

    retained = bundle["pipeline"].named_steps[
        "remove_constants"
    ].get_support()

    deviations = []
    inactive_changes = []

    for index, name in enumerate(names):
        observed = float(features[name])
        median = float(bundle["training_medians"][name])
        spread = float(
            bundle["training_standard_deviations"][name]
        )

        if not retained[index]:
            if not np.isclose(
                observed,
                median,
                rtol=1e-9,
                atol=1e-12,
            ):
                inactive_changes.append(name)
            continue

        if spread <= 0:
            continue

        deviation = (observed - median) / spread

        deviations.append({
            "feature": name,
            "observed": observed,
            "training_median": median,
            "training_standard_deviation": spread,
            "standardized_deviation_from_median": float(deviation),
            "direction": "higher" if deviation > 0 else "lower",
        })

    deviations.sort(
        key=lambda item: abs(
            item["standardized_deviation_from_median"]
        ),
        reverse=True,
    )

    supporting_deviations = [
        item
        for item in deviations
        if abs(item["standardized_deviation_from_median"]) >= 2
    ][:8]

    calibration_scores = np.asarray(
        bundle["validation_scores"],
        dtype=float,
    )
    percentile = float(
        100 * np.mean(calibration_scores <= score)
    )

    warnings = [
        "Exploratory model; detection performance remains limited.",
        "A normal result does not establish authenticity or absence of malware.",
        "An anomaly indicates statistical deviation, not proof of maliciousness.",
        "UAV marker presence does not establish enabled settings or runtime behaviour.",
        "New boards, releases and build configurations may differ legitimately.",
        "Training provenance is not verified for every candidate sample.",
    ]

    if inactive_changes:
        warnings.append(
            "Some training-constant features changed but were excluded "
            "from model scoring."
        )

    if consistency["status"] == "mismatch":
        warnings.append(
            "PX4 parameter metadata and binary names disagree. "
            "This finding is separate from the ML score and requires review."
        )
    elif consistency["status"] == "invalid_metadata":
        warnings.append(
            "PX4 metadata could not be validated. "
            "The consistency check is unavailable; the ML result is separate."
        )

    model_hash = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    return {
        "schema_version": "1.1",
        "component": "uav_firmware_anomaly_detection",
        "analysis_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "firmware": {
            "file_name": path.name,
            "family": family,
            "container_sha256": firmware["container_sha256"],
            "payload_sha256": firmware["payload_sha256"],
            "main_image_bytes": len(firmware["main_image"]),
            "external_image_bytes": len(
                firmware["external_image"]
            ),
            "metadata": firmware["metadata"],
        },
        "result": {
            "verdict": verdict,
            "anomaly_score": score,
            "decision_threshold": threshold,
            "score_definition": (
                "Negative One-Class SVM decision_function; "
                "larger means more anomalous"
            ),
            "decision_rule": (
                "anomalous when anomaly_score > threshold"
            ),
            "validation_percentile": percentile,
            "percentile_definition": (
                "Rank against candidate-clean family validation scores; "
                "not a probability of compromise"
            ),
        },
        "model": {
            "model_id": f"ocsvm-{model_hash[:12]}",
            "model_file_sha256": model_hash,
            "family": family,
            "representation": bundle["representation"],
            "experiment": bundle["experiment"],
            "training_dataset_sha256": bundle[
                "dataset_sha256"
            ],
            "candidate_features": len(names),
            "retained_features": int(retained.sum()),
        },
        "explanation": {
            "decision_basis": (
                "The saved model score was compared with its "
                "validation-derived threshold."
            ),
            "method": (
                "Individual feature deviations from training medians. "
                "These are supporting observations, not causal model attribution."
            ),
            "supporting_feature_deviations": supporting_deviations,
            "inactive_feature_changes": inactive_changes,
        },
        "uav_evidence": uav_evidence,
        "px4_parameter_name_consistency": consistency,
        "warnings": warnings,
        "analysis_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("firmware")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)

    if output.resolve() == Path(args.firmware).resolve():
        raise ValueError("Report must not overwrite the firmware")

    if output.exists():
        raise ValueError(
            "Report already exists; choose a new output name"
        )

    report = analyze_firmware(args.firmware)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Firmware:", report["firmware"]["file_name"])
    print("Family:", report["firmware"]["family"])
    print("Verdict:", report["result"]["verdict"])
    print("Anomaly score:", report["result"]["anomaly_score"])
    print("Threshold:", report["result"]["decision_threshold"])
    print(
        "Supporting feature deviations:",
        len(report["explanation"]["supporting_feature_deviations"]),
    )
    print(
        "PX4 parameter-name consistency:",
        report["px4_parameter_name_consistency"]["status"],
    )
    print("Report saved to:", output)


if __name__ == "__main__":
    main()