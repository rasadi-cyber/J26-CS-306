from pathlib import Path
import base64
import hashlib
import json
import zlib

import pandas as pd

from controlled_mutations import generate_mutations
from firmware_loader import load_firmware


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/validation_features.csv"
    )
    parent = validation[
        validation["meta_family"] == "ArduPilot"
    ].iloc[0]

    source = PROJECT_ROOT / "data/raw" / parent["meta_relative_path"]
    original_bytes = source.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    if original_hash != parent["meta_container_sha256"]:
        raise ValueError("Original file differs from the validation record.")

    firmware = load_firmware(source)

    mutation = next(
        (
            item
            for item in generate_mutations(
                firmware["main_image"], firmware["external_image"]
            )
            if item["mutation_type"] == "zero_region_1pct"
        ),
        None,
    )

    if mutation is None:
        raise ValueError("Zero-region modification would make no change.")

    container = json.loads(original_bytes.decode("utf-8-sig"))

    for image_key, size_key, image in (
        ("image", "image_size", mutation["main_image"]),
        ("extf_image", "extf_image_size", mutation["external_image"]),
    ):
        container[image_key] = base64.b64encode(
            zlib.compress(image)
        ).decode("ascii")
        container[size_key] = len(image)

    output = PROJECT_ROOT / "data/demo/controlled_zero_region.apj"

    if output.exists():
        raise ValueError("Demo sample already exists; keeping it unchanged.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(container), encoding="utf-8")

    decoded = load_firmware(output)
    assert decoded["payload_sha256"] == mutation["modified_payload_sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash

    manifest = {
        key: value
        for key, value in mutation.items()
        if key not in ("main_image", "external_image")
    }
    manifest["source_file"] = parent["meta_relative_path"]
    manifest["parent_split"] = "validation"
    manifest["purpose"] = "Synthetic UI demonstration; never deploy or flash"

    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Source:", source.name)
    print("Demo sample:", output)
    print("Changed bytes:", mutation["changed_bytes"])
    print("Original firmware unchanged: YES")
    print("Synthetic demonstration only; never flash this firmware.")


if __name__ == "__main__":
    main()