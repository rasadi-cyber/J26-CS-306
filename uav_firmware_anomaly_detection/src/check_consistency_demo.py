from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import pandas as pd

from audit_px4_metadata import decode_xml, elements_named
from firmware_loader import load_firmware
from px4_consistency import check_px4_consistency, GROUPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/train_features.csv"
    )
    sample = train[train["meta_family"] == "PX4"].iloc[0]

    path = PROJECT_ROOT / "data/raw" / sample["meta_relative_path"]
    original_file = path.read_bytes()
    original_hash = hashlib.sha256(original_file).hexdigest()

    if original_hash != sample["meta_container_sha256"]:
        raise ValueError("Training file differs from its recorded hash.")

    container = json.loads(original_file.decode("utf-8-sig"))
    firmware = load_firmware(path)
    main_image = firmware["main_image"]
    external_image = firmware["external_image"]

    document, _ = decode_xml(container, "parameter_xml")
    names = {
        item.get("name")
        for item in elements_named(document, "parameter")
        if item.get("name")
    }

    clean = check_px4_consistency(
        container, main_image, external_image
    )

    print("Training sample:", path.name)
    print("Original consistency:", clean["status"])

    if clean["status"] != "consistent":
        raise ValueError("This demonstration requires a consistent original.")

    original_counts = Counter(main_image) + Counter(external_image)

    for group, prefixes in GROUPS.items():
        candidates = sorted(
            name for name in names if name.startswith(prefixes)
        )

        if not candidates:
            print(group, ": no declared names available")
            continue

        name = candidates[0]
        replacement = name[::-1]

        if replacement == name or replacement in names:
            raise ValueError("Cannot use this reversed-name replacement.")

        pattern = re.compile(
            rb"(?<![A-Za-z0-9_])"
            + re.escape(name.encode("ascii"))
            + rb"(?![A-Za-z0-9_])"
        )

        modified_main, main_matches = pattern.subn(
            replacement.encode("ascii"), main_image
        )
        modified_external, external_matches = pattern.subn(
            replacement.encode("ascii"), external_image
        )

        if main_matches + external_matches == 0:
            raise ValueError(f"Declared name not found: {name}")

        assert len(modified_main) == len(main_image)
        assert len(modified_external) == len(external_image)
        assert (
            Counter(modified_main) + Counter(modified_external)
            == original_counts
        )

        result = check_px4_consistency(
            container, modified_main, modified_external
        )
        mismatches = [
            key
            for key, value in result["groups"].items()
            if value["status"] == "mismatch"
        ]

        print(f"\nGroup: {group}")
        print("Name changed:", name, "->", replacement)
        print("Occurrences changed:", main_matches + external_matches)
        print("Detected mismatch groups:", mismatches)
        print("Payload sizes and byte frequencies preserved: YES")

        assert mismatches == [group], "Unexpected mismatch result"

    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash

    print("\nOriginal file unchanged: YES")
    print("No modified firmware saved. Reserved test files NOT inspected.")


if __name__ == "__main__":
    main()