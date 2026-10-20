from pathlib import Path
import hashlib
import json
import re

import pandas as pd

from audit_px4_metadata import decode_xml, elements_named
from firmware_loader import load_firmware


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    "flight_control": ("MC_", "FW_R_", "FW_P_", "FW_Y_", "CA_"),
    "navigation_rtl": ("RTL_", "MIS_"),
    "arming_failsafe": ("COM_ARM_", "COM_FAIL_", "NAV_RCL_", "NAV_DLL_"),
    "geofence": ("GF_",),
}

TOKEN_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]{2,}(?![A-Za-z0-9_])"
)


def main():
    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/validation_features.csv"
    )
    rows = []
    missing_examples = {}

    for _, sample in train[train["meta_family"] == "PX4"].iterrows():
        path = PROJECT_ROOT / "data/raw" / sample["meta_relative_path"]
        original = path.read_bytes()

        if hashlib.sha256(original).hexdigest() != sample["meta_container_sha256"]:
            raise ValueError(f"Training file changed: {path.name}")

        container = json.loads(original.decode("utf-8-sig"))
        document, _ = decode_xml(container, "parameter_xml")

        if document is None:
            raise ValueError(f"No parameter metadata: {path.name}")

        declared_names = {
            element.get("name")
            for element in elements_named(document, "parameter")
            if element.get("name")
        }

        firmware = load_firmware(path)
        recovered_names = set()

        # Do not join regions: that could create artificial tokens.
        for image in (
            firmware["main_image"],
            firmware["external_image"],
        ):
            recovered_names.update(
                match.group().decode("ascii")
                for match in TOKEN_PATTERN.finditer(image)
            )

        row = {
            "relative_path": sample["meta_relative_path"],
            "build_identity": sample["meta_git_identity"],
        }

        for group, prefixes in GROUPS.items():
            expected = {
                name for name in declared_names
                if name.startswith(prefixes)
            }
            matched = expected & recovered_names
            missing = sorted(expected - recovered_names)

            row[f"{group}_declared"] = len(expected)
            row[f"{group}_recovered"] = len(matched)
            row[f"{group}_correspondence"] = (
                len(matched) / len(expected) if expected else None
            )

            if missing and group not in missing_examples:
                missing_examples[group] = {
                    "file": sample["meta_relative_path"],
                    "names": missing[:5],
                }

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No PX4 training samples found")

    output = PROJECT_ROOT / "reports/px4_validation_correspondence.csv"
    result.to_csv(output, index=False)

    columns = [
        f"{group}_correspondence" for group in GROUPS
    ]

    print("Validation samples inspected:", len(result))
    print("\nCorrespondence ranges by release:")
    print(
        result.groupby("build_identity")[columns]
        .agg(["min", "max"])
        .to_string()
    )

    print("\nNumber of distinct correspondence values:")
    print(result[columns].nunique().to_string())

    print("\nExamples of declared names not recovered:")
    for group, example in missing_examples.items():
        print(group, example)

    print("\nSaved:", output)
    print("Models unchanged. Reserved test files NOT inspected.")


if __name__ == "__main__":
    main()
