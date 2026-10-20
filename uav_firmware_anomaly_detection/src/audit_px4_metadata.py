from pathlib import Path
import base64
import hashlib
import json
import zlib
import xml.etree.ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_XML_BYTES = 16 * 1024 * 1024


def decode_xml(container, field):
    encoded = container.get(field)

    if not encoded:
        return None, b""

    compressed = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    content = decoder.decompress(compressed, MAX_XML_BYTES + 1)

    if len(content) > MAX_XML_BYTES or not decoder.eof:
        raise ValueError(f"Invalid or oversized {field}")

    if decoder.unused_data:
        raise ValueError(f"Trailing compressed data in {field}")

    declared = container.get(field + "_size")
    if declared is not None and len(content) != declared:
        raise ValueError(f"Decoded size mismatch in {field}")

    if not content.strip():
        return None, content

    if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ValueError("Unsupported XML declaration")

    return ET.fromstring(content), content


def elements_named(document, name):
    if document is None:
        return []

    return [
        element
        for element in document.iter()
        if element.tag.rsplit("}", 1)[-1] == name
    ]


def main():
    train = pd.read_csv(
        PROJECT_ROOT / "data/processed/train_features.csv"
    )
    px4_train = train[train["meta_family"] == "PX4"]

    prefixes = {
        "flight_control": ("MC_", "FW_R_", "FW_P_", "FW_Y_", "CA_"),
        "navigation_rtl": ("RTL_", "MIS_"),
        "arming_failsafe": ("COM_ARM_", "COM_FAIL_", "NAV_RCL_", "NAV_DLL_"),
        "geofence": ("GF_",),
    }

    rows = []

    for _, sample in px4_train.iterrows():
        path = PROJECT_ROOT / "data/raw" / sample["meta_relative_path"]
        original = path.read_bytes()

        if hashlib.sha256(original).hexdigest() != sample["meta_container_sha256"]:
            raise ValueError(f"Training file changed: {path.name}")

        container = json.loads(original.decode("utf-8-sig"))
        parameter_root, parameter_bytes = decode_xml(
            container, "parameter_xml"
        )
        airframe_root, airframe_bytes = decode_xml(
            container, "airframe_xml"
        )

        parameters = elements_named(parameter_root, "parameter")
        airframes = elements_named(airframe_root, "airframe")

        row = {
            "relative_path": sample["meta_relative_path"],
            "build_identity": sample["meta_git_identity"],
            "parameter_definitions": len(parameters),
            "airframe_definitions": len(airframes),
            "parameter_xml_sha256": hashlib.sha256(
                parameter_bytes
            ).hexdigest(),
            "airframe_xml_sha256": hashlib.sha256(
                airframe_bytes
            ).hexdigest(),
        }

        for group, group_prefixes in prefixes.items():
            row[f"{group}_definitions"] = sum(
                element.get("name", "").startswith(group_prefixes)
                for element in parameters
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("No PX4 training samples found")

    output = PROJECT_ROOT / "reports/px4_training_metadata_audit.csv"
    result.to_csv(output, index=False)

    print("PX4 training files inspected:", len(result))
    print("\nMetadata diversity by build:")
    print(result.groupby("build_identity").agg(
        samples=("relative_path", "size"),
        parameter_xml_variants=("parameter_xml_sha256", "nunique"),
        airframe_xml_variants=("airframe_xml_sha256", "nunique"),
        parameter_count_min=("parameter_definitions", "min"),
        parameter_count_max=("parameter_definitions", "max"),
        airframe_count_min=("airframe_definitions", "min"),
        airframe_count_max=("airframe_definitions", "max"),
    ).to_string())

    print("\nFlight-related definition counts:")
    columns = [
        "relative_path",
        *[f"{group}_definitions" for group in prefixes],
    ]
    print(result[columns].to_string(index=False))

    print("\nSaved:", output)
    print("Models and feature dataset unchanged. Test files NOT inspected.")


if __name__ == "__main__":
    main()
    