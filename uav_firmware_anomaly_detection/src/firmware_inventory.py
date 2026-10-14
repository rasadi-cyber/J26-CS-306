from pathlib import Path
import csv
import hashlib

from firmware_loader import load_firmware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"


def calculate_sha256(path):
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():
    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".apj", ".px4"}
    )

    if not files:
        raise SystemExit("No APJ or PX4 files found in data/raw.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    seen_containers = {}
    seen_payloads = {}
    rows = []

    for path in files:
        relative_path = path.relative_to(RAW_DIR).as_posix()
        container_hash = calculate_sha256(path)

        row = {
            "relative_path": relative_path,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "container_sha256": container_hash,
            "exact_duplicate_of": seen_containers.get(container_hash, ""),
            "validation_status": "",
            "validation_error": "",
            "main_image_bytes": "",
            "external_image_bytes": "",
            "payload_sha256": "",
            "payload_duplicate_of": "",
            "board_id": "",
            "summary": "",
            "git_identity": "",
        }

        seen_containers.setdefault(container_hash, relative_path)

        try:
            result = load_firmware(path)
            payload_hash = result["payload_sha256"]
            metadata = result["metadata"]

            row.update({
                "validation_status": "valid_container",
                "main_image_bytes": len(result["main_image"]),
                "external_image_bytes": len(result["external_image"]),
                "payload_sha256": payload_hash,
                "payload_duplicate_of": seen_payloads.get(payload_hash, ""),
                "board_id": metadata.get("board_id"),
                "summary": metadata.get("summary"),
                "git_identity": metadata.get("git_identity"),
            })

            seen_payloads.setdefault(payload_hash, relative_path)

        except (ValueError, TypeError, OSError, EOFError) as error:
            row["validation_status"] = "error"
            row["validation_error"] = str(error)

        rows.append(row)

    output_path = REPORT_DIR / "firmware_inventory_validated.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        row for row in rows
        if row["validation_status"] == "valid_container"
    ]

    print(f"Firmware files scanned: {len(rows)}")
    print(f"Unique original files: {len(seen_containers)}")
    print(f"Successfully decoded containers: {len(valid_rows)}")
    print(f"Validation errors: {len(rows) - len(valid_rows)}")
    print(f"Unique complete payloads: {len(seen_payloads)}")
    print(
        "Duplicate payload files:",
        sum(bool(row["payload_duplicate_of"]) for row in valid_rows),
    )
    print(
        "Files with external-flash content:",
        sum(row["external_image_bytes"] > 0 for row in valid_rows),
    )
    print(f"Inventory saved to: {output_path}")

    for row in rows:
        if row["validation_status"] == "error":
            print(
                f"ERROR: {row['relative_path']}"
                f" -> {row['validation_error']}"
            )
        elif row["payload_duplicate_of"]:
            print(
                f"PAYLOAD DUPLICATE: {row['relative_path']}"
                f" -> {row['payload_duplicate_of']}"
            )


if __name__ == "__main__":
    main()