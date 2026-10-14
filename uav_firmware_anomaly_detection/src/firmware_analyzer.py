
from pathlib import Path
import hashlib
import math


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as firmware:
        for chunk in iter(lambda: firmware.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_static_features(file_path):
    """Extract basic static features without executing the firmware."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Firmware file not found: {path}")

    file_size = path.stat().st_size
    sha256_hash = calculate_sha256(path)

    byte_counts = [0] * 256
    total_bytes = 0
    printable_count = 0

    with path.open("rb") as firmware:
        while True:
            chunk = firmware.read(1024 * 1024)
            if not chunk:
                break

            total_bytes += len(chunk)

            for byte in chunk:
                byte_counts[byte] += 1
                if 32 <= byte <= 126:
                    printable_count += 1

    entropy = 0.0

    if total_bytes:
        for count in byte_counts:
            if count:
                probability = count / total_bytes
                entropy -= probability * math.log2(probability)

    return {
        "file_name": path.name,
        "file_size_bytes": file_size,
        "sha256": sha256_hash,
        "entropy_bits_per_byte": round(entropy, 4),
        "printable_byte_ratio": round(
            printable_count / total_bytes, 4
        ) if total_bytes else 0.0,
    }
