from pathlib import Path
import base64
import hashlib
import json
import zlib


MAX_CONTAINER_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


def decode_image(container, image_key, size_key):
    declared_size = container.get(size_key)

    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or not 0 <= declared_size <= MAX_IMAGE_BYTES
    ):
        raise ValueError(f"Invalid {size_key}: {declared_size}")

    encoded_image = container.get(image_key)

    if encoded_image is None and declared_size == 0:
        return b""

    if not isinstance(encoded_image, str):
        raise ValueError(f"Missing or invalid {image_key}")

    compressed = base64.b64decode(encoded_image, validate=True)

    decoder = zlib.decompressobj()
    image = decoder.decompress(compressed, MAX_IMAGE_BYTES + 1)

    if len(image) > MAX_IMAGE_BYTES:
        raise ValueError(f"{image_key} exceeds the size limit")

    if not decoder.eof:
        raise ValueError(f"{image_key} is incomplete or oversized")

    if decoder.unused_data:
        raise ValueError(f"{image_key} contains trailing compressed data")

    if len(image) != declared_size:
        raise ValueError(
            f"{image_key} size mismatch: "
            f"declared {declared_size}, decoded {len(image)}"
        )

    return image


def calculate_payload_hash(main_image, external_image):
    """Hash both flash regions, preserving their boundaries."""
    hasher = hashlib.sha256()
    hasher.update(b"uav-firmware-payload-v1")

    for region_name, image in (
        (b"main", main_image),
        (b"external", external_image),
    ):
        hasher.update(region_name)
        hasher.update(len(image).to_bytes(8, "big"))
        hasher.update(image)

    return hasher.hexdigest()


def load_firmware(file_path):
    path = Path(file_path)

    expected_magic = {
        ".apj": "APJFWv1",
        ".px4": "PX4FWv1",
    }.get(path.suffix.lower())

    if expected_magic is None:
        raise ValueError("Only APJ and PX4 containers are supported")

    if not 0 < path.stat().st_size <= MAX_CONTAINER_BYTES:
        raise ValueError("Container is empty or exceeds the size limit")

    original = path.read_bytes()
    container = json.loads(original.decode("utf-8"))

    if not isinstance(container, dict):
        raise ValueError("Firmware container must be a JSON object")

    if container.get("magic") != expected_magic:
        raise ValueError("Unexpected firmware container magic")

    main_image = decode_image(container, "image", "image_size")

    if "extf_image_size" in container or "extf_image" in container:
        external_image = decode_image(
            container, "extf_image", "extf_image_size"
        )
    else:
        external_image = b""

    if not main_image and not external_image:
        raise ValueError("Container has no firmware payload")

    return {
        "main_image": main_image,
        "external_image": external_image,
        "container_sha256": hashlib.sha256(original).hexdigest(),
        "payload_sha256": calculate_payload_hash(
            main_image, external_image
        ),
        "metadata": {
            key: container.get(key)
            for key in (
                "magic",
                "board_id",
                "board_revision",
                "summary",
                "description",
                "git_identity",
                "git_hash",
                "version",
            )
        },
    }