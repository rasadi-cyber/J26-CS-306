import hashlib
import numpy as np

from firmware_loader import calculate_payload_hash


def generate_mutations(main_image, external_image=b""):
    if not main_image and not external_image:
        raise ValueError("Cannot modify an empty payload")

    parent_hash = calculate_payload_hash(main_image, external_image)

    # Modify the larger nonempty flash region.
    region = (
        "main" if len(main_image) >= len(external_image) else "external"
    )
    original = main_image if region == "main" else external_image

    length = len(original)
    offset = length // 2
    block_size = max(1, length // 100)
    block_size = min(block_size, length - offset)

    seed = int(parent_hash[:16], 16)
    rng = np.random.default_rng(seed)

    random_block = rng.integers(
        0, 256, size=block_size, dtype=np.uint8
    ).tobytes()

    # Ensure every byte in this replacement block differs.
    original_block = original[offset:offset + block_size]
    random_block = bytes(
        candidate if candidate != previous else previous ^ 1
        for previous, candidate in zip(original_block, random_block)
    )

    replacements = {
        "single_byte_flip": (
            original[:offset]
            + bytes([original[offset] ^ 1])
            + original[offset + 1:]
        ),
        "random_region_1pct": (
            original[:offset]
            + random_block
            + original[offset + block_size:]
        ),
        "zero_region_1pct": (
            original[:offset]
            + bytes(block_size)
            + original[offset + block_size:]
        ),
        "append_random_1pct": original + random_block,
    }

    mutations = []

    for mutation_type, modified in replacements.items():
        if modified == original:
            # For example, zeroing an already-zero region changes nothing.
            continue

        changed_bytes = sum(
            first != second
            for first, second in zip(original, modified)
        ) + abs(len(modified) - len(original))

        new_main = modified if region == "main" else main_image
        new_external = (
            modified if region == "external" else external_image
        )

        mutations.append({
            "mutation_type": mutation_type,
            "region": region,
            "offset": (
                length if mutation_type == "append_random_1pct"
                else offset
            ),
            "changed_bytes": changed_bytes,
            "parent_payload_sha256": parent_hash,
            "modified_payload_sha256": calculate_payload_hash(
                new_main, new_external
            ),
            "main_image": new_main,
            "external_image": new_external,
        })

    return mutations