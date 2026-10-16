import numpy as np


def extract_binary_features(main_image, external_image=b""):
    """Extract numeric features from decoded flash images."""
    total_bytes = len(main_image) + len(external_image)

    if total_bytes == 0:
        raise ValueError("Cannot extract features from an empty payload")

    byte_counts = np.zeros(256, dtype=np.int64)

    # Count each region separately without creating artificial boundaries.
    for image in (main_image, external_image):
        if image:
            array = np.frombuffer(image, dtype=np.uint8)
            byte_counts += np.bincount(array, minlength=256)

    frequencies = byte_counts / total_bytes

    nonzero_frequencies = frequencies[frequencies > 0]
    entropy = float(
        -np.sum(nonzero_frequencies * np.log2(nonzero_frequencies))
    )

    features = {
        "payload_size_bytes": total_bytes,
        "main_image_size_bytes": len(main_image),
        "external_image_size_bytes": len(external_image),
        "external_image_ratio": len(external_image) / total_bytes,
        "entropy_bits_per_byte": entropy,
        "printable_byte_ratio": float(frequencies[32:127].sum()),
        "zero_byte_ratio": float(frequencies[0]),
        "ff_byte_ratio": float(frequencies[255]),
    }

    for byte_value, frequency in enumerate(frequencies):
        features[f"byte_frequency_{byte_value:03d}"] = float(frequency)

    return features