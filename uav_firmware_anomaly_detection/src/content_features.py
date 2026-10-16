import re
import numpy as np


WINDOW_BYTES = 4096
BIGRAM_BINS = 128


def calculate_entropy(array):
    counts = np.bincount(array, minlength=256)
    probabilities = counts[counts > 0] / len(array)

    return float(
        -np.sum(probabilities * np.log2(probabilities))
    )


def extract_content_features(main_image, external_image=b""):
    regions = [image for image in (main_image, external_image) if image]

    if not regions:
        raise ValueError("Cannot analyze an empty firmware payload")

    total_bytes = sum(len(image) for image in regions)
    string_lengths = []
    window_entropies = []
    bigram_counts = np.zeros(BIGRAM_BINS, dtype=np.int64)
    total_pairs = 0

    for image in regions:
        string_lengths.extend(
            len(match.group())
            for match in re.finditer(rb"[\x20-\x7e]{4,}", image)
        )

        array = np.frombuffer(image, dtype=np.uint8)

        for start in range(0, len(array), WINDOW_BYTES):
            window = array[start:start + WINDOW_BYTES]
            window_entropies.append(calculate_entropy(window))

        if len(array) > 1:
            pairs = (
                array[:-1].astype(np.uint32) * np.uint32(256)
                + array[1:].astype(np.uint32)
            )

            # Fixed hash: no vocabulary is learned from the dataset.
            hashed = pairs * np.uint32(2654435761)
            bins = (hashed >> np.uint32(25)).astype(np.int64)

            bigram_counts += np.bincount(
                bins, minlength=BIGRAM_BINS
            )
            total_pairs += len(pairs)

    lengths = np.asarray(string_lengths, dtype=float)
    entropies = np.asarray(window_entropies, dtype=float)

    features = {
        "strings_per_kib": len(lengths) / (total_bytes / 1024),
        "string_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "string_length_std": float(lengths.std()) if len(lengths) else 0.0,
        "string_length_max": float(lengths.max()) if len(lengths) else 0.0,
        "printable_string_byte_ratio": float(lengths.sum() / total_bytes),
        "long_string_fraction": float(
            (lengths >= 32).mean()
        ) if len(lengths) else 0.0,
        "window_entropy_mean": float(entropies.mean()),
        "window_entropy_std": float(entropies.std()),
        "window_entropy_min": float(entropies.min()),
        "window_entropy_max": float(entropies.max()),
        "window_entropy_p90": float(np.quantile(entropies, 0.9)),
        "high_entropy_window_fraction": float((entropies >= 7.5).mean()),
    }

    frequencies = bigram_counts / max(total_pairs, 1)

    for index, frequency in enumerate(frequencies):
        features[f"hashed_bigram_{index:03d}"] = float(frequency)

    return features
