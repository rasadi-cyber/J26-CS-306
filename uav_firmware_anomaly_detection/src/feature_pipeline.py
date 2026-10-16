import math

from binary_features import extract_binary_features
from content_features import extract_content_features
from uav_features import extract_uav_features


def extract_firmware_features(main_image, external_image=b""):
    uav_features, evidence = extract_uav_features(
        main_image, external_image
    )

    groups = [
        extract_binary_features(main_image, external_image),
        extract_content_features(main_image, external_image),
        uav_features,
    ]

    features = {}

    for group in groups:
        if features.keys() & group.keys():
            raise ValueError("Duplicate feature names.")

        features.update(group)

    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Non-finite extracted features.")

    return features, evidence