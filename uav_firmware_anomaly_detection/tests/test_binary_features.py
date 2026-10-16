from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from binary_features import extract_binary_features


class BinaryFeatureTests(unittest.TestCase):
    def test_uniform_byte_distribution(self):
        features = extract_binary_features(bytes(range(256)))

        self.assertEqual(len(features), 264)
        self.assertAlmostEqual(features["entropy_bits_per_byte"], 8.0)

        for byte_value in range(256):
            self.assertAlmostEqual(
                features[f"byte_frequency_{byte_value:03d}"],
                1 / 256,
            )

    def test_constant_bytes_have_zero_entropy(self):
        features = extract_binary_features(b"\x00" * 100)

        self.assertAlmostEqual(features["entropy_bits_per_byte"], 0.0)
        self.assertEqual(features["zero_byte_ratio"], 1.0)
        self.assertEqual(features["printable_byte_ratio"], 0.0)

    def test_both_flash_regions_contribute(self):
        features = extract_binary_features(
            b"\x00" * 3,
            b"\xff",
        )

        self.assertEqual(features["payload_size_bytes"], 4)
        self.assertEqual(features["main_image_size_bytes"], 3)
        self.assertEqual(features["external_image_size_bytes"], 1)
        self.assertAlmostEqual(features["external_image_ratio"], 0.25)
        self.assertAlmostEqual(features["zero_byte_ratio"], 0.75)
        self.assertAlmostEqual(features["ff_byte_ratio"], 0.25)
        self.assertAlmostEqual(
            features["entropy_bits_per_byte"],
            0.8112781244591328,
        )

    def test_external_only_payload(self):
        features = extract_binary_features(b"", b"AAAA")

        self.assertEqual(features["main_image_size_bytes"], 0)
        self.assertEqual(features["external_image_ratio"], 1.0)
        self.assertEqual(features["printable_byte_ratio"], 1.0)

    def test_printable_byte_boundaries(self):
        features = extract_binary_features(bytes([31, 32, 126, 127]))

        self.assertEqual(features["printable_byte_ratio"], 0.5)

    def test_empty_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_binary_features(b"", b"")

    def test_features_are_finite_and_repeatable(self):
        main_image = bytes(range(256)) * 2
        external_image = b"flight firmware"

        first = extract_binary_features(main_image, external_image)
        second = extract_binary_features(main_image, external_image)

        self.assertEqual(first, second)
        self.assertTrue(
            all(math.isfinite(value) for value in first.values())
        )

        frequency_sum = sum(
            value
            for name, value in first.items()
            if name.startswith("byte_frequency_")
        )
        self.assertAlmostEqual(frequency_sum, 1.0)


if __name__ == "__main__":
    unittest.main()