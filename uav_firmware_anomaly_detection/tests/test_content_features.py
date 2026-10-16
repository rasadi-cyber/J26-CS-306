from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from content_features import extract_content_features


class ContentFeatureTests(unittest.TestCase):
    def test_uniform_windows_have_eight_bit_entropy(self):
        features = extract_content_features(bytes(range(256)) * 32)

        self.assertEqual(len(features), 140)
        self.assertAlmostEqual(features["window_entropy_mean"], 8.0)
        self.assertAlmostEqual(features["window_entropy_std"], 0.0)
        self.assertEqual(features["high_entropy_window_fraction"], 1.0)

    def test_local_windows_capture_different_regions(self):
        image = b"\x00" * 4096 + bytes(range(256)) * 16
        features = extract_content_features(image)

        self.assertAlmostEqual(features["window_entropy_min"], 0.0)
        self.assertAlmostEqual(features["window_entropy_max"], 8.0)
        self.assertAlmostEqual(features["window_entropy_mean"], 4.0)
        self.assertAlmostEqual(
            features["high_entropy_window_fraction"], 0.5
        )

    def test_string_length_measurements(self):
        image = b"ABCD\x00ABCDEFGH\x00"
        features = extract_content_features(image)

        self.assertAlmostEqual(features["string_length_mean"], 6.0)
        self.assertAlmostEqual(features["string_length_std"], 2.0)
        self.assertEqual(features["string_length_max"], 8.0)
        self.assertAlmostEqual(
            features["printable_string_byte_ratio"], 12 / 14
        )

    def test_strings_cannot_cross_flash_regions(self):
        features = extract_content_features(b"AB", b"CD")

        # Each region has only two printable bytes, below the minimum.
        self.assertEqual(features["strings_per_kib"], 0.0)
        self.assertEqual(features["printable_string_byte_ratio"], 0.0)

    def test_bigrams_cannot_cross_flash_regions(self):
        features = extract_content_features(b"A", b"B")

        # Each region contains one byte, so neither has a byte pair.
        frequency_sum = sum(
            value
            for name, value in features.items()
            if name.startswith("hashed_bigram_")
        )
        self.assertEqual(frequency_sum, 0.0)

    def test_external_only_and_repeatability(self):
        image = b"firmware strings\x00" + bytes(range(256))
        first = extract_content_features(b"", image)
        second = extract_content_features(b"", image)

        self.assertEqual(first, second)
        self.assertTrue(
            all(math.isfinite(value) for value in first.values())
        )

        frequency_sum = sum(
            value
            for name, value in first.items()
            if name.startswith("hashed_bigram_")
        )
        self.assertAlmostEqual(frequency_sum, 1.0)

    def test_empty_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_content_features(b"", b"")


if __name__ == "__main__":
    unittest.main()