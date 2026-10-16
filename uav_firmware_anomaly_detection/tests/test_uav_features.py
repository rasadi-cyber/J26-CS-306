from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_features import extract_uav_features


class UavFeatureTests(unittest.TestCase):
    def test_feature_count_and_empty_evidence(self):
        features, evidence = extract_uav_features(b"\x00" * 100)

        self.assertEqual(len(features), 60)
        self.assertTrue(all(value == 0 for value in features.values()))
        self.assertTrue(all(not items for items in evidence.values()))

    def test_accidental_strings_are_rejected(self):
        features, _ = extract_uav_features(b"eKfJ\x00<GF_@\x00")

        self.assertTrue(all(value == 0 for value in features.values()))

    def test_parameter_matching_is_case_sensitive(self):
        features, _ = extract_uav_features(
            b"ATC_RAT_RLL_P\x00atc_rat_rll_p\x00"
        )

        prefix = "uav_flight_control_attitude_parameters"
        self.assertEqual(features[f"{prefix}_present"], 1)

        expected_density = 1 / (
            len(b"ATC_RAT_RLL_P\x00atc_rat_rll_p\x00") / (1024 * 1024)
        )
        self.assertAlmostEqual(
            features[f"{prefix}_count_per_mib"],
            expected_density,
        )

    def test_external_region_evidence_and_offset(self):
        features, evidence = extract_uav_features(
            b"\x00",
            b"\x00\x00FENCE_RADIUS\x00",
        )

        self.assertEqual(
            features["uav_geofence_fence_parameters_present"], 1
        )

        item = next(
            item
            for item in evidence["geofence"]
            if item["marker"] == "fence_parameters"
        )

        self.assertEqual(item["examples"][0]["region"], "external")
        self.assertEqual(item["examples"][0]["offset"], 2)
        self.assertEqual(
            item["examples"][0]["matched_text"], "FENCE_RADIUS"
        )

    def test_markers_cannot_cross_flash_regions(self):
        features, _ = extract_uav_features(b"ATC_", b"RAT_RLL_P")

        self.assertEqual(
            features["uav_flight_control_attitude_parameters_present"],
            0,
        )

    def test_coverage_counts_marker_types(self):
        features, _ = extract_uav_features(
            b"ATC_RAT_RLL_P\x00ATC_RAT_PIT_P\x00"
        )

        # Two parameter matches belong to one of five marker types.
        self.assertAlmostEqual(
            features["uav_flight_control_coverage"], 0.2
        )

    def test_empty_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_uav_features(b"", b"")

    def test_results_are_finite_and_repeatable(self):
        image = b"RTL_ALT\x00ARMING_CHECK\x00geofence\x00"

        first_features, first_evidence = extract_uav_features(image)
        second_features, second_evidence = extract_uav_features(image)

        self.assertEqual(first_features, second_features)
        self.assertEqual(first_evidence, second_evidence)
        self.assertTrue(
            all(math.isfinite(v) for v in first_features.values())
        )


if __name__ == "__main__":
    unittest.main()