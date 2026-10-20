from pathlib import Path
import base64
import sys
import unittest
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from px4_consistency import check_px4_consistency


def make_container(names):
    entries = "".join(
        f'<parameter name="{name}" type="FLOAT" default="1"/>'
        for name in names
    )
    xml = f"<parameters>{entries}</parameters>".encode("utf-8")

    return {
        "magic": "PX4FWv1",
        "parameter_xml": base64.b64encode(
            zlib.compress(xml)
        ).decode("ascii"),
        "parameter_xml_size": len(xml),
    }


class Px4ConsistencyTests(unittest.TestCase):
    def test_non_px4_is_not_applicable(self):
        result = check_px4_consistency(
            {"magic": "APJFWv1"}, b"example"
        )
        self.assertEqual(result["status"], "not_applicable")

    def test_missing_metadata_is_unavailable(self):
        result = check_px4_consistency(
            {"magic": "PX4FWv1"}, b"MC_ROLL_P"
        )
        self.assertEqual(result["status"], "unavailable")

    def test_all_four_groups_are_consistent(self):
        names = [
            "MC_ROLL_P",
            "RTL_RETURN_ALT",
            "COM_ARM_AUTH_REQ",
            "GF_ACTION",
        ]
        image = b"\x00".join(name.encode() for name in names)

        result = check_px4_consistency(
            make_container(names), image
        )

        self.assertEqual(result["status"], "consistent")
        self.assertEqual(result["checked_groups"], 4)
        self.assertEqual(result["mismatch_groups"], 0)

        for group in result["groups"].values():
            self.assertEqual(group["correspondence"], 1.0)

    def test_missing_geofence_name_is_reported(self):
        names = ["MC_ROLL_P", "GF_ACTION"]
        result = check_px4_consistency(
            make_container(names), b"MC_ROLL_P\x00"
        )

        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["mismatch_groups"], 1)

        geofence = result["groups"]["geofence"]
        self.assertEqual(geofence["missing_name_count"], 1)
        self.assertEqual(
            geofence["missing_name_examples"], ["GF_ACTION"]
        )

    def test_external_flash_names_are_recovered(self):
        names = ["MC_ROLL_P", "GF_ACTION"]
        result = check_px4_consistency(
            make_container(names),
            b"MC_ROLL_P\x00",
            b"GF_ACTION\x00",
        )
        self.assertEqual(result["status"], "consistent")

    def test_names_cannot_cross_flash_regions(self):
        result = check_px4_consistency(
            make_container(["GF_ACTION"]),
            b"GF_",
            b"ACTION",
        )
        self.assertEqual(
            result["groups"]["geofence"]["status"], "mismatch"
        )

    def test_partial_name_is_not_a_match(self):
        result = check_px4_consistency(
            make_container(["GF_ACTION"]),
            b"GF_ACTION_EXTRA\x00",
        )
        self.assertEqual(
            result["groups"]["geofence"]["recovered_names"], 0
        )

    def test_absent_group_is_unavailable_not_consistent(self):
        result = check_px4_consistency(
            make_container(["MC_ROLL_P"]),
            b"MC_ROLL_P\x00",
        )

        self.assertEqual(result["checked_groups"], 1)
        self.assertEqual(
            result["groups"]["geofence"]["status"], "unavailable"
        )
        self.assertIsNone(
            result["groups"]["geofence"]["correspondence"]
        )

    def test_wrong_xml_size_is_rejected(self):
        container = make_container(["GF_ACTION"])
        container["parameter_xml_size"] += 1

        with self.assertRaises(ValueError):
            check_px4_consistency(container, b"GF_ACTION")

    def test_results_are_repeatable(self):
        container = make_container(["GF_ACTION"])

        first = check_px4_consistency(container, b"GF_ACTION")
        second = check_px4_consistency(container, b"GF_ACTION")

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()