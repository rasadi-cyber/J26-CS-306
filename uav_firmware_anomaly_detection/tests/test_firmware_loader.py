from pathlib import Path
import base64
import json
import sys
import tempfile
import unittest
import zlib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from firmware_loader import load_firmware


def encode_image(image):
    return base64.b64encode(zlib.compress(image)).decode("ascii")


def make_container(main_image, external_image=b""):
    return {
        "magic": "APJFWv1",
        "board_id": 123,
        "image_size": len(main_image),
        "image": encode_image(main_image),
        "extf_image_size": len(external_image),
        "extf_image": encode_image(external_image),
    }


class FirmwareLoaderTests(unittest.TestCase):
    def load_container(self, container):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.apj"
            path.write_text(json.dumps(container), encoding="utf-8")
            return load_firmware(path)

    def test_main_image_decodes_correctly(self):
        image = bytes(range(256))
        result = self.load_container(make_container(image))
        self.assertEqual(result["main_image"], image)
        self.assertEqual(result["external_image"], b"")

    def test_external_only_firmware_is_supported(self):
        image = b"external flash firmware"
        result = self.load_container(make_container(b"", image))
        self.assertEqual(result["main_image"], b"")
        self.assertEqual(result["external_image"], image)

    def test_metadata_changes_do_not_change_payload_hash(self):
        first = make_container(b"same firmware")
        second = dict(first)
        second["description"] = "Different container metadata"

        result_a = self.load_container(first)
        result_b = self.load_container(second)

        self.assertEqual(
            result_a["payload_sha256"], result_b["payload_sha256"]
        )
        self.assertNotEqual(
            result_a["container_sha256"], result_b["container_sha256"]
        )

    def test_flash_region_boundaries_affect_payload_hash(self):
        first = self.load_container(make_container(b"AB", b"C"))
        second = self.load_container(make_container(b"A", b"BC"))

        self.assertNotEqual(
            first["payload_sha256"], second["payload_sha256"]
        )

    def test_wrong_declared_size_is_rejected(self):
        container = make_container(b"firmware")
        container["image_size"] += 1

        with self.assertRaises(ValueError):
            self.load_container(container)

    def test_wrong_magic_is_rejected(self):
        container = make_container(b"firmware")
        container["magic"] = "UNKNOWN"

        with self.assertRaises(ValueError):
            self.load_container(container)

    def test_empty_firmware_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_container(make_container(b""))


if __name__ == "__main__":
    unittest.main()