from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from controlled_mutations import generate_mutations
from firmware_loader import calculate_payload_hash


class ControlledMutationTests(unittest.TestCase):
    def test_reproducible_and_original_preserved(self):
        main = bytes(range(256)) * 100
        external = b"external firmware"
        original_hash = calculate_payload_hash(main, external)

        first = generate_mutations(main, external)
        second = generate_mutations(main, external)

        self.assertEqual(first, second)
        self.assertEqual(
            calculate_payload_hash(main, external), original_hash
        )

    def test_single_byte_flip_changes_exactly_one_byte(self):
        original = bytes(range(256)) * 100
        mutation = next(
            item for item in generate_mutations(original)
            if item["mutation_type"] == "single_byte_flip"
        )

        self.assertEqual(mutation["changed_bytes"], 1)
        self.assertEqual(len(mutation["main_image"]), len(original))
        self.assertEqual(
            sum(
                a != b
                for a, b in zip(original, mutation["main_image"])
            ),
            1,
        )

    def test_append_preserves_original_prefix(self):
        original = bytes(range(256)) * 100
        mutation = next(
            item for item in generate_mutations(original)
            if item["mutation_type"] == "append_random_1pct"
        )

        self.assertTrue(mutation["main_image"].startswith(original))
        self.assertEqual(len(mutation["main_image"]), 25856)
        self.assertEqual(mutation["offset"], len(original))

    def test_external_region_modified_when_larger(self):
        main = b"small"
        external = bytes(range(256)) * 10

        for mutation in generate_mutations(main, external):
            self.assertEqual(mutation["region"], "external")
            self.assertEqual(mutation["main_image"], main)
            self.assertNotEqual(mutation["external_image"], external)

    def test_no_change_zero_mutation_is_skipped(self):
        mutations = generate_mutations(b"\x00" * 1000)
        kinds = {item["mutation_type"] for item in mutations}

        self.assertNotIn("zero_region_1pct", kinds)
        self.assertEqual(len(mutations), 3)

    def test_reported_hashes_match_payloads(self):
        original = bytes(range(256)) * 10
        parent_hash = calculate_payload_hash(original, b"")

        for mutation in generate_mutations(original):
            actual_hash = calculate_payload_hash(
                mutation["main_image"],
                mutation["external_image"],
            )

            self.assertEqual(
                mutation["parent_payload_sha256"], parent_hash
            )
            self.assertEqual(
                mutation["modified_payload_sha256"], actual_hash
            )
            self.assertNotEqual(actual_hash, parent_hash)

    def test_one_byte_external_only_payload(self):
        mutations = generate_mutations(b"", b"A")

        self.assertTrue(mutations)
        for mutation in mutations:
            self.assertEqual(mutation["region"], "external")
            self.assertEqual(mutation["main_image"], b"")
            self.assertGreater(mutation["changed_bytes"], 0)

    def test_empty_payload_rejected(self):
        with self.assertRaises(ValueError):
            generate_mutations(b"", b"")


if __name__ == "__main__":
    unittest.main()