from pathlib import Path
import re

from firmware_loader import load_firmware


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Candidate search terms, not validated detection rules.
CANDIDATE_GROUPS = {
    "flight_control": (
        "ATC_", "PSC_", "MOT_",
        "MC_ROLL", "MC_PITCH", "MC_YAW", "control_allocator",
    ),
    "navigation_return_to_launch": (
        "RTL_", "WPNAV_", "SRTL_", "MIS_", "navigator",
    ),
    "arming_failsafe": (
        "ARMING", "PreArm", "FS_THR", "FS_GCS",
        "COM_ARM", "COM_FAIL", "failsafe",
    ),
    "geofence": (
        "FENCE", "GF_", "geofence",
    ),
    "state_estimation": (
        "EKF", "AHRS", "EK3_", "estimator",
    ),
}


def inspect_sample(path):
    firmware = load_firmware(path)

    print(f"\nFILE: {path.relative_to(PROJECT_ROOT)}")
    print("Build identity:", firmware["metadata"]["git_identity"])

    strings = []

    for region, image in (
        ("main", firmware["main_image"]),
        ("external", firmware["external_image"]),
    ):
        for match in re.finditer(rb"[\x20-\x7e]{4,}", image):
            strings.append({
                "region": region,
                "offset": match.start(),
                "text": match.group().decode("ascii"),
            })

    for group, terms in CANDIDATE_GROUPS.items():
        matches = [
            item
            for item in strings
            if any(
                term.lower() in item["text"].lower()
                for term in terms
            )
        ]

        print(f"\n{group}: {len(matches)} matching strings")

        for item in matches[:5]:
            print(
                f"  [{item['region']} offset {item['offset']}] "
                f"{item['text'][:160]}"
            )


def main():
    for folder, extension in (
        ("ardupilot", "*.apj"),
        ("px4", "*.px4"),
    ):
        directory = PROJECT_ROOT / "data" / "raw" / folder
        samples = sorted(directory.rglob(extension))

        if not samples:
            print(f"No samples found for {folder}")
            continue

        inspect_sample(samples[0])


if __name__ == "__main__":
    main()
