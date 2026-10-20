import re

from audit_px4_metadata import decode_xml, elements_named


GROUPS = {
    "flight_control": ("MC_", "FW_R_", "FW_P_", "FW_Y_", "CA_"),
    "navigation_rtl": ("RTL_", "MIS_"),
    "arming_failsafe": ("COM_ARM_", "COM_FAIL_", "NAV_RCL_", "NAV_DLL_"),
    "geofence": ("GF_",),
}

TOKEN_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]{2,}(?![A-Za-z0-9_])"
)


def check_px4_consistency(container, main_image, external_image=b""):
    if container.get("magic") != "PX4FWv1":
        return {
            "status": "not_applicable",
            "reason": "This check requires a PX4 container.",
            "groups": {},
        }

    document, _ = decode_xml(container, "parameter_xml")

    if document is None:
        return {
            "status": "unavailable",
            "reason": "Nonempty parameter XML is unavailable.",
            "groups": {},
        }

    root_name = document.tag.rsplit("}", 1)[-1]
    if root_name != "parameters":
        raise ValueError("Unexpected parameter XML root element")

    declared_names = {
        element.get("name")
        for element in elements_named(document, "parameter")
        if element.get("name")
    }

    recovered_names = set()

    for image in (main_image, external_image):
        recovered_names.update(
            match.group().decode("ascii")
            for match in TOKEN_PATTERN.finditer(image)
        )

    groups = {}
    checked_groups = 0
    mismatch_groups = 0

    for group, prefixes in GROUPS.items():
        expected = {
            name for name in declared_names
            if name.startswith(prefixes)
        }
        matched = expected & recovered_names
        missing = sorted(expected - recovered_names)

        if not expected:
            status = "unavailable"
            correspondence = None
        else:
            checked_groups += 1
            status = "mismatch" if missing else "consistent"
            correspondence = len(matched) / len(expected)
            mismatch_groups += int(bool(missing))

        groups[group] = {
            "status": status,
            "declared_names": len(expected),
            "recovered_names": len(matched),
            "correspondence": correspondence,
            "missing_name_count": len(missing),
            "missing_name_examples": missing[:10],
        }

    overall_status = (
        "unavailable" if checked_groups == 0
        else "mismatch" if mismatch_groups
        else "consistent"
    )

    return {
        "status": overall_status,
        "checked_groups": checked_groups,
        "mismatch_groups": mismatch_groups,
        "groups": groups,
        "interpretation": (
            "Checks declared flight-related parameter names against names "
            "recoverable from decoded firmware. A mismatch requires review; "
            "agreement does not establish correct implementation or authenticity."
        ),
    }