import re


# Candidate flight-stack indicators.
# Parameter prefixes require a meaningful uppercase suffix.
GROUP_PATTERNS = {
    "flight_control": {
        "attitude_parameters": r"\bATC_[A-Z][A-Z0-9_]*\b",
        "position_parameters": r"\bPSC_[A-Z][A-Z0-9_]*\b",
        "motor_parameters": r"\bMOT_[A-Z][A-Z0-9_]*\b",
        "multicopter_parameters": r"\bMC_(?:ROLL|PITCH|YAW)[A-Z0-9_]*\b",
        "control_allocator": r"\bcontrol_allocator\b",
    },
    "navigation_rtl": {
        "rtl_parameters": r"\bRTL_[A-Z][A-Z0-9_]*\b",
        "waypoint_parameters": r"\bWPNAV_[A-Z][A-Z0-9_]*\b",
        "smart_rtl_parameters": r"\bSRTL_[A-Z][A-Z0-9_]*\b",
        "mission_parameters": r"\bMIS_[A-Z][A-Z0-9_]*\b",
        "navigator_module": r"\bnavigator\b",
    },
    "arming_failsafe": {
        "arming_parameters": r"\bARMING_[A-Z][A-Z0-9_]*\b",
        "prearm_messages": r"\bPreArm\b",
        "flight_failsafe_parameters": r"\bFS_(?:THR|GCS|EKF)[A-Z0-9_]*\b",
        "commander_safety_parameters": r"\bCOM_(?:ARM|FAIL)[A-Z0-9_]*\b",
        "arming_denied_messages": r"(?i:\barming denied\b)",
    },
    "geofence": {
        "fence_parameters": r"\bFENCE_[A-Z][A-Z0-9_]*\b",
        "px4_fence_parameters": r"\bGF_[A-Z][A-Z0-9_]*\b",
        "geofence_references": r"(?i:\bgeofence\b)",
        "fence_breach_messages": r"(?i:\bfence breach\b)",
        "fence_recovery_messages": r"(?i:\bfence recovery\b)",
    },
    "state_estimation": {
        "ek3_parameters": r"\bEK3_[A-Z][A-Z0-9_]*\b",
        "ekf2_parameters": r"\bEKF2_[A-Z][A-Z0-9_]*\b",
        "ahrs_parameters": r"\bAHRS_[A-Z][A-Z0-9_]*\b",
        "ekf2_module": r"\bekf2\b",
        "ekf_diagnostics": r"\bEKF (?:compass|position|velocity|height|attitude)\b",
    },
}


def extract_uav_features(main_image, external_image=b""):
    total_bytes = len(main_image) + len(external_image)

    if not total_bytes:
        raise ValueError("Cannot analyze an empty firmware payload")

    size_mib = total_bytes / (1024 * 1024)
    strings = []

    for region, image in (
        ("main", main_image),
        ("external", external_image),
    ):
        for match in re.finditer(rb"[\x20-\x7e]{4,}", image):
            strings.append(
                (region, match.start(), match.group().decode("ascii"))
            )

    features = {}
    evidence = {}

    for group, patterns in GROUP_PATTERNS.items():
        group_evidence = []
        present_markers = 0
        total_matches = 0

        for marker, pattern in patterns.items():
            expression = re.compile(pattern)
            count = 0
            examples = []

            for region, string_offset, text in strings:
                for match in expression.finditer(text):
                    count += 1

                    if len(examples) < 3:
                        examples.append({
                            "region": region,
                            "offset": string_offset + match.start(),
                            "matched_text": match.group(),
                        })

            prefix = f"uav_{group}_{marker}"
            features[f"{prefix}_present"] = int(count > 0)
            features[f"{prefix}_count_per_mib"] = count / size_mib

            present_markers += int(count > 0)
            total_matches += count

            if count:
                group_evidence.append({
                    "marker": marker,
                    "count": count,
                    "examples": examples,
                })

        features[f"uav_{group}_coverage"] = (
            present_markers / len(patterns)
        )
        features[f"uav_{group}_matches_per_mib"] = (
            total_matches / size_mib
        )
        evidence[group] = group_evidence

    return features, evidence