"""Plain-language reasons for a result, based on UAV flight-stack content.

Compares which flight-stack marker types an uploaded firmware contains with
what is typical (training median) for firmware of the same model scope.
"""
from uav_features import GROUP_PATTERNS


def explain_uav_markers(training_medians, features):
    groups = []

    for group, markers in GROUP_PATTERNS.items():
        missing, unexpected = [], []

        for marker in markers:
            key = f"uav_{group}_{marker}_present"
            if key not in training_medians or key not in features:
                continue

            typical = training_medians[key] >= 0.5
            present = features[key] >= 0.5

            if typical and not present:
                missing.append(marker)
            elif present and not typical:
                unexpected.append(marker)

        coverage_key = f"uav_{group}_coverage"
        status = (
            "expected_content_missing" if missing
            else "unexpected_extra_content" if unexpected
            else "as_expected"
        )

        groups.append({
            "group": group,
            "observed_marker_coverage": float(features[coverage_key]),
            "typical_marker_coverage": float(training_medians[coverage_key]),
            "missing_markers": missing,
            "unexpected_markers": unexpected,
            "status": status,
        })

    flagged = [g for g in groups if g["status"] != "as_expected"]

    if not flagged:
        summary = (
            "All flight-stack content groups match what is typical for "
            "this type of firmware."
        )
    else:
        parts = []
        for item in flagged:
            name = item["group"].replace("_", " ")
            if item["missing_markers"]:
                parts.append(
                    f"{name}: expected content not found "
                    f"({', '.join(item['missing_markers'])})"
                )
            else:
                parts.append(
                    f"{name}: unusual extra content "
                    f"({', '.join(item['unexpected_markers'])})"
                )
        summary = "Differences from typical firmware - " + "; ".join(parts) + "."

    return {"summary": summary, "groups": groups}
