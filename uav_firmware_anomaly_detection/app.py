from pathlib import Path
import hashlib
import json
import sys
import tempfile
import traceback

import pandas as pd
import streamlit as st


COMPONENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from analyze_firmware import analyze_firmware
from pdf_report import generate_pdf_report


st.set_page_config(
    page_title="UAV Firmware Anomaly Detection",
    page_icon="✈️",
    layout="wide",
)

st.title("UAV Firmware Anomaly Detection")
st.caption("Static firmware screening and investigation support")

st.info(
    "Research prototype: results identify statistical deviations. "
    "They do not certify authenticity or prove maliciousness."
)

with st.sidebar:
    st.header("Analysis")
    st.write("Supported containers: APJ and PX4")
    st.write("Maximum file size: 64 MiB")
    st.write("Firmware is analyzed without execution.")
    st.write("Uploads use saved models; they do not retrain them.")

uploaded = st.file_uploader(
    "Choose UAV firmware",
    type=["apj", "px4"],
)

upload_bytes = uploaded.getvalue() if uploaded is not None else None

upload_identity = (
    uploaded.name + ":" + hashlib.sha256(upload_bytes).hexdigest()
    if uploaded is not None
    else None
)

# Clear old results when the uploaded file changes or is removed.
if upload_identity != st.session_state.get("current_upload"):
    st.session_state["current_upload"] = upload_identity
    st.session_state.pop("analysis_report", None)

if st.button(
    "Analyze firmware",
    type="primary",
    disabled=uploaded is None,
):
    st.session_state.pop("analysis_report", None)

    if not upload_bytes or len(upload_bytes) > 64 * 1024 * 1024:
        st.error("Upload a nonempty firmware file no larger than 64 MiB.")
    else:
        try:
            with st.spinner("Extracting features and analyzing firmware..."):
                suffix = Path(uploaded.name).suffix.lower()

                with tempfile.TemporaryDirectory() as directory:
                    temporary_path = Path(directory) / f"firmware{suffix}"
                    temporary_path.write_bytes(upload_bytes)

                    report = analyze_firmware(temporary_path)

                # Preserve the user's original display filename.
                report["firmware"]["file_name"] = uploaded.name
                st.session_state["analysis_report"] = report

        except Exception:
            traceback.print_exc()
            st.error(
                "Analysis could not be completed. Check that the file is "
                "a valid supported container and the saved models are available. "
                "Technical details are in the terminal."
            )

report = st.session_state.get("analysis_report")

if report is not None:
    result = report["result"]

    if result["verdict"] == "anomalous":
        st.warning("Anomalous: further investigation recommended.")
    else:
        st.success("Normal: within this model's learned baseline.")

    first, second, third = st.columns(3)

    first.metric("Firmware family", report["firmware"]["family"])
    second.metric("Anomaly score", f"{result['anomaly_score']:.6f}")
    third.metric(
        "Decision threshold",
        f"{result['decision_threshold']:.6f}",
    )

    st.write(
        "The firmware is flagged when its anomaly score exceeds the threshold. "
        "Scores are not probabilities and should not be compared across families."
    )

    context_tab, deviations_tab, evidence_tab = st.tabs([
        "Firmware and model",
        "Supporting deviations",
        "UAV evidence",
    ])

    with context_tab:
        st.write("Filename:", report["firmware"]["file_name"])
        st.write("Analysis time (UTC):", report["analysis_timestamp_utc"])
        st.write("Model ID:", report["model"]["model_id"])
        st.write(
            "Board ID:",
            report["firmware"]["metadata"].get("board_id"),
        )
        st.write(
            "Build identity:",
            report["firmware"]["metadata"].get("git_identity"),
        )
        st.write(
            "Candidate feature dimensions:",
            report["model"]["candidate_features"],
        )
        st.write(
            "Retained feature dimensions:",
            report["model"]["retained_features"],
        )

        st.caption("Original-container SHA-256")
        st.code(report["firmware"]["container_sha256"], language=None)

        st.caption("Complete-payload SHA-256")
        st.code(report["firmware"]["payload_sha256"], language=None)

    with deviations_tab:
        st.write(report["explanation"]["decision_basis"])
        st.caption(report["explanation"]["method"])

        deviations = report["explanation"][
            "supporting_feature_deviations"
        ]

        if deviations:
            st.dataframe(pd.DataFrame(deviations), hide_index=True)
        else:
            st.write(
                "No individual feature exceeded the reporting cutoff. "
                "The model still assesses the combined feature pattern."
            )

        if report["explanation"]["inactive_feature_changes"]:
            st.warning(
                "Some excluded training-constant features changed. "
                "See the complete report for details."
            )

    with evidence_tab:
        st.caption(
            "Markers provide static content evidence. They do not establish "
            "effective parameter settings or runtime behaviour."
        )

        for group, markers in report["uav_evidence"].items():
            heading = group.replace("_", " ").title().replace("Rtl", "RTL")

            with st.expander(heading):
                rows = []

                for marker in markers:
                    for example in marker["examples"]:
                        rows.append({
                            "marker": marker["marker"],
                            "total_matches": marker["count"],
                            "region": example["region"],
                            "region_offset": example["offset"],
                            "matched_text": example["matched_text"],
                        })

                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
                else:
                    st.write("No selected markers matched.")

    with st.expander("Interpretation and limitations"):
        for warning in report["warnings"]:
            st.write(f"- {warning}")

    report_bytes = json.dumps(
        report,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")

    st.download_button(
        "Download JSON report",
        data=report_bytes,
        file_name=(
            f"firmware_report_"
            f"{report['firmware']['container_sha256'][:12]}.json"
        ),
        mime="application/json",
    )

    try:
        pdf_bytes = generate_pdf_report(report)

        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=(
                f"firmware_report_"
                f"{report['firmware']['container_sha256'][:12]}.pdf"
            ),
            mime="application/pdf",
        )

    except Exception:
        traceback.print_exc()
        st.error(
            "PDF generation failed. The analysis result and JSON report "
            "remain available. Technical details are in the terminal."
        )