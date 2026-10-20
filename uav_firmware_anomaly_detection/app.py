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
from model_routing import UnsupportedVehicleError
from pdf_report import generate_pdf_report


st.set_page_config(
    page_title="UAV Firmware Check",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

GROUP_LABELS = {
    "flight_control": "Flight control",
    "navigation_rtl": "Return-to-home & navigation",
    "arming_failsafe": "Arming & failsafes",
    "geofence": "Geofence (no-fly limits)",
    "state_estimation": "Position & attitude tracking",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        color: #E8D2C4;
    }
    .stApp { background: #3E3239; }
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stCaptionContainer"], .stApp summary { color: #E8D2C4; }
    header[data-testid="stHeader"], #MainMenu, footer,
    [data-testid="stToolbar"], [data-testid="stSidebar"],
    [data-testid="collapsedControl"] { display: none; }
    .block-container { max-width: 980px; padding-top: 1.2rem;
        padding-bottom: 0.5rem; }

    .topbar { display: flex; justify-content: space-between;
        align-items: center; padding-bottom: 0.6rem; }
    .logo { font-weight: 600; font-size: 1.05rem; color: #F7EFE7; }
    .tag { font-size: 0.68rem; letter-spacing: 0.06em;
        text-transform: uppercase; color: #C7A9A3; }

    .hero { text-align: center; padding: 0.6rem 0 0.8rem; }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #F7EFE7;
        margin: 0; padding: 0; line-height: 1.2; }
    .hero h3 { font-size: 1rem; font-weight: 500; color: #E8D2C4;
        margin: 0.3rem 0 0; }

    [data-testid="stFileUploader"] section {
        background: #F7EFE7; border: 0; border-radius: 14px; }
    [data-testid="stFileUploader"] section * { color: #3E3239 !important; }
    [data-testid="stFileUploader"] button { background:#E8D2C4; border:0; }
    .stButton button[kind="primary"] {
        background: #F7EFE7; color: #3E3239; border: 0;
        border-radius: 999px; padding: 0.55rem 1.6rem;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; }
    .stButton button[kind="primary"]:hover { background: #F7EFE7; color:#3E3239; }
    .stButton button[kind="primary"]:disabled {
        background: #6B5560; color: #C7A9A3; }
    .stDownloadButton button {
        border-radius: 999px; border: 1px solid #F7EFE7;
        background: transparent; color: #F7EFE7;
        font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.05em; text-transform: uppercase; }
    .stDownloadButton button:hover { background: #F7EFE7; color: #3E3239; }
    .cards { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:0.6rem; }
    .card { background:#F7EFE7; border:0; border-radius:14px; padding:14px 18px; }
    .card h4 { margin:0 0 6px; font-size:0.95rem; color:#3E3239; font-weight:700; }
    .card ul { margin:0; padding-left:1.1rem; }
    .card li { font-size:0.82rem; line-height:1.5; margin-bottom:4px; color:#6B5560 !important; }
    .card p { margin:0; font-size:0.82rem; line-height:1.55; color:#6B5560; }
    [data-testid="stExpander"] { background:#F7EFE7; border:0; }
    [data-testid="stExpander"] * { color:#3E3239 !important; }
    hr { border-color:#6B5560 !important; }
    .stApp [data-testid="stMarkdownContainer"] .card h4 { color:#3E3239 !important; font-size:1rem; padding:0; }
    .stApp .card h4 span { color:#3E3239 !important; }
    .card [data-testid="stHeaderActionElements"] { display:none; }
    div[data-testid="stAlert"] { background:#F7EFE7 !important; border-radius:12px; }
    div[data-testid="stAlert"] * { color:#3E3239 !important; }
    .stApp .card p { color:#6B5560 !important; }
    [data-testid="stFileUploader"] section [data-testid="stFileChip"] > div:first-child, [data-testid="stFileUploader"] section svg { background:#E8D2C4 !important; }
    [data-testid="stFileUploader"] section svg { color:#3E3239 !important; fill:#3E3239 !important; }
    [data-testid="stExpander"] { border-radius:14px !important; overflow:hidden; }
    [data-testid="stFileUploaderFile"], [data-testid="stFileChip"] { background: transparent !important; }
    [data-testid="stFileUploader"] section [data-testid="stFileChip"] * { color:#3E3239 !important; }
    .stButton button[kind="primary"] p, .stButton button[kind="primary"] span { color:#3E3239 !important; font-weight:700; }
    .stButton button[kind="primary"]:disabled p { color:#C7A9A3 !important; }
    .stDownloadButton button p { color:#F7EFE7 !important; }
    .stDownloadButton button:hover p { color:#3E3239 !important; }
    .hero h1 { color:#F7EFE7 !important; }
    div[data-testid="stAlert"] { border:1px solid #6B5560; }
    hr { margin: 0.6rem 0; }
    .small { font-size: 0.68rem; color: #C7A9A3; text-align: center;
        padding-top: 0.6rem; }
    </style>
    <div class="topbar">
        <div class="logo">✈️ UAV Firmware Check</div>
        <div class="tag">Research prototype</div>
    </div>
    <div class="hero">
        <h1>UAV Anomaly Detection</h1>
        <h3>Are the flight-safety features still there? Upload drone
        firmware to find out.</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, button_column = st.columns([4, 1], vertical_alignment="center")

with upload_column:
    uploaded = st.file_uploader(
        "Firmware file",
        type=["apj", "px4"],
        help="ArduPilot (.apj) or PX4 (.px4). Maximum 64 MiB.",
        label_visibility="collapsed",
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

with button_column:
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        disabled=uploaded is None,
        width="stretch",
    )

st.caption(
    "Supports ArduCopter, ArduPlane and PX4. The file is read, never run."
)

if analyze_clicked:
    st.session_state.pop("analysis_report", None)

    if not upload_bytes or len(upload_bytes) > 64 * 1024 * 1024:
        st.error("Please upload a firmware file that is not empty and is "
                 "no larger than 64 MiB.")
    else:
        try:
            with st.spinner("Checking the firmware..."):
                suffix = Path(uploaded.name).suffix.lower()

                with tempfile.TemporaryDirectory() as directory:
                    temporary_path = Path(directory) / f"firmware{suffix}"
                    temporary_path.write_bytes(upload_bytes)

                    report = analyze_firmware(temporary_path)

                # Preserve the user's original display filename.
                report["firmware"]["file_name"] = uploaded.name
                st.session_state["analysis_report"] = report

        except UnsupportedVehicleError as error:
            st.error(str(error))

        except Exception:
            traceback.print_exc()
            st.error(
                "We couldn't analyze this file. Please check that it is a "
                "valid .apj or .px4 firmware file. Technical details are "
                "in the terminal."
            )

report = st.session_state.get("analysis_report")

if report is None and not analyze_clicked:
    st.markdown(
        '''
        <div class="cards">
          <div class="card"><h4>How it works</h4><ul>
            <li>Upload a drone firmware file (.apj or .px4).</li>
            <li>The tool reads the file but never runs it.</li>
            <li>It checks for flight-safety content: geofence, failsafes and return-to-home.</li>
            <li>A model trained on normal ArduPilot and PX4 firmware flags anything different.</li>
            <li>You get a clear result plus a JSON or PDF report.</li></ul></div>
          <div class="card"><h4>Why this is significant</h4><ul>
            <li>Firmware can be changed to remove safety features without anyone noticing.</li>
            <li>Generic IoT detectors do not understand flight controllers, so they miss this.</li>
            <li>This tool focuses only on UAV firmware.</li>
            <li>It gives investigators a fast first check that a drone's safety logic is intact.</li></ul></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

if report is not None:
    result = report["result"]
    vehicle = report["firmware"].get("vehicle", {})
    vehicle_name = vehicle.get("vehicle", "this")
    reasoning = report["explanation"].get("uav_reasoning")
    anomalous = result["verdict"] == "anomalous"

    st.divider()

    left, right = st.columns(2)

    with left:
        if anomalous:
            st.error(
                "**Unusual firmware.** Further investigation is recommended."
            )
        else:
            st.success(
                f"**Looks normal.** It matches typical {vehicle_name} "
                "firmware."
            )

        st.caption(
            f"{report['firmware']['family']} · {vehicle_name} · "
            f"unusualness score {result['anomaly_score']:.3f} "
            f"(flagged above {result['decision_threshold']:.4f})"
        )

        baseline = report.get("baseline_comparison")
        if baseline:
            if baseline["agrees_with_main_model"]:
                st.caption(
                    "Second opinion (Isolation Forest): also "
                    f"{baseline['verdict']}."
                )
            else:
                st.caption(
                    "Second opinion (Isolation Forest): "
                    f"{baseline['verdict']}, which differs from the main "
                    "result."
                )

        report_bytes = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")

        json_column, pdf_column = st.columns(2)

        with json_column:
            st.download_button(
                "JSON report",
                data=report_bytes,
                file_name=(
                    f"firmware_report_"
                    f"{report['firmware']['container_sha256'][:12]}.json"
                ),
                mime="application/json",
                width="stretch",
            )

        with pdf_column:
            try:
                pdf_bytes = generate_pdf_report(report)

                st.download_button(
                    "PDF report",
                    data=pdf_bytes,
                    file_name=(
                        f"firmware_report_"
                        f"{report['firmware']['container_sha256'][:12]}.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch",
                )

            except Exception:
                traceback.print_exc()
                st.error(
                    "The PDF could not be created. The result and JSON "
                    "report are still available."
                )

    with right:
        if reasoning:
            st.markdown("**Flight-safety checks**")
            for item in reasoning["groups"]:
                label = GROUP_LABELS.get(
                    item["group"], item["group"].replace("_", " ").title()
                )
                if item["status"] == "as_expected":
                    st.write(f"✅ {label}")
                elif item["status"] == "expected_content_missing":
                    st.write(f"❌ **{label}** – expected content not found")
                else:
                    st.write(f"⚠️ **{label}** – unusual extra content")

    with st.expander("More details"):
        metadata = report["firmware"]["metadata"]
        st.write("**File:**", report["firmware"]["file_name"])
        st.write("**Board:**", metadata.get("summary") or "not recorded")
        st.write("**Build identity:**", metadata.get("git_identity"))
        st.write("**Analyzed at (UTC):**", report["analysis_timestamp_utc"])
        st.write(
            "**Model:**", report["model"]["scope"],
            f"· {report['model']['retained_features']} features used",
        )

        st.caption("File fingerprint (SHA-256)")
        st.code(report["firmware"]["container_sha256"], language=None)

        if reasoning:
            st.caption("What was found compared with typical firmware")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Area": GROUP_LABELS.get(
                            item["group"], item["group"]),
                        "Found": round(item["observed_marker_coverage"], 2),
                        "Typical": round(item["typical_marker_coverage"], 2),
                        "Expected but not found":
                            ", ".join(item["missing_markers"]),
                    }
                    for item in reasoning["groups"]
                ]),
                hide_index=True,
                width="stretch",
            )

        st.caption("Things to keep in mind")
        for warning in report["warnings"]:
            st.write(f"- {warning}")

        deviations = report["explanation"]["supporting_feature_deviations"]
        if deviations:
            st.caption("Features that deviated most from typical firmware")
            st.dataframe(pd.DataFrame(deviations), hide_index=True)

        for group, markers in report["uav_evidence"].items():
            rows = [
                {
                    "marker": marker["marker"],
                    "total_matches": marker["count"],
                    "region": example["region"],
                    "region_offset": example["offset"],
                    "matched_text": example["matched_text"],
                }
                for marker in markers
                for example in marker["examples"]
            ]
            if rows:
                st.caption(
                    "Evidence found: "
                    + GROUP_LABELS.get(group, group.replace("_", " ").title())
                )
                st.dataframe(pd.DataFrame(rows), hide_index=True)

st.markdown(
    '<div class="small">Results show statistical differences from typical '
    "firmware. They do not prove tampering or malicious intent.</div>",
    unsafe_allow_html=True,
)
