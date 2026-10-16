from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


def generate_pdf_report(report):
    buffer = BytesIO()
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportCell",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        splitLongWords=True,
    ))

    def paragraph(value, style="BodyText"):
        return Paragraph(escape(str(value)), styles[style])

    def table(rows, widths):
        cells = [
            [paragraph(value, "ReportCell") for value in row]
            for row in rows
        ]

        result = Table(
            cells,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )

        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
        ]))

        return result

    firmware = report["firmware"]
    result = report["result"]
    model = report["model"]

    story = [
        paragraph("UAV Firmware Anomaly Detection", "Title"),
        paragraph("Exploratory firmware screening report"),
        Spacer(1, 0.4 * cm),
        paragraph("Analysis result", "Heading2"),
    ]

    summary_rows = [
        ["Field", "Value"],
        ["Filename", firmware["file_name"]],
        ["Family", firmware["family"]],
        ["Verdict", result["verdict"].upper()],
        ["Anomaly score", f"{result['anomaly_score']:.9g}"],
        ["Decision threshold", f"{result['decision_threshold']:.9g}"],
        ["Analysis time (UTC)", report["analysis_timestamp_utc"]],
        ["Model ID", model["model_id"]],
        ["Board ID", firmware["metadata"].get("board_id")],
        ["Build identity", firmware["metadata"].get("git_identity")],
        ["Main flash bytes", firmware["main_image_bytes"]],
        ["External flash bytes", firmware["external_image_bytes"]],
        [
            "Candidate / retained features",
            f"{model['candidate_features']} / {model['retained_features']}",
        ],
    ]

    story.append(table(summary_rows, [5 * cm, 12.4 * cm]))

    relation = (
        "exceeds"
        if result["verdict"] == "anomalous"
        else "does not exceed"
    )

    story.extend([
        Spacer(1, 0.3 * cm),
        paragraph(
            f"The anomaly score {relation} the model threshold. "
            "This determines the reported verdict. Scores are not "
            "probabilities of compromise."
        ),
        paragraph("Evidence identification", "Heading2"),
        paragraph("Original-container SHA-256:"),
        paragraph(firmware["container_sha256"], "ReportCell"),
        Spacer(1, 0.15 * cm),
        paragraph("Complete-payload SHA-256:"),
        paragraph(firmware["payload_sha256"], "ReportCell"),
        Spacer(1, 0.15 * cm),
        paragraph("Model-file SHA-256:"),
        paragraph(model["model_file_sha256"], "ReportCell"),
        paragraph("Supporting feature observations", "Heading2"),
        paragraph(report["explanation"]["method"]),
        Spacer(1, 0.2 * cm),
    ])

    deviations = report["explanation"]["supporting_feature_deviations"]

    if deviations:
        rows = [
            ["Feature", "Observed", "Training median", "Departure / SD"]
        ]

        for item in deviations:
            rows.append([
                item["feature"],
                f"{item['observed']:.6g}",
                f"{item['training_median']:.6g}",
                f"{item['standardized_deviation_from_median']:.3f}",
            ])

        story.append(table(
            rows,
            [8 * cm, 3 * cm, 3.2 * cm, 3.2 * cm],
        ))
    else:
        story.append(paragraph(
            "No individual feature exceeded the reporting cutoff. "
            "The model evaluates the combined feature pattern."
        ))

    inactive = report["explanation"]["inactive_feature_changes"]

    if inactive:
        story.append(paragraph(
            "Changed features excluded from scoring: "
            + ", ".join(inactive)
        ))

    story.extend([
        paragraph("UAV static content evidence", "Heading2"),
        paragraph(
            "Matches identify selected flight-stack content. They do not "
            "establish enabled settings, correct implementation or runtime "
            "behaviour. Offsets are relative to the named decoded flash region."
        ),
    ])

    for group, markers in report["uav_evidence"].items():
        heading = paragraph(
            group.replace("_", " ").title().replace("Rtl", "RTL"),
            "Heading3",
        )

        if not markers:
            story.append(KeepTogether([
                heading,
                paragraph("No selected markers matched."),
            ]))
            continue

        rows = [["Marker", "Count", "Example evidence"]]

        for marker in markers:
            examples = "; ".join(
                f"{example['matched_text']} "
                f"[{example['region']} offset {example['offset']}]"
                for example in marker["examples"]
            )

            rows.append([
                marker["marker"],
                marker["count"],
                examples,
            ])

        story.append(KeepTogether([
            heading,
            table(rows, [5 * cm, 1.4 * cm, 11 * cm]),
        ]))

    story.append(
        paragraph("Interpretation and limitations", "Heading2")
    )

    for warning in report["warnings"]:
        story.append(paragraph(warning))
        story.append(Spacer(1, 0.12 * cm))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)

        canvas.drawString(
            1.8 * cm,
            1 * cm,
            "UAV firmware screening - research prototype",
        )

        canvas.drawRightString(
            A4[0] - 1.8 * cm,
            1 * cm,
            f"Page {document.page}",
        )

        canvas.restoreState()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    return buffer.getvalue()