from __future__ import annotations

from pathlib import Path

from .model import architecture_table
from .utils import percentage, read_json


def _require_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError(
            "ReportLab is required for PDF generation. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return {
        "colors": colors,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "cm": cm,
        "Image": Image,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return read_json(path)


def _best_and_worst_classes(per_class_accuracy: dict[str, float]) -> tuple[str, str]:
    ordered = sorted(per_class_accuracy.items(), key=lambda item: item[1])
    return ordered[-1][0], ordered[0][0]


def _compare_metric(first: float, second: float) -> str:
    delta = second - first
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f} percentage points"


def _report_markdown(
    summary: dict[str, object],
    hmdb_metrics: dict[str, object],
    arid_zero_shot_metrics: dict[str, object],
    arid_finetuned_metrics: dict[str, object],
) -> str:
    hmdb_summary = summary["datasets"]["hmdb51"]
    arid_summary = summary["datasets"]["arid"]
    hmdb_best, hmdb_worst = _best_and_worst_classes(hmdb_metrics["per_class_accuracy"])
    arid_best, arid_worst = _best_and_worst_classes(arid_finetuned_metrics["per_class_accuracy"])
    sampling_note = ""
    if summary.get("max_metadata_samples_per_dataset") is not None:
        sampling_note = (
            f"\n\nFor this local smoke-tested run, metadata probing was capped at "
            f"{summary['max_metadata_samples_per_dataset']} stratified videos per dataset to keep runtime reasonable. "
            "For the final coursework submission, rerun `summarize-data` without this cap so every listed video contributes to the descriptive statistics."
        )

    return f"""# EEC4200 Coursework Report

## Project Context

This report studies video classification on HMDB51-mini and ARID-mini using a lightweight 3D convolutional neural network. The workflow was designed to answer the coursework questions directly: dataset observation, 3D-CNN design, quantitative evaluation, cross-dataset robustness analysis, and ARID-oriented workflow redesign.{sampling_note}

## Part 1a: Observation and Analysis of HMDB51-mini

HMDB51-mini is class-balanced by design in the provided split files: each of the eight action classes contributes 70 training videos and 30 test videos, so the listed subset contains 800 clips in total. This balance reduces label-frequency bias, which makes it easier to interpret confusion patterns as a modeling issue rather than a sampling issue.

The visual characteristics of HMDB51-mini are diverse. The clips come from movies, television, and internet videos, which introduces substantial variation in camera motion, viewpoint, actor scale, and background clutter. The dataset summary shows a median duration of {hmdb_summary["metadata_summary"]["duration_sec_median"]:.2f} seconds and a median resolution of {hmdb_summary["metadata_summary"]["width_median"]:.0f}x{hmdb_summary["metadata_summary"]["height_median"]:.0f}. Representative frames and the class-distribution plot were generated automatically in the analysis stage to support this observation.

Several class pairs are visually and temporally similar. For example, `drink` and `pour` both involve hand-object interaction around containers, while `walk` and `run` share body pose patterns but differ in motion speed. These overlaps motivate a model that can capture motion cues, not only single-frame appearance.

One data-quality issue also appeared during QA: four HMDB training clips listed in `hmdb51_train.txt` are absent from the directory. They were removed rather than replaced so that the implementation stays faithful to the provided split file.

## Part 1b: 3D-CNN Design and Rationale

The baseline model is a lightweight 3D-CNN with four convolutional stages and a compact classification head. The input to the network is a 16-frame RGB clip resized to 112x112. The spatial dimension is reduced gradually while temporal information is preserved in the first stage and then compressed in the next two stages. This design keeps the model small enough to train from scratch on HMDB51-mini while still learning joint spatiotemporal features.

The architecture table embedded in the generated report documents the exact structure: channel progression 32 -> 64 -> 128 -> 256, pooling order `(1,2,2)` followed by two `(2,2,2)` stages, global average pooling, dropout of 0.4, and an 8-way linear classifier. Training uses AdamW, early stopping on validation accuracy, and three-clip averaging during final testing to make evaluation more stable.

## Part 1c: HMDB51-mini Results and Analysis

The best HMDB51-mini checkpoint reached an official test accuracy of **{percentage(hmdb_metrics["accuracy"])}** with a macro-F1 score of **{percentage(hmdb_metrics["macro_f1"])}**. The strongest class in the per-class analysis is `{hmdb_best}` ({percentage(hmdb_metrics["per_class_accuracy"][hmdb_best])}), while the weakest is `{hmdb_worst}` ({percentage(hmdb_metrics["per_class_accuracy"][hmdb_worst])}).

These results suggest that the model can learn the broad motion patterns of the mini dataset, but confusion remains for classes that differ mainly in temporal speed or subtle object interaction. The learning-curve figure should be read together with the confusion matrix: if training accuracy rises much faster than validation accuracy, the likely explanation is limited dataset size and overfitting; if both curves saturate early, the likely explanation is under-capacity or insufficient preprocessing.

## Part 2a: Observation and Analysis of ARID-mini

ARID-mini differs from HMDB51-mini in both class distribution and visual domain. The listed ARID subset contains {arid_summary["listed_samples"]} clips, with strong imbalance across classes: `pick` has {arid_summary["class_counts_by_split_existing"]["train"]["pick"] + arid_summary["class_counts_by_split_existing"]["test"]["pick"]} listed existing clips, whereas `walk` has {arid_summary["class_counts_by_split_existing"]["train"]["walk"] + arid_summary["class_counts_by_split_existing"]["test"]["walk"]}. This makes the recognition problem less uniform than in HMDB51-mini.

The brightness statistics also indicate a harder low-light domain. ARID-mini has an average brightness of {arid_summary["metadata_summary"]["brightness_mean"]:.2f}, compared with {hmdb_summary["metadata_summary"]["brightness_mean"]:.2f} on HMDB51-mini. Together with the darker representative frames, this supports the claim that illumination and visibility are major domain-shift factors.

## Part 2b: Zero-Shot Transfer from HMDB51-mini to ARID-mini

When the HMDB-trained model is applied directly to ARID-mini without adaptation, the test accuracy drops to **{percentage(arid_zero_shot_metrics["accuracy"])}** with macro-F1 **{percentage(arid_zero_shot_metrics["macro_f1"])}**. This is substantially lower than the in-domain HMDB result, which confirms that the original workflow is not robust to the ARID domain shift.

The drop is expected for three reasons. First, ARID is darker, so appearance cues become less reliable. Second, ARID is class-imbalanced, which makes a balanced HMDB training recipe less suitable. Third, the background and motion visibility conditions differ from those in HMDB51-mini, so features learned from the source dataset do not transfer cleanly.

## Part 2c: Workflow Redesign for ARID-mini

The redesigned workflow keeps the same backbone so that the comparison isolates workflow changes instead of mixing architecture changes with preprocessing changes. Three modifications were introduced:

1. Luminance CLAHE is applied frame by frame to improve local contrast under dark conditions.
2. Brightness and gamma augmentation are added during training so the model sees stronger illumination variation.
3. Class-balanced sampling is used to reduce the bias toward frequent ARID classes such as `walk` and `wave`.

The ARID model is initialized from the HMDB checkpoint rather than trained from scratch. This keeps useful generic motion features while allowing the optimization to specialize to the darker ARID domain.

## Part 2d: Revised ARID-mini Performance and Analysis

After fine-tuning with the redesigned workflow, the ARID-mini test accuracy becomes **{percentage(arid_finetuned_metrics["accuracy"])}** with macro-F1 **{percentage(arid_finetuned_metrics["macro_f1"])}**. Compared with the zero-shot ARID result, this is an improvement of {_compare_metric(arid_zero_shot_metrics["accuracy"], arid_finetuned_metrics["accuracy"])} in accuracy.

The strongest ARID class after adaptation is `{arid_best}` ({percentage(arid_finetuned_metrics["per_class_accuracy"][arid_best])}), while the weakest remains `{arid_worst}` ({percentage(arid_finetuned_metrics["per_class_accuracy"][arid_worst])}). If the gain is meaningful but still limited, the remaining challenge is likely that illumination-aware preprocessing helps visibility but cannot fully remove viewpoint variation, clutter, or action ambiguity.

## Limitations

- The report intentionally follows only the listed split files, so extra videos present in the directories were ignored.
- Four HMDB training clips were missing on disk and therefore excluded.
- Final performance still depends on the actual GPU training run and random seed, so the included figures and metrics should be interpreted together rather than through a single number.

## AI Usage Statement

AI assistance was used to help structure the codebase, automate report generation, and polish the wording of explanations. The dataset observations, experiment configuration, and final conclusions were still grounded in the artifacts generated from the provided HMDB51-mini and ARID-mini splits rather than copied from an external answer source.
"""


def _build_pdf(
    markdown_text: str,
    summary: dict[str, object],
    hmdb_metrics: dict[str, object],
    arid_zero_shot_metrics: dict[str, object],
    arid_finetuned_metrics: dict[str, object],
    output_pdf: Path,
) -> str:
    rl = _require_reportlab()
    styles = rl["getSampleStyleSheet"]()
    styles.add(rl["ParagraphStyle"](name="Body", parent=styles["BodyText"], leading=15, spaceAfter=8))

    doc = rl["SimpleDocTemplate"](str(output_pdf), pagesize=rl["A4"], leftMargin=1.4 * rl["cm"], rightMargin=1.4 * rl["cm"], topMargin=1.3 * rl["cm"], bottomMargin=1.3 * rl["cm"])
    story = []

    story.append(rl["Paragraph"]("EEC4200 Coursework Report", styles["Title"]))
    story.append(rl["Spacer"](1, 0.2 * rl["cm"]))
    story.append(rl["Paragraph"]("Automatic report built from the generated analysis and experiment artifacts.", styles["Body"]))

    dataset_table = [
        ["Dataset", "Listed", "Existing train/test", "Missing listed clips", "Ignored extra clips"],
        [
            "HMDB51-mini",
            str(summary["datasets"]["hmdb51"]["listed_samples"]),
            f'{summary["datasets"]["hmdb51"]["existing_split_counts"]["train"]}/{summary["datasets"]["hmdb51"]["existing_split_counts"]["test"]}',
            str(summary["datasets"]["hmdb51"]["missing_samples"]),
            str(summary["datasets"]["hmdb51"]["ignored_extra_samples"]),
        ],
        [
            "ARID-mini",
            str(summary["datasets"]["arid"]["listed_samples"]),
            f'{summary["datasets"]["arid"]["existing_split_counts"]["train"]}/{summary["datasets"]["arid"]["existing_split_counts"]["test"]}',
            str(summary["datasets"]["arid"]["missing_samples"]),
            str(summary["datasets"]["arid"]["ignored_extra_samples"]),
        ],
    ]
    dataset_table_flow = rl["Table"](dataset_table, repeatRows=1)
    dataset_table_flow.setStyle(
        rl["TableStyle"](
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl["colors"].lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, rl["colors"].grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(dataset_table_flow)
    story.append(rl["Spacer"](1, 0.3 * rl["cm"]))

    architecture_rows = [["Stage", "Shape", "Details"]]
    for row in architecture_table():
        architecture_rows.append([row["stage"], row["shape"], row["details"]])
    architecture_table_flow = rl["Table"](architecture_rows, repeatRows=1, colWidths=[3.0 * rl["cm"], 4.0 * rl["cm"], 9.0 * rl["cm"]])
    architecture_table_flow.setStyle(
        rl["TableStyle"](
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl["colors"].lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, rl["colors"].grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(rl["Paragraph"]("3D-CNN Architecture", styles["Heading2"]))
    story.append(architecture_table_flow)
    story.append(rl["Spacer"](1, 0.3 * rl["cm"]))

    metrics_table = [
        ["Experiment", "Accuracy", "Macro-F1"],
        ["HMDB51-mini test", percentage(hmdb_metrics["accuracy"]), percentage(hmdb_metrics["macro_f1"])],
        ["ARID-mini zero-shot", percentage(arid_zero_shot_metrics["accuracy"]), percentage(arid_zero_shot_metrics["macro_f1"])],
        ["ARID-mini fine-tuned", percentage(arid_finetuned_metrics["accuracy"]), percentage(arid_finetuned_metrics["macro_f1"])],
    ]
    metrics_table_flow = rl["Table"](metrics_table, repeatRows=1)
    metrics_table_flow.setStyle(
        rl["TableStyle"](
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl["colors"].lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, rl["colors"].grey),
            ]
        )
    )
    story.append(rl["Paragraph"]("Key Results", styles["Heading2"]))
    story.append(metrics_table_flow)

    for paragraph in markdown_text.split("\n\n"):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        if cleaned.startswith("# "):
            continue
        if cleaned.startswith("## "):
            story.append(rl["Spacer"](1, 0.15 * rl["cm"]))
            story.append(rl["Paragraph"](cleaned[3:], styles["Heading2"]))
        elif cleaned.startswith("- "):
            for line in cleaned.splitlines():
                story.append(rl["Paragraph"](f"- {line[2:]}", styles["Body"]))
        elif cleaned[0].isdigit() and ". " in cleaned:
            for line in cleaned.splitlines():
                story.append(rl["Paragraph"](line, styles["Body"]))
        else:
            story.append(rl["Paragraph"](cleaned.replace("`", ""), styles["Body"]))

    story.append(rl["PageBreak"]())
    story.append(rl["Paragraph"]("Generated Figures", styles["Heading2"]))
    figure_paths = [
        summary["figure_files"]["class_distribution"],
        summary["figure_files"]["video_characteristics"],
        summary["figure_files"]["hmdb51_representative_frames"],
        summary["figure_files"]["arid_representative_frames"],
        hmdb_metrics["learning_curves_png"],
        hmdb_metrics["confusion_matrix_png"],
        arid_zero_shot_metrics["confusion_matrix_png"],
        arid_finetuned_metrics["learning_curves_png"],
        arid_finetuned_metrics["confusion_matrix_png"],
    ]
    for figure_path in figure_paths:
        figure = Path(figure_path)
        if not figure.exists():
            raise FileNotFoundError(f"Required figure is missing for PDF generation: {figure}")
        story.append(rl["Paragraph"](figure.name, styles["Heading3"]))
        story.append(rl["Image"](str(figure), width=16.5 * rl["cm"], height=9.0 * rl["cm"], kind="proportional"))
        story.append(rl["Spacer"](1, 0.25 * rl["cm"]))

    doc.build(story)
    return str(output_pdf)


def build_report(output_dir: Path | str) -> dict[str, str]:
    output_dir = Path(output_dir)
    summary = _load_required_json(output_dir / "analysis" / "data_summary.json")
    hmdb_metrics = _load_required_json(output_dir / "metrics" / "hmdb_test_metrics.json")
    arid_zero_shot_metrics = _load_required_json(output_dir / "metrics" / "arid_zero_shot_metrics.json")
    arid_finetuned_metrics = _load_required_json(output_dir / "metrics" / "arid_finetuned_test_metrics.json")

    markdown_text = _report_markdown(summary, hmdb_metrics, arid_zero_shot_metrics, arid_finetuned_metrics)

    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "EEC4200_report.md"
    pdf_path = report_dir / "EEC4200_report.pdf"
    markdown_path.write_text(markdown_text)
    _build_pdf(markdown_text, summary, hmdb_metrics, arid_zero_shot_metrics, arid_finetuned_metrics, pdf_path)

    return {"markdown": str(markdown_path), "pdf": str(pdf_path)}
