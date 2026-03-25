from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .constants import CANONICAL_CLASSES
from .data import compute_split_overlap, inventory_summary_record, limit_samples_stratified, load_all_inventories
from .utils import ensure_dir, safe_mean, safe_median, write_json, write_records_csv
from .video import can_decode_video, estimate_video_brightness, extract_reference_frame, probe_video


def _require_plotting():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "Plotting dependencies are missing. Install them with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return plt, pd, sns


def _write_manifests(inventories, output_dir: Path) -> dict[str, str]:
    manifest_dir = ensure_dir(output_dir / "manifests")
    generated: dict[str, str] = {}

    for dataset_name, inventory in inventories.items():
        for split_name in ("train", "test"):
            listed_path = manifest_dir / f"{dataset_name}_{split_name}_listed.csv"
            existing_path = manifest_dir / f"{dataset_name}_{split_name}_existing.csv"
            write_records_csv(listed_path, [sample.to_record() for sample in inventory.samples(split_name)])
            write_records_csv(existing_path, [sample.to_record() for sample in inventory.existing_samples(split_name)])
            generated[f"{dataset_name}_{split_name}_listed"] = str(listed_path)
            generated[f"{dataset_name}_{split_name}_existing"] = str(existing_path)

        missing_path = manifest_dir / f"{dataset_name}_missing.csv"
        ignored_path = manifest_dir / f"{dataset_name}_ignored_extra.csv"
        write_records_csv(missing_path, [{"rel_path": rel_path} for rel_path in inventory.missing_paths])
        write_records_csv(ignored_path, [{"rel_path": rel_path} for rel_path in inventory.ignored_paths])
        generated[f"{dataset_name}_missing"] = str(missing_path)
        generated[f"{dataset_name}_ignored_extra"] = str(ignored_path)

    return generated


def _collect_metadata(
    inventories,
    output_dir: Path,
    max_metadata_samples_per_dataset: int | None = None,
) -> tuple[list[dict[str, object]], str, list[dict[str, object]]]:
    analysis_dir = ensure_dir(output_dir / "analysis")
    records: list[dict[str, object]] = []
    decode_failures: list[dict[str, object]] = []

    for dataset_name, inventory in inventories.items():
        candidate_samples = inventory.existing_samples("train") + inventory.existing_samples("test")
        selected_samples = limit_samples_stratified(
            candidate_samples,
            max_metadata_samples_per_dataset,
            seed=42,
        )
        for sample in selected_samples:
            split_name = sample.split
            try:
                if not can_decode_video(sample.abs_path):
                    raise RuntimeError(f"Failed to decode video: {sample.abs_path}")
                probe = probe_video(sample.abs_path)
                brightness = estimate_video_brightness(sample.abs_path)
                records.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "sample_id": sample.sample_id,
                        "label": sample.label,
                        "class_name": sample.class_name,
                        "canonical_class": sample.canonical_class,
                        "rel_path": sample.rel_path,
                        "width": probe["width"],
                        "height": probe["height"],
                        "fps": probe["fps"],
                        "frame_count": probe["frame_count"],
                        "duration_sec": probe["duration_sec"],
                        "brightness": brightness,
                    }
                )
            except Exception as exc:
                decode_failures.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "sample_id": sample.sample_id,
                        "rel_path": sample.rel_path,
                        "reason": str(exc),
                    }
                )

    metadata_path = analysis_dir / "video_metadata.csv"
    write_records_csv(metadata_path, records)
    failures_path = analysis_dir / "decode_failures.csv"
    write_records_csv(failures_path, decode_failures)
    return records, str(metadata_path), decode_failures


def _build_dataset_summary(
    inventory,
    metadata_records: list[dict[str, object]],
    decode_failures: list[dict[str, object]],
) -> dict[str, object]:
    dataset_records = [record for record in metadata_records if record["dataset"] == inventory.dataset]
    dataset_failures = [record for record in decode_failures if record["dataset"] == inventory.dataset]
    existing_train = inventory.existing_samples("train")
    existing_test = inventory.existing_samples("test")

    duration_values = [float(record["duration_sec"]) for record in dataset_records]
    brightness_values = [float(record["brightness"]) for record in dataset_records]
    widths = [int(record["width"]) for record in dataset_records]
    heights = [int(record["height"]) for record in dataset_records]

    return {
        **inventory_summary_record(inventory),
        "split_counts": {split: len(inventory.samples(split)) for split in ("train", "test")},
        "existing_split_counts": {
            "train": len(existing_train),
            "test": len(existing_test),
        },
        "class_counts_by_split": {
            split: inventory.class_counts(split, existing_only=False) for split in ("train", "test")
        },
        "class_counts_by_split_existing": {
            split: inventory.class_counts(split, existing_only=True) for split in ("train", "test")
        },
        "missing_paths": inventory.missing_paths,
        "ignored_extra_sample_examples": inventory.ignored_paths[:20],
        "train_test_overlap_paths": compute_split_overlap(inventory),
        "label_mismatches": inventory.label_mismatches,
        "decode_failures": dataset_failures,
        "decode_failure_count": len(dataset_failures),
        "metadata_summary": {
            "count": len(dataset_records),
            "duration_sec_mean": safe_mean(duration_values),
            "duration_sec_median": safe_median(duration_values),
            "brightness_mean": safe_mean(brightness_values),
            "brightness_median": safe_median(brightness_values),
            "width_median": safe_median(widths),
            "height_median": safe_median(heights),
        },
    }


def _plot_class_distribution(inventories, output_dir: Path) -> str:
    plt, pd, sns = _require_plotting()
    sns.set_theme(style="whitegrid")

    rows = []
    for dataset_name, inventory in inventories.items():
        for split_name in ("train", "test"):
            counts = inventory.class_counts(split_name, existing_only=True)
            for class_name, value in counts.items():
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "class_name": class_name,
                        "count": value,
                    }
                )

    df = pd.DataFrame(rows)
    figure_dir = ensure_dir(output_dir / "figures")
    figure_path = figure_dir / "class_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for axis, dataset_name in zip(axes, ("hmdb51", "arid")):
        subset = df[df["dataset"] == dataset_name]
        sns.barplot(data=subset, x="class_name", y="count", hue="split", ax=axis)
        axis.set_title(f"{inventories[dataset_name].display_name} class distribution")
        axis.set_xlabel("Class")
        axis.set_ylabel("Number of listed samples")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path)


def _plot_video_characteristics(metadata_records: list[dict[str, object]], output_dir: Path) -> str:
    plt, pd, sns = _require_plotting()
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(metadata_records)
    figure_dir = ensure_dir(output_dir / "figures")
    figure_path = figure_dir / "video_characteristics.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="dataset", y="duration_sec", ax=axes[0])
    axes[0].set_title("Video duration distribution")
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Duration (seconds)")

    sns.boxplot(data=df, x="dataset", y="brightness", ax=axes[1])
    axes[1].set_title("Average frame brightness distribution")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Mean pixel intensity")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path)


def _plot_representative_frames(inventory, output_dir: Path) -> str:
    plt, _, _ = _require_plotting()
    figure_dir = ensure_dir(output_dir / "figures")
    figure_path = figure_dir / f"{inventory.dataset}_representative_frames.png"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for axis, class_name in zip(axes.flatten(), CANONICAL_CLASSES):
        candidate = None
        for sample in inventory.existing_samples("train"):
            if sample.canonical_class != class_name:
                continue
            try:
                frame = extract_reference_frame(sample.abs_path)
                candidate = sample
                break
            except Exception:
                continue
        if candidate is None:
            axis.axis("off")
            continue
        axis.imshow(frame)
        axis.set_title(f"{class_name}\n{Path(candidate.rel_path).name}", fontsize=9)
        axis.axis("off")

    fig.suptitle(f"{inventory.display_name} representative training frames", fontsize=16)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path)


def generate_dataset_summary(
    data_root: Path | str,
    output_dir: Path | str,
    max_metadata_samples_per_dataset: int | None = None,
) -> dict[str, object]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    inventories = load_all_inventories(data_root)
    manifest_files = _write_manifests(inventories, output_dir)
    metadata_records, metadata_path, decode_failures = _collect_metadata(
        inventories,
        output_dir,
        max_metadata_samples_per_dataset=max_metadata_samples_per_dataset,
    )

    figure_paths = {
        "class_distribution": _plot_class_distribution(inventories, output_dir),
        "video_characteristics": _plot_video_characteristics(metadata_records, output_dir),
    }
    for dataset_name, inventory in inventories.items():
        figure_paths[f"{dataset_name}_representative_frames"] = _plot_representative_frames(inventory, output_dir)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_metadata_samples_per_dataset": max_metadata_samples_per_dataset,
        "metadata_csv": metadata_path,
        "decode_failures": decode_failures,
        "manifest_files": manifest_files,
        "figure_files": figure_paths,
        "datasets": {
            dataset_name: _build_dataset_summary(inventory, metadata_records, decode_failures)
            for dataset_name, inventory in inventories.items()
        },
    }
    summary_path = output_dir / "analysis" / "data_summary.json"
    write_json(summary_path, payload)
    payload["summary_json"] = str(summary_path)
    return payload
