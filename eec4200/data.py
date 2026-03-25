from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from .constants import CANONICAL_CLASSES, CLASS_DISPLAY_NAMES, DATASET_CONFIGS


@dataclass(frozen=True)
class VideoSample:
    dataset: str
    split: str
    sample_id: int
    label: int
    class_name: str
    canonical_class: str
    rel_path: str
    abs_path: str
    exists: bool

    def to_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "sample_id": self.sample_id,
            "label": self.label,
            "class_name": self.class_name,
            "canonical_class": self.canonical_class,
            "rel_path": self.rel_path,
            "abs_path": self.abs_path,
            "exists": self.exists,
        }


@dataclass
class DatasetInventory:
    dataset: str
    display_name: str
    dataset_root: Path
    samples_by_split: dict[str, list[VideoSample]]
    listed_paths: set[str]
    actual_paths: set[str]
    missing_paths: list[str]
    ignored_paths: list[str]
    label_mismatches: list[dict[str, object]]

    def samples(self, split: str) -> list[VideoSample]:
        return list(self.samples_by_split.get(split, []))

    def existing_samples(self, split: str) -> list[VideoSample]:
        return [sample for sample in self.samples(split) if sample.exists]

    def class_counts(self, split: str, existing_only: bool = False) -> dict[str, int]:
        items = self.existing_samples(split) if existing_only else self.samples(split)
        counts = Counter(sample.canonical_class for sample in items)
        return {key: counts.get(key, 0) for key in CANONICAL_CLASSES}


def canonicalize_class_name(name: str) -> str:
    return name.strip().lower()


def display_class_name(name: str) -> str:
    return CLASS_DISPLAY_NAMES[canonicalize_class_name(name)]


def _scan_actual_paths(dataset_root: Path) -> set[str]:
    actual = set()
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file():
                actual.add(file_path.relative_to(dataset_root).as_posix())
    return actual


def load_inventory(data_root: Path | str, dataset: str) -> DatasetInventory:
    data_root = Path(data_root)
    config = DATASET_CONFIGS[dataset]
    dataset_root = data_root / config["dataset_dir"]
    actual_paths = _scan_actual_paths(dataset_root)
    samples_by_split: dict[str, list[VideoSample]] = defaultdict(list)
    listed_paths: set[str] = set()
    label_mismatches: list[dict[str, object]] = []

    for split_name, split_file in (("train", config["train_split"]), ("test", config["test_split"])):
        split_path = data_root / split_file
        for raw_line in split_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            sample_id_str, label_str, rel_path = line.split("\t")
            label = int(label_str)
            class_name = rel_path.split("/")[0]
            canonical_class = canonicalize_class_name(class_name)
            expected_class = CANONICAL_CLASSES[label]
            if canonical_class != expected_class:
                label_mismatches.append(
                    {
                        "split": split_name,
                        "sample_id": int(sample_id_str),
                        "rel_path": rel_path,
                        "expected_class": expected_class,
                        "observed_class": canonical_class,
                    }
                )
            abs_path = dataset_root / rel_path
            sample = VideoSample(
                dataset=dataset,
                split=split_name,
                sample_id=int(sample_id_str),
                label=label,
                class_name=class_name,
                canonical_class=canonical_class,
                rel_path=rel_path,
                abs_path=str(abs_path),
                exists=abs_path.exists(),
            )
            samples_by_split[split_name].append(sample)
            listed_paths.add(rel_path)

    missing_paths = sorted(
        sample.rel_path
        for split_samples in samples_by_split.values()
        for sample in split_samples
        if not sample.exists
    )
    ignored_paths = sorted(actual_paths - listed_paths)

    return DatasetInventory(
        dataset=dataset,
        display_name=config["display_name"],
        dataset_root=dataset_root,
        samples_by_split=dict(samples_by_split),
        listed_paths=listed_paths,
        actual_paths=actual_paths,
        missing_paths=missing_paths,
        ignored_paths=ignored_paths,
        label_mismatches=label_mismatches,
    )


def load_all_inventories(data_root: Path | str) -> dict[str, DatasetInventory]:
    return {dataset: load_inventory(data_root, dataset) for dataset in DATASET_CONFIGS}


def compute_split_overlap(inventory: DatasetInventory) -> list[str]:
    train_paths = {sample.rel_path for sample in inventory.samples("train")}
    test_paths = {sample.rel_path for sample in inventory.samples("test")}
    return sorted(train_paths & test_paths)


def stratified_train_val_split(
    samples: Iterable[VideoSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[VideoSample], list[VideoSample]]:
    grouped: dict[int, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    train_samples: list[VideoSample] = []
    val_samples: list[VideoSample] = []

    for label in sorted(grouped):
        class_samples = list(grouped[label])
        rng.shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * val_ratio))
        val_items = class_samples[:val_count]
        train_items = class_samples[val_count:]
        if not train_items:
            train_items = val_items[:1]
            val_items = val_items[1:]
        train_samples.extend(train_items)
        val_samples.extend(val_items)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def limit_samples_stratified(
    samples: Iterable[VideoSample],
    max_samples: int | None,
    seed: int = 42,
) -> list[VideoSample]:
    samples = list(samples)
    if max_samples is None or len(samples) <= max_samples:
        return samples

    grouped: dict[int, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    selected: list[VideoSample] = []
    class_labels = sorted(grouped)
    class_index = 0
    while len(selected) < max_samples and any(grouped.values()):
        label = class_labels[class_index % len(class_labels)]
        if grouped[label]:
            selected.append(grouped[label].pop())
        class_index += 1
    return selected


def inventory_summary_record(inventory: DatasetInventory) -> dict[str, object]:
    return {
        "dataset": inventory.dataset,
        "display_name": inventory.display_name,
        "listed_samples": len(inventory.listed_paths),
        "actual_samples": len(inventory.actual_paths),
        "missing_samples": len(inventory.missing_paths),
        "ignored_extra_samples": len(inventory.ignored_paths),
        "train_overlap_with_test": len(compute_split_overlap(inventory)),
    }
