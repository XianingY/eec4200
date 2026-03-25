from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Mapping


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: Path | str, payload: object) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return output_path


def read_json(path: Path | str) -> object:
    return json.loads(Path(path).read_text())


def write_records_csv(path: Path | str, records: Iterable[Mapping[str, object]]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    records = list(records)
    if not records:
        output_path.write_text("")
        return output_path
    fieldnames = list(records[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def safe_mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def safe_median(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    middle = len(items) // 2
    if len(items) % 2 == 1:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2
