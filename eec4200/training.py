from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
from typing import Iterable

from .constants import (
    ARID_TRAINING_DEFAULTS,
    CANONICAL_CLASSES,
    DEFAULT_CLIP_LENGTH,
    DEFAULT_INPUT_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_CLIPS,
    HMDB_TRAINING_DEFAULTS,
)
from .data import (
    DatasetInventory,
    VideoSample,
    limit_samples_stratified,
    load_all_inventories,
    stratified_train_val_split,
)
from .model import Lightweight3DCNN
from .utils import ensure_dir, seed_everything, write_json, write_records_csv
from .video import load_video_clip


def _require_training_stack():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        import torch.nn as nn
        from sklearn.metrics import confusion_matrix, f1_score
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return {
        "plt": plt,
        "sns": sns,
        "torch": torch,
        "nn": nn,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "WeightedRandomSampler": WeightedRandomSampler,
        "tqdm": tqdm,
    }


@dataclass
class ExperimentConfig:
    dataset_name: str
    output_dir: str
    checkpoint_name: str
    metrics_name: str
    history_name: str
    confusion_name: str
    curves_name: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    num_workers: int
    seed: int = DEFAULT_RANDOM_SEED
    clip_length: int = DEFAULT_CLIP_LENGTH
    image_size: int = DEFAULT_INPUT_SIZE
    num_test_clips: int = DEFAULT_TEST_CLIPS
    device: str = "auto"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    use_clahe: bool = False
    use_photometric_aug: bool = False
    class_balanced_sampling: bool = False
    init_checkpoint: str | None = None


def _resolve_device(requested: str):
    stack = _require_training_stack()
    torch = stack["torch"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _worker_seed_init(worker_id: int) -> None:
    seed = DEFAULT_RANDOM_SEED + worker_id
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class VideoClipDataset:
    def __new__(
        cls,
        samples: list[VideoSample],
        clip_length: int,
        image_size: int,
        training: bool,
        seed: int,
        use_clahe: bool = False,
        use_photometric_aug: bool = False,
    ):
        stack = _require_training_stack()
        Dataset = stack["Dataset"]
        torch = stack["torch"]

        class _VideoClipDataset(Dataset):
            def __init__(self):
                self.samples = samples
                self.clip_length = clip_length
                self.image_size = image_size
                self.training = training
                self.seed = seed
                self.use_clahe = use_clahe
                self.use_photometric_aug = use_photometric_aug

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index: int):
                original = self.samples[index]
                same_class_indices = [
                    idx for idx, candidate in enumerate(self.samples) if candidate.label == original.label
                ]
                candidate_indices = [index] + [idx for idx in same_class_indices if idx != index]
                last_error = None
                for candidate_index in candidate_indices:
                    sample = self.samples[candidate_index]
                    try:
                        rng = random.Random(self.seed + candidate_index)
                        clip = load_video_clip(
                            sample.abs_path,
                            clip_length=self.clip_length,
                            image_size=self.image_size,
                            clip_index=0,
                            num_clips=1,
                            jitter=self.training,
                            apply_clahe=self.use_clahe,
                            apply_random_photometric_aug=self.training and self.use_photometric_aug,
                            random_horizontal_flip=self.training,
                            rng=rng,
                        )
                        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
                        label = torch.tensor(original.label, dtype=torch.long)
                        return tensor, label
                    except Exception as exc:
                        last_error = exc
                        continue
                raise RuntimeError(f"Unable to decode any sample for class {original.label}: {last_error}")

        return _VideoClipDataset()


def _split_inventory_samples(
    inventory: DatasetInventory,
    seed: int,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[list[VideoSample], list[VideoSample], list[VideoSample]]:
    train_existing = inventory.existing_samples("train")
    test_existing = inventory.existing_samples("test")
    train_samples, val_samples = stratified_train_val_split(train_existing, val_ratio=0.2, seed=seed)
    train_samples = limit_samples_stratified(train_samples, max_train_samples, seed=seed)
    val_samples = limit_samples_stratified(val_samples, max_val_samples, seed=seed + 1)
    test_samples = limit_samples_stratified(test_existing, max_test_samples, seed=seed + 2)
    return train_samples, val_samples, test_samples


def _build_loader(samples, config: ExperimentConfig, training: bool):
    stack = _require_training_stack()
    DataLoader = stack["DataLoader"]
    WeightedRandomSampler = stack["WeightedRandomSampler"]
    dataset = VideoClipDataset(
        samples=samples,
        clip_length=config.clip_length,
        image_size=config.image_size,
        training=training,
        seed=config.seed,
        use_clahe=config.use_clahe,
        use_photometric_aug=config.use_photometric_aug,
    )

    sampler = None
    shuffle = training
    if training and config.class_balanced_sampling:
        counts = Counter(sample.label for sample in samples)
        weights = [1.0 / counts[sample.label] for sample in samples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(samples), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=_resolve_device(config.device).type == "cuda",
        worker_init_fn=_worker_seed_init,
    )


def _epoch_pass(model, loader, optimizer, loss_fn, device, training: bool):
    stack = _require_training_stack()
    torch = stack["torch"]
    tqdm = stack["tqdm"]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    model.train(training)

    iterator = tqdm(loader, leave=False)
    for inputs, targets in iterator:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = loss_fn(logits, targets)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = logits.argmax(dim=1)
        batch_correct = int((predictions == targets).sum().item())
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += batch_correct
        total_seen += batch_size
        iterator.set_postfix(
            loss=f"{(total_loss / max(1, total_seen)):.4f}",
            acc=f"{(total_correct / max(1, total_seen)):.3f}",
        )

    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
        "num_samples": total_seen,
    }


def _evaluate_samples(
    model,
    samples: list[VideoSample],
    device,
    clip_length: int,
    image_size: int,
    num_test_clips: int,
    use_clahe: bool,
):
    stack = _require_training_stack()
    torch = stack["torch"]
    confusion_matrix = stack["confusion_matrix"]
    f1_score = stack["f1_score"]
    tqdm = stack["tqdm"]

    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    per_class_totals = Counter()
    per_class_correct = Counter()
    skipped_samples: list[dict[str, str]] = []

    with torch.no_grad():
        for sample in tqdm(samples, leave=False):
            try:
                clip_logits = []
                for clip_index in range(num_test_clips):
                    clip = load_video_clip(
                        sample.abs_path,
                        clip_length=clip_length,
                        image_size=image_size,
                        clip_index=clip_index,
                        num_clips=num_test_clips,
                        jitter=False,
                        apply_clahe=use_clahe,
                        apply_random_photometric_aug=False,
                        random_horizontal_flip=False,
                        rng=random.Random(clip_index + sample.sample_id),
                    )
                    inputs = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(device)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        clip_logits.append(model(inputs).cpu())
                mean_logits = torch.stack(clip_logits, dim=0).mean(dim=0)
                prediction = int(mean_logits.argmax(dim=1).item())
                all_targets.append(sample.label)
                all_predictions.append(prediction)
                per_class_totals[sample.label] += 1
                if prediction == sample.label:
                    per_class_correct[sample.label] += 1
            except Exception as exc:
                skipped_samples.append({"rel_path": sample.rel_path, "reason": str(exc)})
                continue

    accuracy = (
        sum(int(target == prediction) for target, prediction in zip(all_targets, all_predictions))
        / max(1, len(all_targets))
    )
    macro_f1 = float(f1_score(all_targets, all_predictions, labels=list(range(len(CANONICAL_CLASSES))), average="macro"))
    per_class_accuracy = {
        CANONICAL_CLASSES[index]: per_class_correct[index] / max(1, per_class_totals[index])
        for index in range(len(CANONICAL_CLASSES))
    }
    confusion = confusion_matrix(all_targets, all_predictions, labels=list(range(len(CANONICAL_CLASSES))))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "num_samples": len(all_targets),
        "requested_num_samples": len(samples),
        "skipped_samples": skipped_samples,
        "targets": all_targets,
        "predictions": all_predictions,
    }


def _plot_learning_curves(history_records: list[dict[str, object]], output_path: Path) -> str:
    stack = _require_training_stack()
    plt = stack["plt"]
    sns = stack["sns"]
    sns.set_theme(style="whitegrid")
    epochs = [record["epoch"] for record in history_records]
    train_loss = [record["train_loss"] for record in history_records]
    val_loss = [record["val_loss"] for record in history_records]
    train_acc = [record["train_accuracy"] for record in history_records]
    val_acc = [record["val_accuracy"] for record in history_records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_confusion(confusion: list[list[int]], output_path: Path, title: str) -> str:
    stack = _require_training_stack()
    plt = stack["plt"]
    sns = stack["sns"]
    sns.set_theme(style="white")
    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=CANONICAL_CLASSES, yticklabels=CANONICAL_CLASSES, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _save_metrics(
    metrics: dict[str, object],
    output_dir: Path,
    metrics_name: str,
    confusion_name: str,
    title: str,
) -> dict[str, object]:
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")
    confusion_path = figures_dir / confusion_name
    metrics["confusion_matrix_png"] = _plot_confusion(metrics["confusion_matrix"], confusion_path, title)
    confusion_csv_path = metrics_dir / f"{Path(metrics_name).stem}_confusion_matrix.csv"
    write_records_csv(
        confusion_csv_path,
        [
            {"true_class": CANONICAL_CLASSES[row_index], **{CANONICAL_CLASSES[col_index]: value for col_index, value in enumerate(row)}}
            for row_index, row in enumerate(metrics["confusion_matrix"])
        ],
    )
    metrics["confusion_matrix_csv"] = str(confusion_csv_path)
    metrics_path = metrics_dir / metrics_name
    write_json(metrics_path, metrics)
    metrics["metrics_json"] = str(metrics_path)
    return metrics


def _prepare_model_and_optimizer(config: ExperimentConfig, device):
    stack = _require_training_stack()
    torch = stack["torch"]
    nn = stack["nn"]
    model = Lightweight3DCNN(num_classes=len(CANONICAL_CLASSES), dropout=0.4).to(device)

    if config.init_checkpoint:
        checkpoint = torch.load(config.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn


def _save_checkpoint(model, config: ExperimentConfig, best_val_accuracy: float, path: Path, history_records: list[dict[str, object]]):
    stack = _require_training_stack()
    torch = stack["torch"]
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "class_names": list(CANONICAL_CLASSES),
            "best_val_accuracy": best_val_accuracy,
            "history": history_records,
        },
        path,
    )
    return str(path)


def run_supervised_experiment(
    inventory: DatasetInventory,
    config: ExperimentConfig,
) -> dict[str, object]:
    stack = _require_training_stack()
    torch = stack["torch"]
    output_dir = Path(config.output_dir)
    device = _resolve_device(config.device)
    seed_everything(config.seed)

    train_samples, val_samples, test_samples = _split_inventory_samples(
        inventory,
        seed=config.seed,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        max_test_samples=config.max_test_samples,
    )

    train_loader = _build_loader(train_samples, config, training=True)
    val_loader = _build_loader(val_samples, config, training=False)
    model, optimizer, loss_fn = _prepare_model_and_optimizer(config, device)

    history_records: list[dict[str, object]] = []
    best_val_accuracy = -math.inf
    best_checkpoint = output_dir / "checkpoints" / config.checkpoint_name
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_stats = _epoch_pass(model, train_loader, optimizer, loss_fn, device, training=True)
        val_stats = _epoch_pass(model, val_loader, optimizer, loss_fn, device, training=False)
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
        }
        history_records.append(record)

        if val_stats["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_stats["accuracy"]
            epochs_without_improvement = 0
            _save_checkpoint(model, config, best_val_accuracy, best_checkpoint, history_records)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    metrics = _evaluate_samples(
        model,
        test_samples,
        device=device,
        clip_length=config.clip_length,
        image_size=config.image_size,
        num_test_clips=config.num_test_clips,
        use_clahe=config.use_clahe,
    )
    metrics.update(
        {
            "dataset": inventory.dataset,
            "display_name": inventory.display_name,
            "evaluation_name": config.metrics_name.replace(".json", ""),
            "num_test_clips": config.num_test_clips,
            "checkpoint": str(best_checkpoint),
            "device_used": str(device),
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
            "config": asdict(config),
        }
    )

    history_path = Path(config.output_dir) / "metrics" / config.history_name
    write_records_csv(history_path, history_records)
    metrics["history_csv"] = str(history_path)
    metrics["learning_curves_png"] = _plot_learning_curves(history_records, Path(config.output_dir) / "figures" / config.curves_name)
    metrics = _save_metrics(
        metrics,
        output_dir=Path(config.output_dir),
        metrics_name=config.metrics_name,
        confusion_name=config.confusion_name,
        title=f"{inventory.display_name} confusion matrix",
    )
    return metrics


def run_cross_dataset_evaluation(
    source_checkpoint: Path | str,
    target_inventory: DatasetInventory,
    output_dir: Path | str,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    image_size: int = DEFAULT_INPUT_SIZE,
    num_test_clips: int = DEFAULT_TEST_CLIPS,
    device: str = "auto",
    max_test_samples: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    stack = _require_training_stack()
    torch = stack["torch"]
    output_dir = Path(output_dir)
    device_obj = _resolve_device(device)
    seed_everything(seed)
    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    model = Lightweight3DCNN(num_classes=len(CANONICAL_CLASSES), dropout=0.4).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    test_samples = limit_samples_stratified(target_inventory.existing_samples("test"), max_test_samples, seed=seed)
    metrics = _evaluate_samples(
        model,
        test_samples,
        device=device_obj,
        clip_length=clip_length,
        image_size=image_size,
        num_test_clips=num_test_clips,
        use_clahe=False,
    )
    metrics.update(
        {
            "dataset": target_inventory.dataset,
            "display_name": target_inventory.display_name,
            "evaluation_name": "arid_zero_shot",
            "num_test_clips": num_test_clips,
            "checkpoint": str(source_checkpoint),
            "device_used": str(device_obj),
            "test_size": len(test_samples),
        }
    )
    return _save_metrics(
        metrics,
        output_dir=output_dir,
        metrics_name="arid_zero_shot_metrics.json",
        confusion_name="arid_zero_shot_confusion_matrix.png",
        title="ARID-mini zero-shot confusion matrix",
    )


def train_hmdb(
    data_root: Path | str,
    output_dir: Path | str,
    epochs: int = HMDB_TRAINING_DEFAULTS["epochs"],
    batch_size: int = HMDB_TRAINING_DEFAULTS["batch_size"],
    lr: float = HMDB_TRAINING_DEFAULTS["lr"],
    weight_decay: float = HMDB_TRAINING_DEFAULTS["weight_decay"],
    patience: int = HMDB_TRAINING_DEFAULTS["patience"],
    num_workers: int = 0,
    seed: int = DEFAULT_RANDOM_SEED,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    inventories = load_all_inventories(Path(data_root))
    config = ExperimentConfig(
        dataset_name="hmdb51",
        output_dir=str(output_dir),
        checkpoint_name="hmdb_best.pt",
        metrics_name="hmdb_test_metrics.json",
        history_name="hmdb_training_history.csv",
        confusion_name="hmdb_confusion_matrix.png",
        curves_name="hmdb_learning_curves.png",
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        num_workers=num_workers,
        seed=seed,
        device=device,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    return run_supervised_experiment(inventories["hmdb51"], config)


def train_arid(
    data_root: Path | str,
    output_dir: Path | str,
    init_checkpoint: Path | str,
    epochs: int = ARID_TRAINING_DEFAULTS["epochs"],
    batch_size: int = ARID_TRAINING_DEFAULTS["batch_size"],
    lr: float = ARID_TRAINING_DEFAULTS["lr"],
    weight_decay: float = ARID_TRAINING_DEFAULTS["weight_decay"],
    patience: int = ARID_TRAINING_DEFAULTS["patience"],
    num_workers: int = 0,
    seed: int = DEFAULT_RANDOM_SEED,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    inventories = load_all_inventories(Path(data_root))
    config = ExperimentConfig(
        dataset_name="arid",
        output_dir=str(output_dir),
        checkpoint_name="arid_finetuned_best.pt",
        metrics_name="arid_finetuned_test_metrics.json",
        history_name="arid_finetuned_training_history.csv",
        confusion_name="arid_finetuned_confusion_matrix.png",
        curves_name="arid_finetuned_learning_curves.png",
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        num_workers=num_workers,
        seed=seed,
        device=device,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        use_clahe=True,
        use_photometric_aug=True,
        class_balanced_sampling=True,
        init_checkpoint=str(init_checkpoint),
    )
    return run_supervised_experiment(inventories["arid"], config)
