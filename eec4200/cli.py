from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import generate_dataset_summary
from .constants import DEFAULT_OUTPUT_DIR
from .report import build_report
from .training import run_cross_dataset_evaluation, train_arid, train_hmdb


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEC4200 coursework toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize-data", help="Run data QA and generate dataset figures")
    summarize.add_argument("--data-root", default="src")
    summarize.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    summarize.add_argument("--max-metadata-samples-per-dataset", type=int, default=None)

    train_hmdb_parser = subparsers.add_parser("train-hmdb", help="Train the HMDB51-mini 3D-CNN baseline")
    _add_training_args(train_hmdb_parser)

    cross = subparsers.add_parser("eval-cross-dataset", help="Evaluate the HMDB checkpoint directly on ARID-mini")
    cross.add_argument("--data-root", default="src")
    cross.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    cross.add_argument("--checkpoint", default=str(Path(DEFAULT_OUTPUT_DIR) / "checkpoints" / "hmdb_best.pt"))
    cross.add_argument("--device", default="auto")
    cross.add_argument("--max-test-samples", type=int, default=None)
    cross.add_argument("--clip-length", type=int, default=16)
    cross.add_argument("--image-size", type=int, default=112)
    cross.add_argument("--num-test-clips", type=int, default=3)
    cross.add_argument("--seed", type=int, default=42)

    train_arid_parser = subparsers.add_parser("train-arid", help="Fine-tune the workflow on ARID-mini")
    _add_training_args(train_arid_parser)
    train_arid_parser.add_argument("--checkpoint", default=str(Path(DEFAULT_OUTPUT_DIR) / "checkpoints" / "hmdb_best.pt"))
    train_arid_parser.add_argument("--use-focal-loss", action="store_true", default=False)
    train_arid_parser.add_argument("--focal-gamma", type=float, default=2.0)
    train_arid_parser.add_argument("--two-stage-finetune", action="store_true", default=False)
    train_arid_parser.add_argument("--freeze-epochs", type=int, default=10)
    train_arid_parser.add_argument("--brightness-range", type=float, nargs=2, default=None)
    train_arid_parser.add_argument("--gamma-range", type=float, nargs=2, default=None)

    report_parser = subparsers.add_parser("build-report", help="Generate Markdown and PDF report")
    report_parser.add_argument("--data-root", default="src")
    report_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    return parser


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="src")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--use-cosine-lr", action="store_true", default=False)
    parser.add_argument("--use-photometric-aug", action="store_true", default=False)
    parser.add_argument("--use-temporal-jitter", action="store_true", default=False)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--num-test-clips", type=int, default=3)


def main() -> None:
    parser = _base_parser()
    args = parser.parse_args()

    if args.command == "summarize-data":
        result = generate_dataset_summary(
            Path(args.data_root),
            Path(args.output_dir),
            max_metadata_samples_per_dataset=args.max_metadata_samples_per_dataset,
        )
        print(f"Summary written to {result['summary_json']}")
        return

    if args.command == "train-hmdb":
        epochs = args.epochs if args.epochs is not None else 50
        lr = args.lr if args.lr is not None else 5e-4
        patience = args.patience if args.patience is not None else 15
        clip_length = getattr(args, "clip_length", 16)
        num_test_clips = getattr(args, "num_test_clips", 3)
        result = train_hmdb(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            epochs=epochs,
            batch_size=args.batch_size,
            lr=lr,
            weight_decay=args.weight_decay,
            patience=patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            clip_length=clip_length,
            num_test_clips=num_test_clips,
            use_cosine_lr=getattr(args, "use_cosine_lr", False),
            use_photometric_aug=getattr(args, "use_photometric_aug", False),
            use_temporal_jitter=getattr(args, "use_temporal_jitter", False),
        )
        print(f"HMDB metrics written to {result['metrics_json']}")
        return

    if args.command == "eval-cross-dataset":
        from .data import load_all_inventories

        inventories = load_all_inventories(Path(args.data_root))
        result = run_cross_dataset_evaluation(
            source_checkpoint=Path(args.checkpoint),
            target_inventory=inventories["arid"],
            output_dir=Path(args.output_dir),
            clip_length=args.clip_length,
            image_size=112,
            num_test_clips=args.num_test_clips,
            device=args.device,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )
        print(f"Cross-dataset metrics written to {result['metrics_json']}")
        return

    if args.command == "train-arid":
        epochs = args.epochs if args.epochs is not None else 50
        lr = args.lr if args.lr is not None else 5e-5
        patience = args.patience if args.patience is not None else 15
        clip_length = getattr(args, "clip_length", 16)
        num_test_clips = getattr(args, "num_test_clips", 3)
        brightness_range = getattr(args, "brightness_range", None) or (0.7, 1.3)
        gamma_range = getattr(args, "gamma_range", None) or (0.7, 1.3)
        result = train_arid(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            init_checkpoint=Path(args.checkpoint),
            epochs=epochs,
            batch_size=args.batch_size,
            lr=lr,
            weight_decay=args.weight_decay,
            patience=patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            clip_length=clip_length,
            num_test_clips=num_test_clips,
            use_cosine_lr=getattr(args, "use_cosine_lr", False),
            use_focal_loss=getattr(args, "use_focal_loss", False),
            focal_gamma=getattr(args, "focal_gamma", 2.0),
            use_photometric_aug=True,
            photometric_brightness_range=tuple(brightness_range),
            photometric_gamma_range=tuple(gamma_range),
            use_temporal_jitter=getattr(args, "use_temporal_jitter", False),
            two_stage_finetune=getattr(args, "two_stage_finetune", False),
            freeze_epochs=getattr(args, "freeze_epochs", 10),
        )
        print(f"ARID metrics written to {result['metrics_json']}")
        return

    if args.command == "build-report":
        result = build_report(Path(args.output_dir))
        print(f"Report written to {result['pdf']}")
        return

    parser.error(f"Unsupported command: {args.command}")
