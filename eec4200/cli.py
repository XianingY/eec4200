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
    summarize.add_argument("--data-root", default="src", help="Dataset root containing split files and video folders")
    summarize.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated artifacts")
    summarize.add_argument(
        "--max-metadata-samples-per-dataset",
        type=int,
        default=None,
        help="Optional stratified cap for video metadata probing, useful for smoke tests.",
    )

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
    train_arid_parser.add_argument(
        "--checkpoint",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "checkpoints" / "hmdb_best.pt"),
        help="Initialization checkpoint from the HMDB baseline",
    )

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
        epochs = args.epochs if args.epochs is not None else 30
        lr = args.lr if args.lr is not None else 1e-3
        result = train_hmdb(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            epochs=epochs,
            batch_size=args.batch_size,
            lr=lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
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
            image_size=args.image_size,
            num_test_clips=args.num_test_clips,
            device=args.device,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
        )
        print(f"Cross-dataset metrics written to {result['metrics_json']}")
        return

    if args.command == "train-arid":
        epochs = args.epochs if args.epochs is not None else 20
        lr = args.lr if args.lr is not None else 3e-4
        result = train_arid(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            init_checkpoint=Path(args.checkpoint),
            epochs=epochs,
            batch_size=args.batch_size,
            lr=lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
        )
        print(f"ARID metrics written to {result['metrics_json']}")
        return

    if args.command == "build-report":
        result = build_report(Path(args.output_dir))
        print(f"Report written to {result['pdf']}")
        return

    parser.error(f"Unsupported command: {args.command}")
