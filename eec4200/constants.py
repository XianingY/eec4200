from __future__ import annotations

CANONICAL_CLASSES: tuple[str, ...] = (
    "drink",
    "jump",
    "pick",
    "pour",
    "push",
    "run",
    "walk",
    "wave",
)

CLASS_DISPLAY_NAMES = {name: name.capitalize() for name in CANONICAL_CLASSES}

DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "hmdb51": {
        "dataset_dir": "HMDB51",
        "train_split": "hmdb51_train.txt",
        "test_split": "hmdb51_test.txt",
        "display_name": "HMDB51-mini",
    },
    "arid": {
        "dataset_dir": "ARID",
        "train_split": "arid_train.txt",
        "test_split": "arid_test.txt",
        "display_name": "ARID-mini",
    },
}

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_RANDOM_SEED = 42
DEFAULT_INPUT_SIZE = 112
DEFAULT_CLIP_LENGTH = 16
DEFAULT_TEST_CLIPS = 3

HMDB_TRAINING_DEFAULTS = {
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "patience": 5,
    "batch_size": 8,
}

ARID_TRAINING_DEFAULTS = {
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "patience": 5,
    "batch_size": 8,
}
