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
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "patience": 10,
    "batch_size": 8,
}

ARID_TRAINING_DEFAULTS = {
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "patience": 10,
    "batch_size": 8,
}

OPTIMIZED_HMDB_DEFAULTS = {
    "epochs": 80,
    "lr": 5e-4,
    "weight_decay": 5e-5,
    "dropout": 0.5,
    "patience": 15,
    "batch_size": 8,
    "clip_length": 32,
    "num_test_clips": 10,
    "use_cosine_lr": True,
    "use_focal_loss": False,
    "use_photometric_aug": True,
    "photometric_brightness_range": (0.75, 1.25),
    "photometric_gamma_range": (0.75, 1.25),
    "use_temporal_jitter": True,
}

OPTIMIZED_ARID_DEFAULTS = {
    "epochs": 80,
    "lr": 5e-5,
    "weight_decay": 5e-5,
    "dropout": 0.5,
    "patience": 15,
    "batch_size": 8,
    "clip_length": 32,
    "num_test_clips": 10,
    "use_cosine_lr": True,
    "use_focal_loss": True,
    "focal_gamma": 2.0,
    "focal_alpha": None,
    "use_photometric_aug": True,
    "photometric_brightness_range": (0.7, 1.3),
    "photometric_gamma_range": (0.7, 1.3),
    "use_temporal_jitter": True,
    "two_stage_finetune": True,
    "freeze_epochs": 10,
}
