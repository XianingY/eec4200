# EEC4200 Course Project Toolkit

This repository contains a small end-to-end toolkit for the EEC4200 video classification coursework. It supports:

- split parsing and data QA for HMDB51-mini and ARID-mini
- dataset summarization with figures and manifests
- lightweight 3D-CNN training on HMDB51-mini
- zero-shot cross-dataset evaluation on ARID-mini
- ARID-focused fine-tuning with illumination-aware preprocessing
- English report generation as Markdown and PDF

## Install

```bash
python3 -m pip install -r requirements.txt
```

For a lighter CPU-only local smoke-test environment, install `torch` from the CPU wheel index and install the remaining dependencies normally:

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install numpy pandas matplotlib seaborn scikit-learn tqdm reportlab opencv-python-headless
```

## CLI

```bash
python3 -m eec4200 summarize-data --data-root src --output-dir outputs
python3 -m eec4200 train-hmdb --data-root src --output-dir outputs
python3 -m eec4200 eval-cross-dataset --data-root src --output-dir outputs
python3 -m eec4200 train-arid --data-root src --output-dir outputs
python3 -m eec4200 build-report --data-root src --output-dir outputs
```

Useful smoke-test flags:

```bash
python3 -m eec4200 summarize-data --data-root src --output-dir outputs --max-metadata-samples-per-dataset 64
python3 -m eec4200 train-hmdb --data-root src --output-dir outputs --epochs 1 --max-train-samples 32 --max-val-samples 16 --max-test-samples 16 --num-workers 0
python3 -m eec4200 train-arid --data-root src --output-dir outputs --epochs 1 --max-train-samples 32 --max-val-samples 16 --max-test-samples 16 --num-workers 0
```

## Final GPU Run

Recommended full-run assumptions:

- use a CUDA-capable GPU
- keep the official full splits with no sample caps
- keep `seed=42` for the final reported run
- start with `batch-size=8`; if you hit CUDA OOM, reduce to `4`
- start with `num-workers=4`; increase only if your storage is fast enough

One-command full pipeline:

```bash
chmod +x scripts/run_full_gpu_pipeline.sh
PYTHON_BIN=.venv/bin/python DEVICE=cuda OUTPUT_DIR=outputs_final scripts/run_full_gpu_pipeline.sh
```

The script runs the full submission pipeline in order and stores logs under `outputs_final/logs/`.

Equivalent explicit commands:

```bash
.venv/bin/python -m eec4200 summarize-data --data-root src --output-dir outputs_final
.venv/bin/python -m eec4200 train-hmdb --data-root src --output-dir outputs_final --device cuda --epochs 30 --batch-size 8 --lr 1e-3 --weight-decay 1e-4 --patience 5 --num-workers 4 --seed 42
.venv/bin/python -m eec4200 eval-cross-dataset --data-root src --output-dir outputs_final --checkpoint outputs_final/checkpoints/hmdb_best.pt --device cuda --clip-length 16 --image-size 112 --num-test-clips 3 --seed 42
.venv/bin/python -m eec4200 train-arid --data-root src --output-dir outputs_final --checkpoint outputs_final/checkpoints/hmdb_best.pt --device cuda --epochs 20 --batch-size 8 --lr 3e-4 --weight-decay 1e-4 --patience 5 --num-workers 4 --seed 42
.venv/bin/python -m eec4200 build-report --output-dir outputs_final
```

Useful overrides for the GPU script:

```bash
HMDB_BATCH_SIZE=4 ARID_BATCH_SIZE=4 scripts/run_full_gpu_pipeline.sh
NUM_WORKERS=8 scripts/run_full_gpu_pipeline.sh
OUTPUT_DIR=outputs_seed7 SEED=7 scripts/run_full_gpu_pipeline.sh
DRY_RUN=1 scripts/run_full_gpu_pipeline.sh
```

## Output Layout

- `outputs/manifests/`: parsed manifests and QA CSV files
- `outputs/analysis/`: dataset statistics, figures, and summary JSON
- `outputs/checkpoints/`: trained model checkpoints
- `outputs/metrics/`: JSON and CSV metrics
- `outputs/figures/`: learning curves, confusion matrices, and sample-frame grids
- `outputs/report/`: generated Markdown and PDF report

## Experiment Results (RTX 5090, seed=42)

| Experiment | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| HMDB51-mini test | 44.58% | 43.13% | 23 epochs (early stopped @ 18), 445/111/240 split |
| ARID-mini zero-shot | 13.82% | 5.09% | HMDB checkpoint, 1289 test samples |
| ARID-mini fine-tuned | 20.31% | 17.68% | CLAHE + photometric aug + class-balanced sampling, 128 test samples |

**Key findings**:
- HMDB51-mini is class-balanced (70 train / 30 test per class); ARID-mini is heavily imbalanced
- ARID brightness median (10.1) is ~6.7x lower than HMDB (67.6), confirming severe low-light domain shift
- Zero-shot transfer drops from 44.58% → 13.82%, demonstrating significant domain gap
- Fine-tuning with illumination-aware preprocessing recovers to 20.31% (+6.5pp over zero-shot)
