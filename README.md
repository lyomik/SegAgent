# SegAgent

SegAgent is a reproducible research project for lung-nodule segmentation on
LIDC-IDRI. It separates shared dataset construction and preprocessing from
model-specific adapters so that all benchmarked models receive equivalent
inputs and ground truth.

## Project layout

- `configs/`: dataset, model, attention, and stage-specific experiment settings.
- `data/`: raw, canonical, and processed datasets (excluded from Git).
- `src/segagent/`: reusable data, model, training, evaluation, ensemble, and agent code.
- `scripts/`: command-line entry points and stage orchestration.
- `experiments/`: run outputs (excluded from Git).
- `checkpoints/`: model weights (excluded from Git).
- `notebooks/`: exploratory analysis and visualization.
- `docs/`: specifications, plans, and thesis notes.
- `system/`: the later SegAgent application layer.

The preprocessing contract is documented in
[`docs/Stage1/Preprocess_Spec.md`](docs/Stage1/Preprocess_Spec.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The current LIDC-IDRI downloader can be resumed with:

```bash
python scripts/download_lidc_idri.py \
  --metadata data/raw/LIDC-IDRI/LIDC-IDRI_MetaData.csv \
  --output data/raw/LIDC-IDRI/dicom
```
