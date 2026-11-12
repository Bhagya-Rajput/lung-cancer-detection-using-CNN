# Lung Cancer Detection using CNN

An educational CNN project for classifying lung CT images from the IQ-OTHNCCD dataset into three dataset labels: Benign, Malignant, and Normal.

> **Medical disclaimer:** This is an educational machine-learning project, not a medical device. It must not be used to diagnose cancer, rule out cancer, or make treatment decisions.

## Dataset

Place the IQ-OTHNCCD dataset in the repository root with these exact folder names:

```
IQ-OTHNCCD/
├── Bengin cases/
├── Malignant cases/
└── Normal cases/
```

The spelling `Bengin cases` is retained to match the folder name in the dataset.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the model

Run from the repository root:

```bash
python main.py
```

Training resizes images to 128×128, converts them to RGB, scales pixel values to [0, 1], uses a stratified train/validation split, and saves the best validation checkpoint to:

```
models/lung_cancer_model.keras
models/class_names.json
```

The training script prints a classification report and confusion matrix. Review per-class precision, recall, and F1-score rather than accuracy alone.

## Run the Streamlit app

```bash
streamlit run app.py
```

Upload a JPG, JPEG, PNG, BMP, or TIFF lung CT image. The app uses the same resize, RGB conversion, and [0, 1] scaling as training.

## Important limitation: images outside the dataset

A three-class softmax classifier always produces scores for its three trained classes, even for unrelated images. The app includes a conservative low-score warning, but a softmax threshold is **not** reliable out-of-distribution detection and cannot guarantee rejection of photographs, other scans, or unfamiliar CT images.

For a robust reject option, collect a separate, representative set of out-of-distribution examples (non-lung images, other medical scans, and CT images from different sources), then evaluate and calibrate an explicit rejection method. Do not interpret the displayed softmax score as a clinical probability.

## Project files

- `main.py` — dataset loading, CNN training, validation and model saving
- `app.py` — Streamlit inference interface
- `create_model.py` — guard against accidentally creating a dummy untrained model
- `requirements.txt` — Python dependencies
