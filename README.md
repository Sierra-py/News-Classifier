# News Classifier

An end-to-end pipeline for training a BERT-based news article classifier across 36 categories.

## Project Structure

```
NEWS-CLASSIFIER/
├── artifacts/
│   ├── encoder/
│   │   ├── class_names.pkl
│   │   └── label_encoder.pkl
│   ├── models/
│   │   ├── hf_pretrained/
│   │   └── model_v1/
│   └── tokenizer/
├── configs/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── notebooks/
│   └── news-classifier.ipynb
├── src/
│   ├── evaluate.py
│   ├── extract_data.py
│   ├── inference.py
│   ├── model.py
│   ├── news_dataset.py
│   ├── preprocessing.py
│   ├── tokenizer.py
│   └── train.py
├── tests/
│   └── data_load_check.py
├── main.py
└── README.md
```

## About the Data

Dataset: [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) — 209k+ articles with headline, short description, and category label.

### Preprocessing decisions

**Category merging** — 42 original categories reduced to 36 by merging overlapping or near-duplicate labels (e.g. `WORLDPOST` + `THE WORLDPOST` → `WORLD NEWS`, `PARENTING` → `PARENTS`, `STYLE` → `STYLE & BEAUTY`).

**Feature construction** — `headline` and `short_description` concatenated into a single `text` column. Other columns (link, authors, date) dropped.

**Class imbalance** — handled via class weights during training to penalize errors on rare categories more heavily.

## About the Model

Pre-trained `bert-base-uncased` fine-tuned for sequence classification using HuggingFace Transformers.

BERT was chosen for its bidirectional context understanding and strong performance on short text classification tasks without training from scratch.

## Training

Two-stage approach:

**Stage 1 — Full dataset training** (3 epochs, early stopping with patience=2):
- Validation accuracy: 0.6711
- Test accuracy: 0.6704
- Macro F1: 0.60


Fine-tuning marginally improved recall on rare classes but slightly hurt overall accuracy. The initial model (`model_v1`) is used for inference.

## Setup

**Requirements**

Python 3.10+

```bash
pip install torch transformers scikit-learn pandas numpy tqdm joblib kaggle
```

GPU recommended for training. CPU sufficient for inference only.

**Kaggle credentials**

The pipeline downloads the dataset automatically via the Kaggle API. Set up your credentials first:

```bash
# Place kaggle.json in ~/.kaggle/
# or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

## Running the Pipeline

```bash
python main.py
```

`main.py` is the single entry point. It runs the full pipeline in order:

1. Downloads and extracts raw data if not present
2. Preprocesses data, encodes labels, creates train/val/test splits
3. Downloads pretrained BERT tokenizer and model weights if not present
4. Trains with early stopping, saves best checkpoint to `artifacts/models/model_v1/`
5. Evaluates on the held-out test set and prints classification report

Intermediate artifacts are cached — re-running skips steps whose outputs already exist.

## Inference

```python
from src.inference import load_artifacts, predict

model, tokenizer, label_encoder, device = load_artifacts()

predict("NASA launches new telescope to explore deep space", model, tokenizer, label_encoder, device)
# → 'SCIENCE'
```

## Results

Test set performance (model_v1, 41,906 samples):

| Metric | Score |
|---|---|
| Accuracy | 0.6704 |
| Macro F1 | 0.60 |
| Weighted F1 | 0.68 |

Selected per-class results:

| Category | F1 |
|---|---|
| STYLE & BEAUTY | 0.85 |
| HOME & LIVING | 0.83 |
| TRAVEL | 0.84 |
| WEDDINGS | 0.84 |
| POLITICS | 0.72 |
| EDUCATION | 0.39 |
| U.S. NEWS | 0.32 |
| FIFTY | 0.41 |

## Limitations

- Categories with vague definitions (FIFTY, GOOD NEWS, IMPACT) consistently underperform regardless of training approach
- POLITICS recall is lower than expected due to semantic overlap with U.S. NEWS, WORLD NEWS, and WOMEN
- Model assumes English input only
- MAX_LEN of 128 tokens may truncate longer descriptions
