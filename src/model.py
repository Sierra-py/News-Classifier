"""Download pretrained model from Hugging Face"""

from transformers import AutoModelForSequenceClassification
from configs.config import MODEL_NAME, HF_MODEL_DIR, ENCODER_DIR
import joblib

def download_pretrained_model():
    label_encoder = joblib.load(ENCODER_DIR / "label_encoder.pkl")
    NUM_LABELS = len(label_encoder.classes_)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # save model
    if not HF_MODEL_DIR.exists():
        HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(HF_MODEL_DIR)

if __name__ == "__main__":
    download_pretrained_model()