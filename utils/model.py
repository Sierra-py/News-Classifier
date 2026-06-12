"""Download pretrained model from Hugging Face"""

from transformers import AutoModelForSequenceClassification
from configs.config import MODEL_NAME
import joblib


label_encoder = joblib.load("artifacts/encoder/label_encoder.pkl")
NUM_LABELS = len(label_encoder.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# save model
model.save_pretrained("artifacts/models")