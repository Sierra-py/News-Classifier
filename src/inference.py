from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import torch
from configs.config import MODEL_DIR_V1, TOKENIZER_DIR, ENCODER_DIR

def load_artifacts():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_V1)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    label_encoder = joblib.load(ENCODER_DIR / "label_encoder.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder, device

def predict(text, model, tokenizer, label_encoder, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    model, tokenizer, label_encoder, device = load_artifacts()
    text = input("Enter Your News Article...")
    print(predict(text, model, tokenizer, label_encoder, device))
