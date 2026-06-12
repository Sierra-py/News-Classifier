from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import (NUM_EPOCHS, LR, MAX_LEN, BATCH_SIZE, NUM_WORKERS,
                             SPLIT_DIR, ENCODER_DIR, TOKENIZER_DIR, HF_MODEL_DIR, MODEL_DIR_V1)
from src.news_dataset import NewsDataset
import joblib
from src.train import train_epoch
from src.evaluate import evaluate
from src.tokenizer import download_pretrained_tokenizer
from src.model import download_pretrained_model
from src.preprocessing import process_data

# Loading Data
if not SPLIT_DIR.exists():
    process_data()

# X data
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
X_val   = pd.read_csv(SPLIT_DIR / "X_val.csv")
X_test  = pd.read_csv(SPLIT_DIR / "X_test.csv")

# y data
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze("columns")
y_val   = pd.read_csv(SPLIT_DIR / "y_val.csv").squeeze("columns")
y_test  = pd.read_csv(SPLIT_DIR / "y_test.csv").squeeze("columns")

# Setting device to cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class Labels
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

# Loading Saved Tokenizer
if not TOKENIZER_DIR.exists():
    # call tokenizer
    download_pretrained_tokenizer()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# Loading Saved Model
if not HF_MODEL_DIR.exists():
    # call model
    download_pretrained_model()
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR).to(device)

# load label encoder
label_encoder = joblib.load(ENCODER_DIR / "label_encoder.pkl")
NUM_LABELS = len(label_encoder.classes_)

# DataLoaders
train_dataset = NewsDataset(X_train['text'], y_train, tokenizer, max_len=MAX_LEN)
val_dataset = NewsDataset(X_val['text'], y_val, tokenizer, max_len=MAX_LEN)
test_dataset = NewsDataset(texts=X_test['text'], labels=y_test, tokenizer=tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# optimizer
optimizer = AdamW(model.parameters(), lr=LR)
num_epochs = NUM_EPOCHS
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


# Initialize the mixed precision scaler (Updated API)
scaler = torch.amp.GradScaler('cuda')

# start training
best_val_loss = float('inf')  
patience = 2                  
epochs_without_improvement = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler, device, class_weights=class_weights)
    val_loss, accuracy = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Is this the best model so far?
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        model.save_pretrained(MODEL_DIR_V1)   # save best weights
        tokenizer.save_pretrained(MODEL_DIR_V1)
        print(f"  → Best model saved (val_loss: {best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  → No improvement ({epochs_without_improvement}/{patience})")
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

# Load best weights back for evaluation
print("\nLoading best weights for final evaluation...")
del model 
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_V1).to(device)

# Test on test set
class_names_path = Path(ENCODER_DIR / "class_names.pkl")
class_names = joblib.load(class_names_path)
test_loss, test_accuracy = evaluate(model, test_loader, device, class_names=class_names, full_report=True)