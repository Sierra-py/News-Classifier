from sklearn.metrics import accuracy_score, classification_report
import torch
from pathlib import Path
import joblib


def test_model(model, dataloader, device):

    model.eval()

    predictions = []
    true_labels = []
    

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(classification_report(true_labels, predictions, target_names=class_names, zero_division=0))

