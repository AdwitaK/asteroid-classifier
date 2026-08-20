import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import preprocess, convert_to_tensors
from dataset import AsteroidDataset
from torch.utils.data import DataLoader
from model import AsteroidClassifier

# Load data
data = preprocess("../data/nasa.csv")
x_train, y_train, x_val, y_val, x_test, y_test = convert_to_tensors(data)

val_data = AsteroidDataset(x_val, y_val)
val_loader = DataLoader(val_data, 64)

# Load model
model = AsteroidClassifier(x_train.shape[1])
model.load_state_dict(torch.load("asteroid_model.pt"))

# Inference
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        logits = model(x_batch).squeeze()

        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).float()

        all_preds.extend(preds.numpy())

        all_labels.extend(y_batch.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
confusion_matrix = confusion_matrix(all_labels, all_preds)

# Show results
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix)
