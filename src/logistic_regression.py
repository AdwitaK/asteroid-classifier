from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import preprocess

x_train, y_train, x_val, y_val, x_test, y_test = preprocess("../data/nasa.csv")

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(x_train, y_train)

preds = model.predict(x_val)

accuracy = accuracy_score(y_val, preds)
precision = precision_score(y_val, preds)
recall = recall_score(y_val, preds)
f1 = f1_score(y_val, preds)
cm = confusion_matrix(y_val, preds)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)