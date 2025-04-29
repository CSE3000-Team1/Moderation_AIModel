import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load model, vectorizer, and data
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Split data (same way as in model training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neither"}

# === FULL CLASSIFICATION REPORT ===
print("=== Full Classification Report ===")
report = classification_report(y_test, y_pred, target_names=list(label_map.values()))
print(report)

# === PER-CLASS METRICS (Precision, Recall, F1) ===
print("\n=== Per-Class Metrics ===")
for label, name in label_map.items():
    prec = precision_score(y_test, y_pred, labels=[label], average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, labels=[label], average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=[label], average='macro', zero_division=0)
    print(f"{name}: Precision={prec:.2f}, Recall={rec:.2f}, F1-Score={f1:.2f}")

# === CONFUSION MATRIX ===
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# === TOP CONFUSIONS (Mistakes) ===
print("\n=== Top Confusions (Mistakes) ===")
for actual_class in range(3):
    for predicted_class in range(3):
        if actual_class != predicted_class:
            print(f"Actual {label_map[actual_class]} predicted as {label_map[predicted_class]}: {cm[actual_class][predicted_class]} times")

# === FALSE POSITIVE AND FALSE NEGATIVE RATES ===
fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)

print("\n=== False Positives and False Negatives ===")
for i, label in label_map.items():
    print(f"{label}: False Positives={fp[i]}, False Negatives={fn[i]}")

# === FULL CLASS BIAS CHECK ===
print("\n=== Class Prediction Distribution (Bias Check) ===")
total_preds = len(y_pred)
for label, name in label_map.items():
    class_preds = sum(y_pred == label)
    print(f"{name} Predictions: {class_preds} out of {total_preds} total ({class_preds/total_preds*100:.2f}%)")


# === ADVERSARIAL TEST CASES ===
adversarial_samples = [
    "I'm proud to be a queer woman",   # should be clean
    "Go back to your country!",        # should be hate
    "That's sick dude",                # sarcasm
    "You're acting like a total Karen",# borderline offensive
    "Shut up, idiot",                  # offensive
    "Love wins ❤️",                    # should be clean
    "You don't belong here",           # hate
]

# Define true labels manually for your adversarial examples
true_labels = [2, 0, 2, 1, 1, 2, 0]  # 2=Neither, 0=Hate, 1=Offensive, etc.

# Transform and predict
X_adv = vectorizer.transform(adversarial_samples)
y_adv_pred = model.predict(X_adv)

# Print with both predicted and true labels
print("\n=== Adversarial Test Cases (with True Labels) ===")
for text, pred, true in zip(adversarial_samples, y_adv_pred, true_labels):
    print(f'"{text}" → Predicted: {label_map[pred]} | Should be: {label_map[true]}')
