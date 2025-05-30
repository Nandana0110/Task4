# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, 
    recall_score, roc_auc_score, roc_curve
)
from scipy.special import expit  # for sigmoid

df = pd.read_csv(r"C:\Users\AL SharQ\Downloads\intern\data.csv") 

df.drop(columns=["id", "Unnamed: 32"], inplace=True)
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  # M = 1, B = 0

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and predicted probabilities
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Threshold tuning
print("\nThreshold tuning analysis:")
for threshold in [0.3, 0.5, 0.7]:
    custom_pred = (y_proba >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, custom_pred))
    print("Precision:", precision_score(y_test, custom_pred))
    print("Recall:", recall_score(y_test, custom_pred))

# Plot the sigmoid function
x_vals = np.linspace(-10, 10, 200)
y_vals = expit(x_vals)  # sigmoid
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals)
plt.title("Sigmoid Function")
plt.xlabel("Input (z)")
plt.ylabel("Output (sigmoid(z))")
plt.grid(True)
plt.axvline(0, color='red', linestyle='--', label='Threshold = 0.5')
plt.legend()
plt.show()
