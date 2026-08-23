# Breast Cancer Prediction
# Dataset: Breast Cancer Wisconsin Dataset
# Uses sklearn's built-in load_breast_cancer() dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset Shape:", X.shape)
print("\nFeature Names:")
print(X.columns.tolist())

print("\nTarget Names:")
print(data.target_names)

# 2. Basic exploration
print("\nFirst 5 records:")
print(X.head())

print("\nSummary Statistics:")
print(X.describe())

print("\nMissing Values:")
print(X.isnull().sum().sum())

# 3. Target variable distribution
print("\nTarget Distribution:")
print(y.value_counts())

# 4. Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(X.corr(), cmap="coolwarm", linewidths=0.2)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Build machine learning model
model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train_scaled, y_train)

# 8. Prediction
y_pred = model.predict(X_test_scaled)

# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----- Model Evaluation -----")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 11. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

# 12. Observation
print("\n----- Observation -----")
print("The Logistic Regression model is trained to classify tumors as")
print("malignant or benign using the 30 medical features.")
print("Accuracy, Precision, Recall and F1-Score are used to evaluate the model.")
