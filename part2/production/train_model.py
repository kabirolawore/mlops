import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import mlflow
import mlflow.sklearn  # so we can use autolog if we want

# -------------------------------------------------------------------
# Parse arguments (Azure ML will pass the dataset path as --trainingdata)
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainingdata",
    type=str,
    required=True,
    help="Dataset for training"
)
args = parser.parse_args()

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv(args.trainingdata)
print("Head of dataset:")
print(df.head())

# Pick out the columns we want to use as inputs
X = df[["sepalLength", "sepalWidth"]].values
Y = df["type"].values

# Check basic info
print("Class distribution:", np.unique(Y, return_counts=True))

# Split the data and keep 20% back for testing later
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y
)
print("Train length", len(X_train))
print("Test length", len(X_test))

# -------------------------------------------------------------------
# OPTIONAL: automatic logging (for simple models)
# This single line will make MLflow log params, metrics and the model.
# Commented out so we can demonstrate manual logging below.
# -------------------------------------------------------------------
# mlflow.sklearn.autolog()

# -------------------------------------------------------------------
# Train model
# -------------------------------------------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# -------------------------------------------------------------------
# Evaluate model
# -------------------------------------------------------------------
testPredictions = model.predict(X_test)
acc = np.mean(testPredictions == Y_test)
print("Accuracy", acc)

# For ROC/AUC we need probability scores for each class
Y_scores = model.predict_proba(X_test)

# Overall multi-class AUC (one-vs-rest)
macro_auc = roc_auc_score(Y_test, Y_scores, multi_class="ovr")
print("Macro AUC", macro_auc)

# -------------------------------------------------------------------
# Create ROC curve figure
# -------------------------------------------------------------------
fig = plt.figure(figsize=(6, 4))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

# Plot the 50% line
plt.plot([0, 1], [0, 1], "k--")

class_names = df["type"].unique()
per_class_auc = {}

# Plot ROC curve for the different classes
for idx, className in enumerate(class_names):
    # Binary labels: 1 for current class, 0 for others
    fpr, tpr, thresholds = roc_curve(Y_test == className, Y_scores[:, idx])
    plt.plot(fpr, tpr, label=f"ROC for {className}")

    # Per-class AUC
    auc_value = roc_auc_score(Y_test == className, Y_scores[:, idx])
    per_class_auc[className] = auc_value

plt.legend()

# -------------------------------------------------------------------
# Log metrics and artifacts with MLflow (Task 1)
# -------------------------------------------------------------------
# NOTE: when this code runs inside an Azure ML job created with `az ml job create`,
# an active MLflow run is already started for you. So you can directly call
# mlflow.log_metric and mlflow.log_figure.

# Log core metrics
mlflow.log_metric("test_accuracy", float(acc))
mlflow.log_metric("macro_ovr_auc", float(macro_auc))

# Log per-class AUC metrics (optional but nice)
for className, auc_value in per_class_auc.items():
    metric_name = f"auc_{className}"
    mlflow.log_metric(metric_name, float(auc_value))

# Log the ROC curve as an artifact (image file)
mlflow.log_figure(fig, "roc_curve.png")

# Close the figure to free memory
plt.close(fig)

print("Metrics and ROC curve logged to MLflow/Azure ML.")