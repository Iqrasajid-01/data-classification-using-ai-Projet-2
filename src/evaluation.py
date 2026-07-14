import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import os


def plot_confusion_matrix(cm, target_names, save_path="outputs/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def plot_elbow_curve(train_scores, test_scores, save_path="outputs/elbow_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    k_values = range(1, len(train_scores) + 1)
    plt.plot(k_values, train_scores, marker="o", label="Training Accuracy")
    plt.plot(k_values, test_scores, marker="s", label="Test Accuracy")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("Elbow Method — Finding Optimal K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Elbow curve plot saved to: {save_path}")


def evaluate(y_test, y_pred, target_names):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (Trustworthiness — Spam Filters)")
    print(f"Recall:    {recall:.4f} (Sensitivity — Medical Diagnosis)")
    print(f"F1 Score:  {f1:.4f} (Harmonic Mean of Precision & Recall)")
    print()

    print("Confusion Matrix:")
    print(f"{'':>12} {'Pred Setosa':>12} {'Pred Versi':>12} {'Pred Virg':>12}")
    print(f"{'Actual Setosa':>12} {cm[0,0]:>12} {cm[0,1]:>12} {cm[0,2]:>12}")
    print(f"{'Actual Versi':>12} {cm[1,0]:>12} {cm[1,1]:>12} {cm[1,2]:>12}")
    print(f"{'Actual Virg':>12} {cm[2,0]:>12} {cm[2,1]:>12} {cm[2,2]:>12}")
    print()

    print(f"Diagnostic Breakdown (Confusion Matrix):")
    print(f"  TP (True Positive):  Correctly predicted")
    print(f"  FP (False Positive): False Alarm (Type I Error)")
    print(f"  FN (False Negative): Missed Detection (Type II Error)")
    print(f"  TN (True Negative):  Correctly rejected")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print()

    print("OUTPUT VALIDATION NOTE:")
    print("  The Iris dataset is balanced (50 samples per class),")
    print("  so accuracy is a reliable metric here.")
    print("  In imbalanced data, accuracy can be misleading.")
    print("  Always look deeper — use Precision, Recall, F1 Score,")
    print("  and Confusion Matrix for a complete picture.")
    print("=" * 60)

    plot_confusion_matrix(cm, target_names)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm}
