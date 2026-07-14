# Data Classification Using AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

**Project 2** — Industrial Training Kit | Batch 2026 | [DecodeLabs](https://decodelabs.com)

A supervised learning pipeline for classifying the classic **Iris dataset** using K-Nearest Neighbors (KNN) and Logistic Regression, with automated evaluation, visualization, and model comparison.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Results](#results)
- [License](#license)

---

## Overview

This project implements a complete machine learning workflow:

- **Dataset:** [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (150 samples, 4 features, 3 classes)
- **Algorithms:**
  - K-Nearest Neighbors (K=5, Majority Vote)
  - Logistic Regression (for comparison)
- **Key techniques:**
  - Train/Test split (80/20) with stratification
  - Feature scaling via `StandardScaler`
  - Elbow method to find optimal K
  - Confusion matrix, precision, recall, F1-score

---

## Pipeline

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Load  │ →  │ Preprocessing│ →  │ Model Train  │ →  │ Evaluation   │
│ (Iris)     │    │ Split+Scale  │    │ KNN / LogReg │    │ Metrics+Plots│
└────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ↓
                                     ┌────────────────┐
                                     │  Comparison    │
                                     │  Best Algo     │
                                     └────────────────┘
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/project-2.git
cd project-2

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy` — Numerical computation
- `pandas` — Data manipulation
- `scikit-learn` — ML algorithms & metrics
- `matplotlib` — Plotting
- `seaborn` — Statistical visualizations

---

## Usage

Run the full pipeline with a single command:

```bash
python main.py
```

The script will:

1. Load and describe the Iris dataset
2. Split data (80% train, 20% test) and apply `StandardScaler`
3. Run the Elbow Method for K values 1–30
4. Train a KNN classifier with K=5
5. Evaluate the model (accuracy, precision, recall, F1-score)
6. Generate a confusion matrix and elbow curve plot
7. Compare KNN against Logistic Regression

### Output Plots

| Plot | File |
|------|------|
| Confusion Matrix | `outputs/confusion_matrix.png` |
| Elbow Curve | `outputs/elbow_curve.png` |

---

## Project Structure

```
project-2/
├── main.py                  # Entry point — orchestrates the pipeline
├── requirements.txt         # Python dependencies
├── outputs/
│   ├── confusion_matrix.png # Confusion matrix visualization
│   └── elbow_curve.png      # Elbow method plot
└── src/
    ├── __init__.py
    ├── data_loader.py       # Load & describe Iris dataset
    ├── preprocessing.py     # Train/test split & feature scaling
    ├── model.py             # KNN, Logistic Regression, Elbow, comparison
    └── evaluation.py        # Metrics, confusion matrix, plots
```

---

## Results

### Classification Report (KNN, K=5)

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Setosa      | 1.00      | 1.00   | 1.00     | 10      |
| Versicolor  | 1.00      | 1.00   | 1.00     | 10      |
| Virginica   | 1.00      | 1.00   | 1.00     | 10      |

**Overall Accuracy:** 1.0000 (100%)

### Elbow Method

The Elbow Method evaluates K values from 1 to 30. While K=1 often yields the highest test accuracy on this small dataset, **K=5** is chosen as the default to avoid overfitting and maintain generalization.

### Algorithm Comparison

| Algorithm            | Accuracy |
|----------------------|----------|
| KNN (K=5)            | 1.0000   |
| Logistic Regression  | 1.0000   |

Both algorithms achieve perfect accuracy on the Iris dataset's 80/20 split, demonstrating the dataset's linear separability.

---

## License

This project is part of the **DecodeLabs Industrial Training Kit**. For educational use only.
