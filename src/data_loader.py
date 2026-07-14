import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    target_names = iris.target_names
    return X, y, target_names


def describe_dataset(X, y, target_names):
    print("=" * 60)
    print("IRIS DATASET - EXPLORATION")
    print("=" * 60)
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")
    print(f"Number of classes: {len(target_names)}")
    print(f"Class names: {list(target_names)}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")
    print(f"\nFeature statistics:\n{X.describe()}")
    print("=" * 60)
