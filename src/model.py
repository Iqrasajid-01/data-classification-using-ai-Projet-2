import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Algorithm: K-Nearest Neighbors (KNN)")
    print(f"K value: {n_neighbors}")
    print(f"Strategy: Majority Vote")
    print(f"Principle: Similar things exist in close proximity")
    print("=" * 60)

    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


def elbow_method(X_train, y_train, X_test, y_test, max_k=30):
    print("=" * 60)
    print("ELBOW METHOD — FINDING OPTIMAL K")
    print("=" * 60)
    print(f"{'K':>3} | {'Train Acc':>10} | {'Test Acc':>10} | {'Notes':>20}")
    print("-" * 50)

    train_scores = []
    test_scores = []

    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        train_scores.append(train_acc)
        test_scores.append(test_acc)

        note = ""
        if k == 1:
            note = "Noise / Overfitting risk"
        elif k == 5:
            note = "Default K (Recommended)"
        elif k == max_k:
            note = "Generic / Underfitting risk"
        if k in [1, 3, 5, 7, 10, 15, 20, 25, 30]:
            print(f"{k:>3} | {train_acc:>10.4f} | {test_acc:>10.4f} | {note:>20}")

    optimal_k = np.argmax(test_scores) + 1
    print("-" * 50)
    print(f"Optimal K (by Elbow Method): {optimal_k} (highest test accuracy)")
    print("=" * 60)

    return optimal_k, train_scores, test_scores


def compare_algorithms(X_train, y_train, X_test, y_test, target_names):
    from sklearn.metrics import classification_report

    print("=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)

    models = {
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc, "predictions": y_pred}
        print(f"{name:>25}: Accuracy = {acc:.4f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest performing algorithm: {best_name}")
    print("\nDetailed report for best model:")
    print(classification_report(y_test, results[best_name]["predictions"], target_names=target_names))
    print("=" * 60)

    return results
