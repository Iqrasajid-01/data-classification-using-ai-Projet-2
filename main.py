from src.data_loader import load_iris_data, describe_dataset
from src.preprocessing import split_and_scale
from src.model import train_knn, train_logistic_regression, predict, elbow_method, compare_algorithms
from src.evaluation import evaluate, plot_elbow_curve


def main():
    print()
    print("=" * 60)
    print("PROJECT 2: DATA CLASSIFICATION USING AI")
    print("Industrial Training Kit | Batch 2026 | DecodeLabs")
    print("=" * 60)
    print()

    X, y, target_names = load_iris_data()
    describe_dataset(X, y, target_names)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.2)

    optimal_k, train_scores, test_scores = elbow_method(X_train, y_train, X_test, y_test, max_k=30)
    plot_elbow_curve(train_scores, test_scores)

    print("NOTE: Although K=1 gives the highest test accuracy on this small dataset,")
    print("      K=5 is preferred as the default to avoid overfitting (noise).")
    print("      The PDF specifies K=5 with Majority Vote as the standard configuration.\n")

    model = train_knn(X_train, y_train, n_neighbors=5)

    y_pred = predict(model, X_test)

    results = evaluate(y_test, y_pred, target_names)

    compare_algorithms(X_train, y_train, X_test, y_test, target_names)

    print()

if __name__ == "__main__":
    main()
