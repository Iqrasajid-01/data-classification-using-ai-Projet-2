from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    print(f"Train set size: {X_train_scaled.shape[0]} samples ({len(X_train_scaled)/len(X)*100:.0f}%)")
    print(f"Test set size: {X_test_scaled.shape[0]} samples ({len(X_test_scaled)/len(X)*100:.0f}%)")
    print("Scaling: StandardScaler (Mean=0, Variance=1)")
    print("Shuffle: Applied before split to remove order bias")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
