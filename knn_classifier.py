"""
Train a KNN classifier to predict the segment label for new customers.

Usage:
    Run 01_kmeans_segmentation.py first, then:
        python 02_knn_classifier.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_PATH = Path("customers_segmented.csv")
MODEL_PATH = Path("knn_segment_classifier.joblib")
RANDOM_STATE = 42

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
TARGET = "segment_label"

def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_columns = set(FEATURES + [TARGET]) - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert df[FEATURES + [TARGET]].notna().all().all() #Data contains missing values
    assert df[TARGET].nunique() >= 2 #Need at least two segments

    print(f"Received {len(df)} customers from the K-Means step.")
    print(df[TARGET].value_counts().rename("customers").to_frame(), end="\n\n")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )


def main() -> None:
    df = load_and_validate(DATA_PATH)

    # Hold out 20% as a test set; stratify keeps segment proportions similar
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    # Hyperparameter search
    param_grid = {
        "knn__n_neighbors": list(range(3, 26, 2)),  # how many neighbors vote
        "knn__weights": ["uniform", "distance"],  # closer neighbors count more?
        "knn__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV Macro-F1: {search.best_score_:.3f}\n")

    # Evaluate on the held-out test set
    best_knn = search.best_estimator_
    y_pred = best_knn.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test Macro-F1: {f1_score(y_test, y_pred, average='macro'):.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Refit the best configuration on ALL data and save it
    final_knn = clone(best_knn).fit(df[FEATURES], df[TARGET])
    joblib.dump(final_knn, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Example: predict a segment for a new customer
    new_customer = pd.DataFrame(
        [{"Fresh": 12000, "Milk": 3000, "Grocery": 4000,
          "Frozen": 1500, "Detergents_Paper": 800, "Delicassen": 600}]
    )
    print(f"Example prediction: {final_knn.predict(new_customer)[0]}")


if __name__ == "__main__":
    main()