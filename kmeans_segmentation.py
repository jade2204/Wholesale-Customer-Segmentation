#Discover customer segments using K-Means clustering.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Config
DATA_PATH = Path("wholesale_customers.csv")
OUTPUT_PATH = Path("customers_segmented.csv")
RANDOM_STATE = 42
BUSINESS_K = 3  # chosen after reviewing the k report below

# Channel and Region are intentionally NOT used to build labels.
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def build_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )


def evaluate_k_values(X_scaled: np.ndarray, k_range=range(2, 9)):
    """Fit K-Means for each k and collect quality metrics.

    - inertia: within-cluster sum of squares (lower is better, use the elbow).
    - silhouette: in [-1, 1]; closer to 1 means points sit well inside their
      own cluster and far from others.
    - calinski_harabasz: between-cluster vs. within-cluster dispersion
      (higher is better).
    """
    rows, models = [], {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        models[k] = model
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows), models


def build_label_map(centers_z: pd.DataFrame) -> dict:
    assert len(centers_z) == 3 #Label mapping is written for exactly 3 clusters.

    remaining = set(centers_z.index)

    # Cluster with the highest average (standardized) spend across all categories
    high_spend_id = centers_z.mean(axis=1).idxmax()
    remaining.remove(high_spend_id)

    # Among the rest: heavy on grocery/detergents, light on fresh/frozen
    retail_score = (
        centers_z["Grocery"]
        + centers_z["Detergents_Paper"]
        - centers_z["Fresh"]
        - centers_z["Frozen"]
    )
    retail_id = retail_score.loc[list(remaining)].idxmax()
    remaining.remove(retail_id)

    fresh_food_id = remaining.pop()

    return {
        int(high_spend_id): "High diverse spending",
        int(retail_id): "Grocery & household goods",
        int(fresh_food_id): "Traditional fresh food",
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Data quality check
    quality_report = pd.DataFrame(
        {
            "dtype": df[FEATURES].dtypes.astype(str),
            "missing": df[FEATURES].isna().sum(),
            "unique": df[FEATURES].nunique(),
        }
    )
    print("=== Data quality ===")
    print(quality_report, end="\n\n")
    print("=== Summary statistics ===")
    print(df[FEATURES].describe().T.round(1), end="\n\n")

    # Preprocess and compare different k values
    preprocessor = build_preprocessor()
    X_scaled = preprocessor.fit_transform(df[FEATURES])

    k_report, models = evaluate_k_values(X_scaled)
    print("=== K selection report ===")
    print(k_report.round(3), end="\n\n")

    # Use the chosen k
    kmeans = models[BUSINESS_K]
    df_segmented = df.copy()
    df_segmented["cluster_id"] = kmeans.labels_

    centers_z = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)
    centers_z.index.name = "cluster_id"
    print("=== Cluster centers (z-scores) ===")
    print(centers_z.round(2), end="\n\n")

    # Human-in-the-loop naming (automated here via centroid rules)
    label_map = build_label_map(centers_z)
    df_segmented["segment_label"] = df_segmented["cluster_id"].map(label_map)
    print("=== Label map ===")
    print(label_map, end="\n\n")

    # Segment profiles
    profiles = (
        df_segmented.groupby(["cluster_id", "segment_label"])[FEATURES]
        .median()
        .round(0)
    )
    counts = df_segmented["segment_label"].value_counts().rename("customers")
    print("=== Segment profiles (median spend) ===")
    print(profiles, end="\n\n")
    print("=== Segment sizes ===")
    print(counts.to_frame(), end="\n\n")

    # Export for the KNN step
    df_segmented.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} for the KNN script.")


if __name__ == "__main__":
    main()