# Wholesale Customer Segmentation: K-Means + KNN + Streamlit

An end-to-end machine learning project that first discovers customer segments from unlabeled data, then trains a classifier to assign new customers to those segments, and finally serves it through a Streamlit web app.

```
Unlabeled data -> log + scale -> K-Means -> human review & naming
              -> labeled data -> train/test split -> tuned KNN
              -> saved model -> Streamlit app (single + batch prediction)
```

## Why two algorithms?

KNN needs labeled neighbors, but the raw data has no labels. So the workflow is:

1. **K-Means** finds a starting structure when no labels exist.
2. **A human** checks the clusters and gives them business meaning.
3. **KNN** uses the reviewed labels to classify new customers quickly.

> **Note:** The KNN scores measure how well the model reproduces the handed-over labels. They do not prove the segments are correct from a business standpoint.

## Dataset

[Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) from the UCI Machine Learning Repository: 440 customers and their annual spending (in monetary units) on six product categories.

| Feature | Description |
|---|---|
| `Fresh` | Fresh products |
| `Milk` | Dairy products |
| `Grocery` | Grocery products |
| `Frozen` | Frozen products |
| `Detergents_Paper` | Detergents and paper products |
| `Delicassen` | Delicatessen products |

The `Channel` and `Region` columns exist in the original file but are **not** used to create labels.

## Project structure

```
.
├── 01_kmeans_segmentation.py      # Step 1: discover and name segments
├── 02_knn_classifier.py           # Step 2: train and evaluate KNN
├── app.py                         # Step 3: Streamlit web app
├── requirements.txt               # Dependencies (local + deployment)
├── wholesale_customers.csv        # Input data (you provide this)
├── customers_segmented.csv        # Output of step 1 (generated)
├── knn_segment_classifier.joblib  # Output of step 2 (generated)
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Download the dataset from the link above and save it as `wholesale_customers.csv` in the project folder.

## Usage

Run the scripts in order, from the project folder:

```bash
python 01_kmeans_segmentation.py
python 02_knn_classifier.py
streamlit run app.py
```

On macOS/Linux, use `python3` if `python` is not found. The app opens at `http://localhost:8501`.

## What each step does

### Step 1: K-Means segmentation (`01_kmeans_segmentation.py`)

- Checks data quality (types, missing values, summary statistics).
- Applies `log1p` and `StandardScaler` to handle skewed spending values.
- Tries `k = 2..8` and reports inertia, silhouette score, and Calinski-Harabasz score.
- Fits the chosen `k` (`BUSINESS_K = 3`) and names the clusters from their centroids:
  - **High diverse spending**
  - **Grocery & household goods**
  - **Traditional fresh food**
- Exports `customers_segmented.csv`.

### Step 2: KNN classification (`02_knn_classifier.py`)

- Validates the file produced by step 1.
- Splits the data 80/20 with stratification.
- Tunes `n_neighbors`, `weights`, and distance metric (`p`) with `GridSearchCV`, optimizing Macro-F1 under 5-fold stratified cross-validation.
- Preprocessing lives inside the `Pipeline`, so each fit sees only training data (no test-set leakage).
- Reports Accuracy, Macro-F1, and a classification report on the test set.
- Refits the best model on all data and saves it to `knn_segment_classifier.joblib`.

### Step 3: Streamlit app (`app.py`)

The app loads the saved model and offers two tabs:

- **Single customer:** enter spending for the six categories and get the predicted segment plus a probability chart.
- **Batch (CSV):** upload a CSV containing the six feature columns, view predictions, and download them as `predictions.csv`. Input is validated for missing columns, missing values, and negative spending.

## Deploying to Streamlit Community Cloud

1. Push these files to a GitHub repository: `app.py`, `requirements.txt`, and `knn_segment_classifier.joblib`.
2. Go to [share.streamlit.io](https://share.streamlit.io), select the repository, and set the main file path to `app.py`.
3. Click **Deploy**.

> **Version note:** The `.joblib` file depends on the scikit-learn version used for training. If the deployed app fails to load the model, pin that version in `requirements.txt` (check it with `pip freeze | grep scikit-learn`), for example `scikit-learn==1.5.2`.

## Using the saved model in your own code

```python
import joblib
import pandas as pd

model = joblib.load("knn_segment_classifier.joblib")

new_customer = pd.DataFrame([{
    "Fresh": 12000, "Milk": 3000, "Grocery": 4000,
    "Frozen": 1500, "Detergents_Paper": 800, "Delicassen": 600,
}])

print(model.predict(new_customer)[0])
```

## Configuration

| Setting | File | Default | Notes |
|---|---|---|---|
| `BUSINESS_K` | `01_kmeans_segmentation.py` | `3` | Number of clusters. The label-naming logic only supports 3. |
| `RANDOM_STATE` | steps 1 and 2 | `42` | For reproducible results. |
| `param_grid` | `02_knn_classifier.py` | see file | KNN search space. |
| `DEFAULTS` | `app.py` | see file | Default values shown in the input form. |

## Limitations

- Cluster naming is rule-based on centroids and assumes exactly 3 clusters. Changing `BUSINESS_K` requires updating `build_label_map`.
- K-Means labels are a starting point, not ground truth. Review them with domain experts before using them for business decisions.
- The dataset is small (440 rows), so metrics can vary noticeably between splits.

## License / Data attribution

Dataset: Cardoso, Margarida. *Wholesale customers.* UCI Machine Learning Repository. Check the UCI page for its license terms.