from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_segment_classifier.joblib"

FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]

SEGMENT_DESCRIPTIONS = {
    "High diverse spending": (
        "This customer has relatively high spending across multiple "
        "product categories."
    ),
    "Grocery & household goods": (
        "This customer tends to spend more on grocery products "
        "and household supplies."
    ),
    "Traditional fresh food": (
        "This customer tends to spend more on fresh food products."
    ),
}

DEFAULT_VALUES = {
    "Fresh": 12000,
    "Milk": 3000,
    "Grocery": 4000,
    "Frozen": 1500,
    "Detergents_Paper": 800,
    "Delicassen": 600,
}


# Page configuration
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    page_icon="📊",
    layout="centered",
)


# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Title
st.title("📊 Wholesale Customer Segmentation")

st.write(
    "Enter the customer's annual spending across different product "
    "categories to predict their customer segment."
)


# Check model availability
if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run "
        "`02_knn_classifier.py` first and ensure that "
        "`knn_segment_classifier.joblib` is in the project directory."
    )
    st.stop()

model = load_model()


# Input fields
st.subheader("Customer Information")

st.write(
    "Enter the customer's annual spending in each product category."
)


with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        fresh = st.number_input(
            "Fresh",
            min_value=0,
            value=DEFAULT_VALUES["Fresh"],
            step=100,
        )

        milk = st.number_input(
            "Milk",
            min_value=0,
            value=DEFAULT_VALUES["Milk"],
            step=100,
        )

        grocery = st.number_input(
            "Grocery",
            min_value=0,
            value=DEFAULT_VALUES["Grocery"],
            step=100,
        )

    with col2:
        frozen = st.number_input(
            "Frozen",
            min_value=0,
            value=DEFAULT_VALUES["Frozen"],
            step=100,
        )

        detergents_paper = st.number_input(
            "Detergents & Paper",
            min_value=0,
            value=DEFAULT_VALUES["Detergents_Paper"],
            step=100,
        )

        delicassen = st.number_input(
            "Delicassen",
            min_value=0,
            value=DEFAULT_VALUES["Delicassen"],
            step=100,
        )

    submitted = st.form_submit_button(
        "Predict Customer Segment",
        type="primary",
        use_container_width=True,
    )


# Prediction
if submitted:

    customer = pd.DataFrame(
        [
            {
                "Fresh": fresh,
                "Milk": milk,
                "Grocery": grocery,
                "Frozen": frozen,
                "Detergents_Paper": detergents_paper,
                "Delicassen": delicassen,
            }
        ],
        columns=FEATURES,
    )

    # Make prediction
    prediction = model.predict(customer)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(customer)[0]
    classes = model.named_steps["knn"].classes_

    confidence = max(probabilities)

    # Display prediction result
    st.divider()
    st.subheader("Prediction Result")

    st.success(f"Predicted Customer Segment: **{prediction}**")

    st.write(
        SEGMENT_DESCRIPTIONS.get(
            prediction,
            "No description is available for this customer segment.",
        )
    )

    st.metric(
        label="Model Confidence",
        value=f"{confidence:.2%}",
        help=(
            "The estimated probability assigned by the model to "
            "the predicted segment. This does not guarantee "
            "that the prediction is correct."
        ),
    )

    # Probability by segment
    st.subheader("Probability by Segment")

    probability_df = pd.DataFrame(
        {
            "Segment": classes,
            "Probability": probabilities,
        }
    ).sort_values(
        by="Probability",
        ascending=False,
    )

    st.bar_chart(
        probability_df.set_index("Segment"),
        y="Probability",
    )

    # Customer spending profile
    st.subheader("Customer Spending Profile")

    spending_df = pd.DataFrame(
        {
            "Category": FEATURES,
            "Annual Spending": [
                fresh,
                milk,
                grocery,
                frozen,
                detergents_paper,
                delicassen,
            ],
        }
    )

    st.bar_chart(
        spending_df.set_index("Category"),
        y="Annual Spending",
    )

    # Disclaimer
    st.info(
        "This application is for educational and demonstration "
        "purposes. Predictions are based on historical customer "
        "spending and customer segments generated using K-Means."
    )
