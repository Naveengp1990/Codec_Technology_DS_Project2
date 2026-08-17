# ============================================================
# app.py
# STREAMLIT CAR LOAN CREDIT RISK PREDICTION APP
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Car Loan Credit Risk Prediction",

    page_icon="🚗",

    layout="wide"
)


# ============================================================
# ARTIFACT PATHS
# ============================================================

ARTIFACT_DIR = "artifacts"

PREPROCESSOR_FILE = os.path.join(
    ARTIFACT_DIR,
    "preprocessor.pkl"
)

MODEL_FILE = os.path.join(
    ARTIFACT_DIR,
    "model.pkl"
)

FEATURE_CONFIG_FILE = os.path.join(
    ARTIFACT_DIR,
    "feature_config.pkl"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    feature_config = joblib.load(
        FEATURE_CONFIG_FILE
    )

    return (
        preprocessor,
        model,
        feature_config
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    (
        preprocessor,
        model,
        feature_config
    ) = load_artifacts()

except Exception as e:

    st.error(
        "Unable to load model artifacts."
    )

    st.exception(e)

    st.stop()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🚗 Car Loan Credit Risk Prediction"
)

st.markdown(
    """
    ### Credit Risk Prediction System

    Enter the applicant and loan information below to
    predict the probability of the selected customer
    belonging to the **Credit Risk Class 1**.
    """
)


st.divider()


# ============================================================
# FEATURE INFORMATION
# ============================================================

numeric_columns = feature_config[
    "numeric_columns"
]

categorical_columns = feature_config[
    "categorical_columns"
]

all_features = feature_config[
    "all_features"
]


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = {}


# ============================================================
# NUMERICAL INPUTS
# ============================================================

if len(numeric_columns) > 0:

    st.subheader(
        "📊 Numerical Information"
    )

    numeric_cols = st.columns(2)

    for index, column in enumerate(
        numeric_columns
    ):

        with numeric_cols[index % 2]:

            input_data[column] = st.number_input(

                label=column.replace(
                    "_",
                    " "
                ).title(),

                value=0.0,

                step=1.0,

                format="%.4f"
            )


# ============================================================
# CATEGORICAL INPUTS
# ============================================================

if len(categorical_columns) > 0:

    st.subheader(
        "📝 Categorical Information"
    )

    categorical_cols = st.columns(2)

    for index, column in enumerate(
        categorical_columns
    ):

        # Try to obtain categories learned
        # by the OneHotEncoder

        categories = None

        try:

            cat_pipeline = (
                preprocessor
                .named_transformers_["cat"]
            )

            encoder = (
                cat_pipeline
                .named_steps["encoder"]
            )

            category_index = (
                categorical_columns
                .index(column)
            )

            categories = (
                encoder
                .categories_[category_index]
                .tolist()
            )

        except Exception:

            categories = None


        if not categories:

            categories = ["Unknown"]


        with categorical_cols[index % 2]:

            input_data[column] = st.selectbox(

                label=column.replace(
                    "_",
                    " "
                ).title(),

                options=categories
            )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Credit Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data],
            columns=all_features
        )


        # ----------------------------------------------------
        # PREPROCESS INPUT
        # ----------------------------------------------------

        transformed_input = (
            preprocessor.transform(
                input_df
            )
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            transformed_input
        )[0]


        probability = (
            model.predict_proba(
                transformed_input
            )[0]
        )


        probability_class_0 = probability[0]

        probability_class_1 = probability[1]


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        st.subheader(
            "🎯 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "⚠️ Higher Credit Risk — Class 1"
            )

        else:

            st.success(
                "✅ Lower Credit Risk — Class 0"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Class 0 Probability",
                f"{probability_class_0:.2%}"
            )


        with col2:

            st.metric(
                "Class 1 Probability",
                f"{probability_class_1:.2%}"
            )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.subheader(
            "Risk Probability"
        )

        probability_df = pd.DataFrame(

            {
                "Class 0": [
                    probability_class_0
                ],

                "Class 1": [
                    probability_class_1
                ]
            }
        )

        st.bar_chart(
            probability_df
        )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander(
            "View Input Information"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Model Information"
    )

    st.write(
        "Model: AdaBoostClassifier"
    )

    st.write(
        f"Features: {len(all_features)}"
    )

    st.write(
        f"Numerical: {len(numeric_columns)}"
    )

    st.write(
        f"Categorical: {len(categorical_columns)}"
    )

    st.divider()

    st.subheader(
        "Model Performance"
    )

    st.write(
        f"CV F1: {feature_config['cv_f1']:.4f}"
    )

    st.write(
        f"Test Accuracy: "
        f"{feature_config['test_accuracy']:.4f}"
    )

    st.write(
        f"Test Precision: "
        f"{feature_config['test_precision']:.4f}"
    )

    st.write(
        f"Test Recall: "
        f"{feature_config['test_recall']:.4f}"
    )

    st.write(
        f"Test F1: "
        f"{feature_config['test_f1']:.4f}"
    )

    st.write(
        f"Test ROC-AUC: "
        f"{feature_config['test_roc_auc']:.4f}"
    )

    st.divider()

    st.caption(
        "Car Loan Credit Risk Prediction"
    )
