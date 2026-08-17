```python
# ============================================================
# local.py
# CAR LOAN CREDIT RISK MODEL
# Training + Hyperparameter Tuning + Model Saving
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "Car_Loan_Credit.csv"

ARTIFACT_DIR = "artifacts"

TARGET = "loan_status"

RANDOM_STATE = 33

TEST_SIZE = 0.27


# ============================================================
# CREATE ARTIFACT DIRECTORY
# ============================================================

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = pd.read_csv(DATA_FILE)

print("Original shape:", data.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

data.columns = (
    data.columns
    .str.lower()
    .str.strip()
    .str.replace("person_", "", regex=False)
    .str.replace(" ", "_", regex=False)
)

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 3. CHECK TARGET
# ============================================================

if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' not found.\n"
        f"Available columns:\n{data.columns.tolist()}"
    )


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

duplicate_count = data.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

data = data.drop_duplicates().reset_index(drop=True)

print("Shape after removing duplicates:", data.shape)


# ============================================================
# 5. REMOVE MISSING TARGET VALUES
# ============================================================

missing_target = data[TARGET].isnull().sum()

print("\nMissing target values:", missing_target)

if missing_target > 0:

    data = data.dropna(
        subset=[TARGET]
    ).reset_index(drop=True)


# ============================================================
# 6. SEPARATE X AND y
# ============================================================

X = data.drop(
    columns=[TARGET]
).copy()

y = data[TARGET].copy()


# ============================================================
# 7. ENCODE TARGET
# ============================================================

target_encoder = None

if y.dtype == "object":

    y = y.astype(str).str.strip()

    unique_values = set(y.unique())

    binary_mapping = {
        "Y": 1,
        "N": 0,
        "Yes": 1,
        "No": 0,
        "YES": 1,
        "NO": 0,
        "yes": 1,
        "no": 0,
        "Approved": 1,
        "Rejected": 0,
        "approved": 1,
        "rejected": 0
    }

    if unique_values.issubset(
        set(binary_mapping.keys())
    ):

        y = y.map(binary_mapping)

        target_encoder = {
            "type": "mapping",
            "mapping": binary_mapping
        }

    else:

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

        target_encoder = {
            "type": "label_encoder",
            "encoder": label_encoder
        }


# Convert target to integer
y = pd.Series(y).astype(int)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    stratify=y,

    random_state=RANDOM_STATE
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 9. IDENTIFY FEATURE TYPES
# ============================================================

numeric_columns = (
    X_train
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)

categorical_columns = (
    X_train
    .select_dtypes(exclude=np.number)
    .columns
    .tolist()
)


print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 10. NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),

        (
            "scaler",
            RobustScaler()
        )
    ]
)


# ============================================================
# 11. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 12. COLUMN TRANSFORMER
# ============================================================

preprocess = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_pipeline,
            numeric_columns
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ],

    remainder="drop"
)


# ============================================================
# 13. ADA BOOST MODEL
# ============================================================

adaboost_model = AdaBoostClassifier(
    random_state=RANDOM_STATE
)


# ============================================================
# 14. COMPLETE PIPELINE
# ============================================================

ada_pipe = Pipeline(

    steps=[

        (
            "preprocess",
            preprocess
        ),

        (
            "model",
            adaboost_model
        )
    ]
)


# ============================================================
# 15. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# 16. HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "model__n_estimators": [
        50,
        100,
        150,
        200,
        300,
        400
    ],

    "model__learning_rate": [
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0
    ]
}


# ============================================================
# 17. GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("STARTING GRID SEARCH")
print("=" * 70)

grid_search = GridSearchCV(

    estimator=ada_pipe,

    param_grid=param_grid,

    scoring="f1",

    cv=cv,

    n_jobs=-1,

    verbose=2,

    return_train_score=True
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 18. BEST MODEL
# ============================================================

best_model_pipeline = grid_search.best_estimator_

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Parameters:",
    grid_search.best_params_
)

print(
    "Best CV F1:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# 19. EXTRACT FITTED PREPROCESSOR AND MODEL
# ============================================================

fitted_preprocessor = (
    best_model_pipeline
    .named_steps["preprocess"]
)

fitted_model = (
    best_model_pipeline
    .named_steps["model"]
)


# ============================================================
# 20. TEST PREDICTION
# ============================================================

y_pred = best_model_pipeline.predict(
    X_test
)

y_probability = (
    best_model_pipeline
    .predict_proba(X_test)[:, 1]
)


# ============================================================
# 21. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0
    )
)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 22. SAVE PREPROCESSOR
# ============================================================

PREPROCESSOR_FILE = os.path.join(
    ARTIFACT_DIR,
    "preprocessor.pkl"
)

joblib.dump(
    fitted_preprocessor,
    PREPROCESSOR_FILE
)

print(
    "\nPreprocessor saved:",
    PREPROCESSOR_FILE
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

MODEL_FILE = os.path.join(
    ARTIFACT_DIR,
    "model.pkl"
)

joblib.dump(
    fitted_model,
    MODEL_FILE
)

print(
    "Model saved:",
    MODEL_FILE
)


# ============================================================
# 24. SAVE TARGET ENCODER
# ============================================================

TARGET_ENCODER_FILE = os.path.join(
    ARTIFACT_DIR,
    "target_encoder.pkl"
)

joblib.dump(
    target_encoder,
    TARGET_ENCODER_FILE
)

print(
    "Target encoder saved:",
    TARGET_ENCODER_FILE
)


# ============================================================
# 25. SAVE FEATURE CONFIGURATION
# ============================================================

feature_config = {

    "target": TARGET,

    "numeric_columns": numeric_columns,

    "categorical_columns": categorical_columns,

    "all_features": X.columns.tolist(),

    "random_state": RANDOM_STATE,

    "test_size": TEST_SIZE,

    "best_params": grid_search.best_params_,

    "cv_f1": grid_search.best_score_,

    "test_accuracy": accuracy,

    "test_precision": precision,

    "test_recall": recall,

    "test_f1": f1,

    "test_roc_auc": roc_auc
}


FEATURE_CONFIG_FILE = os.path.join(
    ARTIFACT_DIR,
    "feature_config.pkl"
)

joblib.dump(
    feature_config,
    FEATURE_CONFIG_FILE
)

print(
    "Feature configuration saved:",
    FEATURE_CONFIG_FILE
)


# ============================================================
# 26. SAVE COMPLETE PIPELINE AS BACKUP
# ============================================================

PIPELINE_FILE = os.path.join(
    ARTIFACT_DIR,
    "complete_pipeline.pkl"
)

joblib.dump(
    best_model_pipeline,
    PIPELINE_FILE
)

print(
    "Complete pipeline saved:",
    PIPELINE_FILE
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVING COMPLETED")
print("=" * 70)

print(
    """
Artifacts created:

artifacts/
    ├── preprocessor.pkl
    ├── model.pkl
    ├── feature_config.pkl
    ├── target_encoder.pkl
    └── complete_pipeline.pkl
"""
)

print("You can now deploy the model using Streamlit.")
```
