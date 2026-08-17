````markdown
# 🚗 Car Loan Credit Risk Prediction

An end-to-end Machine Learning application for predicting credit risk associated with car loan applicants.

The project uses an AdaBoost Classification model with a preprocessing pipeline containing:

- KNN Imputation
- Robust Scaling
- Simple Imputation for categorical variables
- One-Hot Encoding
- AdaBoost Classification
- Cross-Validation
- GridSearchCV Hyperparameter Tuning

The trained model is deployed using Streamlit.

---

## 📌 Project Objective

The objective of this project is to develop a Machine Learning model that predicts the credit-risk class of a car loan applicant based on applicant and loan-related information.

The model produces:

- Predicted Class
- Class 0 probability
- Class 1 probability

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Remove Duplicates
     ↓
Train / Test Split
     ↓
Feature Identification
     ↓
Numerical Preprocessing
     ├── KNN Imputation
     └── Robust Scaling
     ↓
Categorical Preprocessing
     ├── Most Frequent Imputation
     └── One-Hot Encoding
     ↓
AdaBoost Classifier
     ↓
Stratified 5-Fold Cross Validation
     ↓
GridSearchCV
     ↓
Best Model
     ↓
Model Evaluation
     ↓
Save Model + Preprocessor
     ↓
Streamlit Deployment
````

---

## 📂 Project Structure

```text
car-loan-credit-risk/
│
├── local.py
├── app.py
├── requirements.txt
├── README.md
├── Car_Loan_Credit.csv
│
└── artifacts/
    ├── preprocessor.pkl
    ├── model.pkl
    ├── feature_config.pkl
    ├── target_encoder.pkl
    └── complete_pipeline.pkl
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 🤖 Machine Learning Model

The primary classification algorithm is:

### AdaBoost Classifier

AdaBoost is an ensemble learning algorithm that combines multiple weak learners to create a stronger classifier.

The model parameters are tuned using GridSearchCV.

Parameters tuned:

```text
n_estimators
learning_rate
```

Example parameter grid:

```python
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
```

---

## 🔧 Preprocessing

### Numerical Features

Numerical features are processed using:

```text
KNNImputer
     ↓
RobustScaler
```

KNNImputer handles missing numerical values.

RobustScaler is used because it is less sensitive to extreme values and outliers than StandardScaler.

---

### Categorical Features

Categorical features are processed using:

```text
SimpleImputer
     ↓
OneHotEncoder
```

The OneHotEncoder uses:

```python
handle_unknown="ignore"
```

This allows the application to handle an unseen categorical value without causing a prediction error.

---

## 🔐 Data Leakage Prevention

The preprocessing operations are fitted only on the training data through a Scikit-learn Pipeline.

The workflow is:

```text
Training Data
     ↓
Fit Preprocessor
     ↓
Transform Training Data
     ↓
Train Model
```

The test data is only transformed:

```text
Test Data
     ↓
Existing Fitted Preprocessor
     ↓
Prediction
```

This prevents information from the test set from leaking into the training process.

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

For a credit-risk application, special attention should be given to the performance of Class 1, particularly:

```text
Class 1 Recall
Class 1 Precision
Class 1 F1 Score
```

---

## 🚀 Installation

Clone or download the project.

Open a terminal inside the project directory.

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Place the dataset in the project directory:

```text
Car_Loan_Credit.csv
```

Run:

```bash
python local.py
```

The training script performs:

1. Dataset loading
2. Data cleaning
3. Duplicate removal
4. Target preparation
5. Train-test split
6. Preprocessing
7. Cross-validation
8. Hyperparameter tuning
9. Model training
10. Model evaluation
11. Artifact saving

After successful execution, the following files will be created:

```text
artifacts/
    preprocessor.pkl
    model.pkl
    feature_config.pkl
    target_encoder.pkl
    complete_pipeline.pkl
```

---

## 🌐 Run Streamlit Application

After the model artifacts are created, run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Streamlit Application

The application allows users to enter:

* Numerical applicant information
* Categorical applicant information
* Loan-related information

The application then:

```text
User Input
    ↓
DataFrame
    ↓
Saved Preprocessor
    ↓
Saved AdaBoost Model
    ↓
Prediction
```

The application displays:

* Predicted risk class
* Class 0 probability
* Class 1 probability
* Input information
* Model performance information

---

## 📦 Saved Model Artifacts

### preprocessor.pkl

Contains the fitted preprocessing pipeline:

```text
KNNImputer
RobustScaler
SimpleImputer
OneHotEncoder
```

---

### model.pkl

Contains the fitted AdaBoost classifier.

---

### feature_config.pkl

Contains:

* Feature names
* Numerical feature names
* Categorical feature names
* Target name
* Best hyperparameters
* Cross-validation score
* Test metrics

---

### target_encoder.pkl

Contains information about target encoding when the original target is categorical.

---

### complete_pipeline.pkl

Contains both:

```text
Preprocessor
+
AdaBoost Model
```

This file is maintained as a complete deployment backup.

---

## ⚠️ Important Deployment Rule

Do not retrain or refit the preprocessing pipeline inside `app.py`.

The application must use:

```python
preprocessor = joblib.load(
    "artifacts/preprocessor.pkl"
)
```

and:

```python
model = joblib.load(
    "artifacts/model.pkl"
)
```

The same preprocessing fitted during model training must be used during inference.

---

## ☁️ Streamlit Cloud Deployment

Upload the project to GitHub with the following structure:

```text
repository/
│
├── app.py
├── requirements.txt
├── README.md
│
└── artifacts/
    ├── preprocessor.pkl
    ├── model.pkl
    └── feature_config.pkl
```

Then create a Streamlit deployment using:

```text
app.py
```

as the main application file.

The application does not need the original CSV file for prediction after the model artifacts have been created.

---

## 🔒 Security Note

Do not upload confidential or personally identifiable customer information to GitHub.

The `.pkl` files should contain only the trained model and preprocessing artifacts.

---

## 👨‍💻 Project Summary

This project demonstrates an end-to-end Machine Learning workflow:

```text
Data Collection
       ↓
Data Cleaning
       ↓
EDA
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Cross Validation
       ↓
Hyperparameter Tuning
       ↓
Model Evaluation
       ↓
Model Serialization
       ↓
Streamlit Deployment
```

---

## 📌 Future Improvements

Potential future improvements include:

* Compare AdaBoost with Random Forest
* Compare AdaBoost with XGBoost
* Compare AdaBoost with LightGBM
* Handle class imbalance using class weights or resampling
* Optimize Class 1 recall
* Tune the classification threshold
* Add SHAP-based model explainability
* Add feature importance visualization
* Add probability-based risk categories
* Deploy the application to Streamlit Cloud

---

## 📜 License

This project is intended for educational, internship, and portfolio purposes.

```
```
