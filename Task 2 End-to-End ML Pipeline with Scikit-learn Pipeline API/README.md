# 📊 End-to-End Machine Learning Pipeline with Scikit-learn

## 📌 Project Overview

This project demonstrates a complete end-to-end Machine Learning
pipeline using **Scikit-learn's Pipeline and ColumnTransformer APIs**.\
The objective is to build, compare, tune, evaluate, and deploy
**multiple machine learning models** for **Customer Churn Prediction**
using the **Telco Customer Churn dataset**.

The pipeline covers: - Data preprocessing - Feature engineering -
Training multiple models - Hyperparameter tuning - Model comparison -
Evaluation - Model persistence - Deployment-ready inference using
**Gradio**

------------------------------------------------------------------------

## 🧠 Problem Statement

Customer churn prediction is a **binary classification problem** where
the goal is to predict whether a customer will **leave the service**
based on demographic, account, and service-related features.

**Target Variable:** - `Churn` (Yes / No)

------------------------------------------------------------------------

## 📂 Project Structure

    ├── Task_2_End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb
    ├── Telco-Customer-Churn.csv
    ├── churn_pipeline.pkl
    ├── requirements.txt
    ├── app.py
    ├── Task 2 Hugging Face Link.txt
    ├── README.md

------------------------------------------------------------------------

## 🗃️ Dataset Information

-   **Dataset Name:** Telco Customer Churn\
-   **Source:** Kaggle Dataset\
-   **Type:** Structured tabular data

**Features include:** - Customer demographics - Account information -
Services subscribed - Billing details

------------------------------------------------------------------------

## ⚙️ Technologies & Libraries Used

-   Python
-   Pandas, NumPy
-   Scikit-learn
-   Matplotlib, Seaborn
-   Joblib
-   Gradio

------------------------------------------------------------------------

## 🔁 Machine Learning Pipeline Workflow

### 1️⃣ Data Loading

-   Dataset loaded using `pandas.read_csv()`
-   Initial exploratory inspection

### 2️⃣ Data Cleaning

-   Removal of unnecessary columns
-   Handling missing and invalid values
-   Correct type casting for numerical features

### 3️⃣ Feature & Target Split

-   Feature matrix (`X`)
-   Target vector (`y`)

------------------------------------------------------------------------

## 🧩 Preprocessing Pipeline

Implemented using **Scikit-learn's `Pipeline` and `ColumnTransformer`**
to ensure reproducibility and prevent data leakage.

### Numerical Features

-   Median Imputation
-   Standard Scaling

### Categorical Features

-   Most Frequent Imputation
-   One-Hot Encoding

This design ensures: - No data leakage - Clean, reusable preprocessing -
Production-ready workflow

------------------------------------------------------------------------

## 🤖 Model Building & Comparison

Two classification models are implemented and evaluated:

### 🔹 Logistic Regression

-   Acts as a strong linear baseline model
-   Interpretable coefficients
-   Efficient and fast for large datasets

### 🔹 Random Forest Classifier

-   Ensemble-based non-linear model
-   Captures complex feature interactions
-   Robust to overfitting

### Pipeline Structure

    Preprocessing → Classifier (Logistic Regression / Random Forest)

------------------------------------------------------------------------

## 🔍 Hyperparameter Tuning

-   **GridSearchCV** applied to both models
-   Cross-validation used to ensure robustness
-   Best-performing model selected based on evaluation metrics

------------------------------------------------------------------------

## 📈 Model Evaluation

Models are evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Confusion Matrix
-   ROC Curve

This allows a **fair comparison between Logistic Regression and Random
Forest** models.

------------------------------------------------------------------------

## 🏆 Model Selection

After evaluation, the **best-performing model** is selected and used as
the final pipeline for deployment.

------------------------------------------------------------------------

## 💾 Model Persistence

The **entire pipeline (preprocessing + selected model)** is saved using
`joblib`:

``` python
joblib.dump(best_model, "telco_churn_pipeline.pkl")
```

This enables direct inference without re-training.

------------------------------------------------------------------------

## 🚀 Deployment & Inference

### 🔄 Loading Saved Model

The saved pipeline is loaded and used for predictions on unseen customer
data.

### 🌐 Gradio Web Interface

A **Gradio-based UI** is implemented to: - Accept user inputs - Perform
real-time churn prediction - Demonstrate deployment readiness

------------------------------------------------------------------------

## 🖥️ How to Run the Project

### 1️⃣ Install Dependencies

``` bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio joblib
```

### 2️⃣ Run the Notebook

Open and execute:

    Task_2_End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb

### 3️⃣ Launch Gradio App

Run the Gradio cell to open the web interface.

------------------------------------------------------------------------

## ✅ Key Highlights

✔ End-to-end ML pipeline\
✔ Multiple models (Logistic Regression & Random Forest)\
✔ Clean preprocessing using Pipeline API\
✔ Hyperparameter tuning & model comparison\
✔ No data leakage\
✔ Production-ready model saving\
✔ Deployment demo using Gradio

------------------------------------------------------------------------

## 📌 Conclusion

This project showcases **best practices in applied machine learning**,
including: - Proper preprocessing pipelines - Model benchmarking -
Robust evaluation - Deployment-oriented design

It serves as a strong reference for **real-world ML systems**, academic
submissions, and professional portfolios.

## 👤 Author: Nayyab Zahra