# HR Attrition Analytics Platform

An end-to-end data science project for predicting employee attrition from the IBM HR dataset.

## What this project includes

- Proper evaluation with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- Class-imbalance handling using class weights and lightweight SMOTE-style oversampling
- Model benchmarking across logistic regression, random forest, histogram gradient boosting, and XGBoost
- Hyperparameter tuning with `RandomizedSearchCV`
- Explainable AI using SHAP
- A Streamlit dashboard for executives and HR stakeholders
- A single-employee attrition risk predictor

## Project structure

- `train.py` - trains models, compares them, tunes the best one, and saves artifacts
- `app.py` - Streamlit dashboard and prediction app
- `src/` - reusable data, modeling, oversampling, and explanation utilities
- `artifacts/` - generated metrics, plots, and trained model files

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the project:

```bash
python train.py
```

3. Launch the dashboard:

```bash
streamlit run app.py
```

## Business framing

This project is strong for a CV because it goes beyond a notebook:

- it handles an imbalanced target properly,
- it compares multiple models instead of reporting one score,
- it explains predictions with SHAP,
- and it exposes the result in a dashboard-style interface.

