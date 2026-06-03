from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import clean_data, infer_column_types, split_xy
from .oversampling import smote_oversample

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    # XGBoost is optional, so keep the project runnable even if it is missing.
    XGBClassifier = None


@dataclass
class EvaluationResult:
    # Small container that keeps all key evaluation numbers together.
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    report: str
    threshold: float = 0.5


def make_one_hot_encoder() -> OneHotEncoder:
    # scikit-learn changed the OneHotEncoder argument name in newer versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Split the dataset into numeric and categorical columns.
    numeric_cols, categorical_cols = infer_column_types(X)

    # Scale numeric columns and one-hot encode categorical columns.
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_model_candidates(random_state: int, scale_pos_weight: float) -> dict[str, Any]:
    # Start with a few strong baseline models.
    models: dict[str, Any] = {
        "LogisticRegression (class_weight=balanced)": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest (class_weight=balanced_subsample)": RandomForestClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            random_state=random_state,
        ),
    }

    # XGBoost is often very strong on tabular data, so include it when available.
    if XGBClassifier is not None:
        models["XGBoost (scale_pos_weight)"] = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        )

    return models


def build_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    # Bundle preprocessing and modeling together so the workflow is reusable.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def evaluate_predictions(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    # Convert predicted probabilities into class labels using a threshold.
    y_pred = (y_prob >= threshold).astype(int)

    # Compute the metrics that matter for an imbalanced classification problem.
    return EvaluationResult(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        confusion=confusion_matrix(y_true, y_pred),
        report=classification_report(y_true, y_pred, digits=3, zero_division=0),
        threshold=threshold,
    )


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> tuple[float, dict[str, float]]:
    # Search for the threshold that gives the best F-beta score.
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if precision == 0 and recall == 0:
            score = 0.0
        else:
            beta_sq = beta**2
            score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-12)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {"precision": float(precision), "recall": float(recall), f"f{int(beta)}_score": float(score)}
    return best_threshold, best_metrics


def benchmark_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], ColumnTransformer]:
    # Fit the preprocessor on the training data only.
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Transform once so SMOTE can work on numeric arrays.
    X_train_enc = preprocessor.transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    # This is the common imbalance ratio used by XGBoost.
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    results: list[EvaluationResult] = []
    fitted_models: dict[str, Any] = {}

    # Evaluate the built-in model candidates first.
    for name, estimator in build_model_candidates(random_state, scale_pos_weight).items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        results.append(evaluate_predictions(name, y_val.to_numpy(), y_prob))
        fitted_models[name] = pipeline

    # Evaluate a couple of SMOTE-based variants on the encoded features.
    for name, estimator in {
        "LogisticRegression + SMOTE": LogisticRegression(max_iter=4000, solver="lbfgs", random_state=random_state),
        "RandomForest + SMOTE": RandomForestClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }.items():
        # Create a balanced training set with synthetic minority samples.
        X_res, y_res = smote_oversample(X_train_enc, y_train.to_numpy(), random_state=random_state)
        estimator.fit(X_res, y_res)
        y_prob = estimator.predict_proba(X_val_enc)[:, 1]
        results.append(evaluate_predictions(name, y_val.to_numpy(), y_prob))
        fitted_models[name] = {"preprocessor": preprocessor, "estimator": estimator}

    # Add the SMOTE version of XGBoost if the package is installed.
    if XGBClassifier is not None:
        xgb_smote = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )
        X_res, y_res = smote_oversample(X_train_enc, y_train.to_numpy(), random_state=random_state)
        xgb_smote.fit(X_res, y_res)
        y_prob = xgb_smote.predict_proba(X_val_enc)[:, 1]
        results.append(evaluate_predictions("XGBoost + SMOTE", y_val.to_numpy(), y_prob))
        fitted_models["XGBoost + SMOTE"] = {"preprocessor": preprocessor, "estimator": xgb_smote}

    results_df = pd.DataFrame(
        [
            {
                "model": r.name,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "roc_auc": r.roc_auc,
                "threshold": r.threshold,
            }
            for r in results
        ]
        ).sort_values(["f1", "recall", "roc_auc"], ascending=False)

    return results_df.reset_index(drop=True), fitted_models, preprocessor


def tune_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    # Rebuild the preprocessing pipeline for the final tuned model.
    preprocessor = build_preprocessor(X_train)

    # Choose a parameter search space based on the selected model family.
    if "LogisticRegression" in model_name:
        estimator = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=random_state)
        param_distributions = {
            "model__C": np.logspace(-2, 2, 15),
        }
        scoring = "f1"
    elif "RandomForest" in model_name:
        estimator = RandomForestClassifier(class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
        param_distributions = {
            "model__n_estimators": [300, 400, 500, 650],
            "model__max_depth": [None, 5, 7, 9, 12],
            "model__min_samples_split": [2, 4, 6, 8],
            "model__min_samples_leaf": [1, 2, 3, 4],
        }
        scoring = "f1"
    elif "XGBoost" in model_name and XGBClassifier is not None:
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        )
        param_distributions = {
            "model__n_estimators": [250, 350, 450, 600],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5, 7],
            "model__gamma": [0.0, 0.1, 0.2, 0.4],
            "model__reg_lambda": [1.0, 2.0, 4.0, 6.0],
        }
        scoring = "recall"
    else:
        estimator = HistGradientBoostingClassifier(random_state=random_state)
        param_distributions = {
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [3, 5, 7, None],
            "model__max_iter": [150, 250, 350],
        }
        scoring = "f1"

    # RandomizedSearchCV is faster than full grid search for larger spaces.
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=min(20, len(param_distributions[next(iter(param_distributions))])),
        scoring=scoring,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_val_test_split(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Clean the raw data first, then split into features and target.
    X, y = split_xy(clean_data(df))

    # Split into train+validation and final test sets with stratification.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Split the remaining data into a training set and a validation set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_joblib(obj: Any, path: str) -> None:
    # Persist the trained model or object so the app can load it later.
    joblib.dump(obj, path)
