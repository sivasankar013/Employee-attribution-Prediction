from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import clean_data, load_raw_data


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
ARTIFACTS = ROOT / "artifacts"


@st.cache_resource
def load_artifacts():
    # Load the saved metrics and trained pipeline only once per session.
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    model = joblib.load(ARTIFACTS / "best_model_pipeline.joblib")
    return metrics, model


@st.cache_data
def load_dataset():
    # Cache the cleaned dataset so the dashboard stays responsive.
    return clean_data(load_raw_data(CSV_PATH))


def risk_band(prob: float) -> str:
    # Turn a numeric probability into a simple business-friendly label.
    if prob >= 0.7:
        return "High risk"
    if prob >= 0.4:
        return "Medium risk"
    return "Low risk"


def is_low_cardinality_numeric(series: pd.Series) -> bool:
    # Some numeric columns behave like categories, so treat them that way in the UI.
    return pd.api.types.is_numeric_dtype(series) and 2 <= series.nunique() <= 5


def series_bounds(series: pd.Series) -> tuple[float, float, float]:
    # Convert the series to numeric values before calculating widget bounds.
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError(f"Cannot build a widget for column {series.name!r} because it has no numeric values.")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    default = float(numeric.median())
    return minimum, maximum, default


def category_options(column: str, feature_frame: pd.DataFrame) -> tuple[list, dict]:
    # Keep the UI source of truth explicit so the dropdowns never go empty.
    fixed_string_columns = {
        "BusinessTravel",
        "Department",
        "EducationField",
        "JobRole",
        "MaritalStatus",
    }
    fixed_numeric_labels = {
        "Gender": {0: "Female", 1: "Male"},
        "OverTime": {0: "No", 1: "Yes"},
    }

    if column in fixed_numeric_labels:
        labels = fixed_numeric_labels[column]
        return list(labels.keys()), labels

    if column in fixed_string_columns:
        options = sorted(feature_frame[column].dropna().astype(str).unique().tolist())
        if not options:
            # Fallback to the raw CSV if the cleaned frame ever behaves unexpectedly in cloud deployment.
            raw_options = sorted(load_raw_data(CSV_PATH)[column].dropna().astype(str).unique().tolist())
            return raw_options, {v: v for v in raw_options}
        return options, {v: v for v in options}

    options = sorted(pd.to_numeric(feature_frame[column], errors="coerce").dropna().unique().tolist())
    return options, {v: str(v) for v in options}


def main():
    # Basic Streamlit page setup.
    st.set_page_config(page_title="HR Attrition Analytics Platform", layout="wide")
    st.title("HR Attrition Analytics Platform")
    st.caption("A portfolio-ready end-to-end project: evaluation, imbalance handling, SHAP, and prediction demo.")

    # Load reusable artifacts for the dashboard.
    if not (ARTIFACTS / "metrics.json").exists() or not (ARTIFACTS / "best_model_pipeline.joblib").exists():
        st.error(
            "Artifacts are missing from the deployed app. "
            "Commit the artifacts folder to GitHub before deploying."
        )
        st.stop()

    metrics, model = load_artifacts()
    data = load_dataset()

    # Split the dashboard into business overview, prediction, and explanation views.
    tab1, tab2, tab3 = st.tabs(["Executive Dashboard", "Employee Risk Predictor", "Model Intelligence"])

    with tab1:
        # Show the key business KPIs first.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attrition rate", f"{data['Attrition'].mean() * 100:.1f}%")
        c2.metric("Employees", f"{len(data):,}")
        c3.metric("Best model", metrics["best_model"])
        c4.metric("Validation threshold", f"{metrics['threshold']:.2f}")

        # Department and overtime charts help HR spot where risk is concentrated.
        left, right = st.columns(2)
        with left:
            st.subheader("Attrition by department")
            fig, ax = plt.subplots(figsize=(8, 4))
            dept = data.groupby("Department")["Attrition"].mean().sort_values(ascending=False) * 100
            dept.plot(kind="bar", ax=ax, color="#2E74B5")
            ax.set_ylabel("Attrition %")
            ax.set_xlabel("")
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig, clear_figure=True)

        with right:
            st.subheader("Overtime and attrition")
            fig, ax = plt.subplots(figsize=(8, 4))
            overtime = data.groupby("OverTime")["Attrition"].mean().rename({0: "No", 1: "Yes"}) * 100
            overtime.plot(kind="bar", ax=ax, color="#1F4D78")
            ax.set_ylabel("Attrition %")
            ax.set_xlabel("")
            st.pyplot(fig, clear_figure=True)

        # Higher income bands often show different churn patterns, so we visualize that too.
        st.subheader("Income distribution by attrition")
        fig, ax = plt.subplots(figsize=(10, 4))
        data.assign(IncomeBand=pd.qcut(data["MonthlyIncome"], q=4, duplicates="drop")).groupby(
            ["IncomeBand", "Attrition"]
        ).size().unstack(fill_value=0).plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Employees")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig, clear_figure=True)

        st.info(
            "Interpretation: higher attrition usually clusters around overtime, lower income, shorter tenure, and less stable role history."
        )

    with tab2:
        # This section lets a user simulate one employee and get a risk score.
        st.subheader("Predict attrition risk for a single employee")
        cols = st.columns(2)
        input_data = {}

        # Rebuild the cleaned feature frame so the input widgets match training features.
        raw = load_raw_data(CSV_PATH)
        cleaned = clean_data(raw)
        feature_frame = cleaned.drop(columns=["Attrition"])

        # Separate the numeric sliders from the categorical dropdowns.
        numeric_cols = [
            c for c in feature_frame.select_dtypes(include="number").columns
            if not is_low_cardinality_numeric(feature_frame[c])
        ]
        low_card_numeric_cols = [
            c for c in feature_frame.select_dtypes(include="number").columns
            if is_low_cardinality_numeric(feature_frame[c])
        ]
        categorical_cols = list(feature_frame.select_dtypes(exclude="number").columns)

        with cols[0]:
            for col in numeric_cols[: len(numeric_cols) // 2]:
                min_v, max_v, default = series_bounds(feature_frame[col])
                input_data[col] = st.slider(col, min_value=min_v, max_value=max_v, value=default)
        with cols[1]:
            for col in numeric_cols[len(numeric_cols) // 2 :]:
                min_v, max_v, default = series_bounds(feature_frame[col])
                input_data[col] = st.slider(col, min_value=min_v, max_value=max_v, value=default)

        # Small-cardinality numeric values are easier to use as dropdowns than sliders.
        st.markdown("#### Categorical inputs")
        cat_cols = st.columns(2)
        combined_categories = categorical_cols + low_card_numeric_cols
        for i, col in enumerate(combined_categories):
            # Make sure every dropdown always gets a non-empty option list.
            options, labels = category_options(col, feature_frame)
            with cat_cols[i % 2]:
                chosen = st.selectbox(col, options, format_func=lambda x, labels=labels: labels.get(x, str(x)))
                input_data[col] = chosen

        if st.button("Predict attrition risk"):
            # Predict a risk probability for the employee profile entered above.
            user_df = pd.DataFrame([input_data])
            prob = float(model.predict_proba(user_df)[0, 1])
            band = risk_band(prob)
            st.metric("Attrition risk", f"{prob * 100:.1f}%")
            st.write(f"Risk band: **{band}**")

            # If the model provides feature importances, show the main drivers.
            top_features = []
            try:
                preprocessor = model.named_steps["preprocessor"]
                model_estimator = model.named_steps["model"]
                feature_names = preprocessor.get_feature_names_out()
                if hasattr(model_estimator, "feature_importances_"):
                    importances = pd.Series(model_estimator.feature_importances_, index=feature_names).sort_values(ascending=False)
                    top_features = importances.head(5).index.tolist()
            except Exception:
                pass
            if top_features:
                st.write("Top model drivers:")
                st.write(top_features)

    with tab3:
        # This tab collects the evaluation story in one place.
        st.subheader("Model evaluation summary")
        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy",
                    "Value": metrics["test_metrics"]["accuracy"],
                },
                {
                    "Metric": "Precision",
                    "Value": metrics["test_metrics"]["precision"],
                },
                {
                    "Metric": "Recall",
                    "Value": metrics["test_metrics"]["recall"],
                },
                {
                    "Metric": "F1",
                    "Value": metrics["test_metrics"]["f1"],
                },
                {
                    "Metric": "ROC-AUC",
                    "Value": metrics["test_metrics"]["roc_auc"],
                },
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.code(metrics["classification_report"], language="text")

        # Show the validation-time benchmark table from the training script.
        comparison_path = ARTIFACTS / "model_comparison.csv"
        if comparison_path.exists():
            comp = pd.read_csv(comparison_path)
            st.subheader("Model comparison")
            st.dataframe(comp, use_container_width=True, hide_index=True)

        # Display SHAP and feature-importance images if they exist.
        shap_path = ARTIFACTS / "shap_summary.png"
        if shap_path.exists():
            st.subheader("SHAP summary")
            st.image(str(shap_path), use_container_width=True)
        feat_path = ARTIFACTS / "feature_importance.png"
        if feat_path.exists():
            st.subheader("Top feature importances")
            st.image(str(feat_path), use_container_width=True)


if __name__ == "__main__":
    main()
