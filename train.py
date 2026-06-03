from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_raw_data
from src.explain import build_shap_summary
from src.modeling import (
    benchmark_models,
    evaluate_predictions,
    optimize_threshold,
    save_joblib,
    train_val_test_split,
    tune_best_model,
)


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    # Create the artifact folder so all saved outputs have one home.
    ARTIFACTS.mkdir(exist_ok=True)

    # Load the raw CSV from disk.
    df = load_raw_data(CSV_PATH)

    # Build train, validation, and test splits with stratification.
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, random_state=42)

    # Compare several models on the validation set.
    benchmark_df, fitted_models, preprocessor = benchmark_models(
        X_train, X_val, y_train, y_val, random_state=42
    )
    benchmark_path = ARTIFACTS / "model_comparison.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    # Take the best performing validation model.
    best_name = str(benchmark_df.iloc[0]["model"])
    best_model = fitted_models[best_name]

    # Some models are stored directly as pipelines, while others are saved as dicts.
    if isinstance(best_model, dict):
        best_pipeline = None
        val_prob = best_model["estimator"].predict_proba(best_model["preprocessor"].transform(X_val))[:, 1]
    else:
        best_pipeline = best_model
        val_prob = best_pipeline.predict_proba(X_val)[:, 1]

    # Find the probability cutoff that gives a better recall-oriented tradeoff.
    best_threshold, threshold_metrics = optimize_threshold(y_val.to_numpy(), val_prob, beta=2.0)

    # Tune the selected model family using randomized search.
    tuned_model = tune_best_model(best_name.replace(" + SMOTE", ""), X_train, y_train, random_state=42)
    tuned_pipeline = tuned_model

    # Retrain on the combined train + validation data so the final model sees more examples.
    trainval_X = pd.concat([X_train, X_val], axis=0)
    trainval_y = pd.concat([y_train, y_val], axis=0)
    tuned_pipeline.fit(trainval_X, trainval_y)

    # Evaluate the final model on the untouched test split.
    test_prob = tuned_pipeline.predict_proba(X_test)[:, 1]
    test_result = evaluate_predictions(best_name, y_test.to_numpy(), test_prob, threshold=best_threshold)

    # Save the final model pipeline for the Streamlit app.
    save_joblib(tuned_pipeline, ARTIFACTS / "best_model_pipeline.joblib")

    # Save human-readable metrics so the dashboard can load them later.
    metadata = {
        "best_model": best_name,
        "threshold": best_threshold,
        "validation_threshold_metrics": threshold_metrics,
        "test_metrics": {
            "accuracy": test_result.accuracy,
            "precision": test_result.precision,
            "recall": test_result.recall,
            "f1": test_result.f1,
            "roc_auc": test_result.roc_auc,
        },
        "classification_report": test_result.report,
        "confusion_matrix": test_result.confusion.tolist(),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Plot feature importance when the model exposes it directly.
    try:
        model = tuned_pipeline.named_steps["model"]
        feature_names = tuned_pipeline.named_steps["preprocessor"].get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            plt.figure(figsize=(12, 6))
            importances.head(15).sort_values().plot(kind="barh")
            plt.title("Top Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(ARTIFACTS / "feature_importance.png", dpi=180, bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    # Build a SHAP summary to explain the model in human terms.
    try:
        shap_info = build_shap_summary(
            tuned_pipeline,
            X_background=X_train.sample(min(200, len(X_train)), random_state=42),
            X_sample=X_test.sample(min(200, len(X_test)), random_state=42),
            out_dir=ARTIFACTS,
        )
        (ARTIFACTS / "shap_meta.json").write_text(json.dumps(shap_info, indent=2), encoding="utf-8")
        shap_error = ARTIFACTS / "shap_error.txt"
        if shap_error.exists():
            shap_error.unlink()
    except Exception as exc:
        # Save the error text so we can debug SHAP issues later.
        (ARTIFACTS / "shap_error.txt").write_text(str(exc), encoding="utf-8")

    # Print a short training summary to the terminal.
    print(f"Best model: {best_name}")
    print(f"Validation threshold: {best_threshold:.2f}")
    print(f"Test recall: {test_result.recall:.3f}")
    print(f"Artifacts written to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
