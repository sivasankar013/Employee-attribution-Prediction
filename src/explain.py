from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap


def _to_builtin(value):
    # Convert NumPy values into plain Python types so JSON can serialize them.
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def build_shap_summary(
    pipeline: Any,
    X_background,
    X_sample,
    out_dir: str | Path,
    *,
    max_display: int = 15,
) -> dict[str, Any]:
    # Create the output folder if it does not already exist.
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract the preprocessing step and the fitted model from the pipeline.
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Convert the sample into model-ready numeric features.
    transformed_sample = preprocessor.transform(X_sample)

    # SHAP helps us understand which features push the prediction up or down.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_sample)

    # Use the encoded feature names so the explanation matches the model input.
    feature_names = preprocessor.get_feature_names_out()
    try:
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception:
        values = shap_values

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        values,
        transformed_sample,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    summary_path = out_path / "shap_summary.png"
    plt.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Return a small JSON-friendly summary for the dashboard and documentation.
    return {
        "summary_path": str(summary_path),
        "base_value": _to_builtin(explainer.expected_value),
        "feature_names": feature_names.tolist(),
    }
