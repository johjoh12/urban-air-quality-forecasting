import os
import matplotlib.pyplot as plt
import numpy as np
import shap


def run_shap_analysis(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list,
    output_dir: str = "figures",
):
    os.makedirs(output_dir, exist_ok=True)

    # TreeExplainer is optimal for LightGBM, XGBoost, and CatBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Handle multidimensional array cases if returned by specific versions
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_val,
        feature_names=feature_names,
        show=False,
        max_display=12,
    )
    plt.title(
        "SHAP Feature Importance (Global Impact on $\log(1 + PM_{2.5})$)",
        fontsize=12,
        weight="bold",
        pad=15,
    )
    plt.tight_layout()

    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary to: {summary_path}")
