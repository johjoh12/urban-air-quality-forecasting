import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "figures",
):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 45-Degree Parity Scatter Plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, color="#1f77b4", edgecolor="none")
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Ideal Parity",
    )
    axes[0].set_title(
        f"{model_name}: Actual vs Predicted ($PM_{{2.5}}$)",
        fontsize=13,
        weight="bold",
    )
    axes[0].set_xlabel("Actual Concentration ($\mu g/m^3$)")
    axes[0].set_ylabel("Predicted Concentration ($\mu g/m^3$)")
    axes[0].legend()

    # 2. Residual Distribution Plot
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c", bins=35)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title(
        f"{model_name}: Residual Distribution Error",
        fontsize=13,
        weight="bold",
    )
    axes[1].set_xlabel("Residual ($y_{true} - y_{pred}$)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    save_path = os.path.join(
        output_dir, f"{model_name.lower().replace(' ', '_')}_eval.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plot to: {save_path}")
