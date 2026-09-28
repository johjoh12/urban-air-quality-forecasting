import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.data_pipeline import build_air_quality_features
from src.evaluate import plot_predictions
from src.interpret import run_shap_analysis
from src.models import evaluate_models

RAW_PATH = "data/raw/city_day.csv"
PROCESSED_PATH = "data/processed/delhi_featured.csv"


def run():
    print("[1/4] Running feature engineering pipeline...")
    os.makedirs("data/processed", exist_ok=True)
    df = build_air_quality_features(RAW_PATH, target_city="Delhi")
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dataset ready. Dimensions: {df.shape}")

    print("\n[2/4] Cross-validating models using TimeSeriesSplit...")
    leaderboard = evaluate_models(df, target_col="Target_PM25_t1")
    print("\n=== MODEL BENCHMARK LEADERBOARD ===")
    print(leaderboard.to_markdown(index=False))

    # Save leaderboard for README
    os.makedirs("figures", exist_ok=True)
    leaderboard.to_csv("figures/benchmark_results.csv", index=False)

    print("\n[3/4] Training final LightGBM surrogate on last split...")
    drop_cols = [
        "City",
        "Date",
        "AQI",
        "AQI_Bucket",
        "Target_PM25_t1",
        "Target_NO2_t1",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df["Target_PM25_t1"].values

    # Train on past, evaluate on final out-of-time horizon
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X):
        pass  # Grab the final outer fold

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    final_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    final_model.fit(X_train, y_train)
    val_preds = np.clip(final_model.predict(X_val), a_min=0, a_max=None)

    print("\n[4/4] Generating diagnostic plots and SHAP values...")
    plot_predictions(
        y_val, val_preds, model_name="LightGBM", output_dir="figures"
    )
    run_shap_analysis(
        final_model,
        X_train,
        X_val,
        feature_names=feature_cols,
        output_dir="figures",
    )
    print("\nExecution complete. Figures saved to ./figures/")


if __name__ == "__main__":
    run()
