import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def evaluate_models(
    df: pd.DataFrame, target_col: str = "Target_PM25_t1"
) -> pd.DataFrame:
    # 1. Separate features and target
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
    y = df[target_col].values

    # 2. Define models (Ridge wrapped in a Pipeline to scale features properly)
    models = {
        "Baseline (Ridge)": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=250,
            learning_rate=0.04,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
    }

    results = []
    tscv = TimeSeriesSplit(n_splits=5)

    # 3. Time-aware cross-validation
    for name, model in models.items():
        rmse_scores, mae_scores, r2_scores = [], [], []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # Enforce non-negative pollutant concentrations
            preds = np.clip(preds, a_min=0, a_max=None)

            rmse_scores.append(
                np.sqrt(mean_squared_error(y_val, preds))
            )  # or root_mean_squared_error in newer scikit-learn
            mae_scores.append(mean_absolute_error(y_val, preds))
            r2_scores.append(r2_score(y_val, preds))

        results.append(
            {
                "Model": name,
                "Mean RMSE": np.round(np.mean(rmse_scores), 3),
                "Mean MAE": np.round(np.mean(mae_scores), 3),
                "Mean R2": np.round(np.mean(r2_scores), 3),
            }
        )

    # Return leaderboard sorted by lowest RMSE
    return (
        pd.DataFrame(results)
        .sort_values(by="Mean RMSE")
        .reset_index(drop=True)
    )
