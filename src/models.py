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
    # Exclude metadata and potential target leakage columns
    drop_cols = [
        "City",
        "Date",
        "AQI",
        "AQI_Bucket",
        "Target_PM25_t1",
        "Target_NO2_t1",
        "Log_Target_PM25_t1",
        "Log_Target_NO2_t1",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target_col].values

    # Regularized models tuned for time-series stability
    models = {
        "Baseline (Ridge)": Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=120,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }

    results = []
    tscv = TimeSeriesSplit(n_splits=5)

    for name, model in models.items():
        rmse_scores, mae_scores, r2_scores = [], [], []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit on log-transformed targets
            y_train_log = np.log1p(y_train)
            model.fit(X_train, y_train_log)

            # Invert back to physical scale (ug/m3)
            raw_preds = model.predict(X_val)
            preds = np.expm1(raw_preds)
            preds = np.clip(preds, a_min=0, a_max=None)

            # Evaluate metrics on original scale
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
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

    return (
        pd.DataFrame(results)
        .sort_values(by="Mean RMSE")
        .reset_index(drop=True)
    )
