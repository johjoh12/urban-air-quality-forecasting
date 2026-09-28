import numpy as np
import pandas as pd

def build_air_quality_features(csv_path: str, target_city: str = "Delhi") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Filter city and parse temporal index
    df = df[df["City"] == target_city].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Forward fill gaps within 3 days, drop untracked initial observations
    core_pollutants = ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3"]
    df[core_pollutants] = df[core_pollutants].interpolate(method="time", limit=3).bfill()
    
    # 1. Cyclical calendar encodings (Sine/Cosine for annual & weekly seasonal cycles)
    day_of_year = df["Date"].dt.dayofyear
    day_of_week = df["Date"].dt.dayofweek
    df["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["sin_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["cos_week"] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # 2. Lag features (t-1, t-2, t-3, t-7)
    lag_days = [1, 2, 3, 7]
    for pollutant in ["PM2.5", "NO2", "CO"]:
        for lag in lag_days:
            df[f"{pollutant}_lag{lag}"] = df[pollutant].shift(lag)
            
    # 3. Rolling statistics (Past 7-day & 14-day trends)
    for pollutant in ["PM2.5", "NO2"]:
        df[f"{pollutant}_roll_mean_7"] = df[pollutant].shift(1).rolling(7).mean()
        df[f"{pollutant}_roll_std_7"] = df[pollutant].shift(1).rolling(7).std()
        df[f"{pollutant}_roll_mean_14"] = df[pollutant].shift(1).rolling(14).mean()

    # 4. Target definition (Forecast next day: t+1)
    df["Target_PM25_t1"] = df["PM2.5"].shift(-1)
    df["Target_NO2_t1"] = df["NO2"].shift(-1)
    
    # Drop rows with NaNs created by lagging & shifting
    df = df.dropna().reset_index(drop=True)
    return df
# src/data_pipeline.py
import numpy as np
import pandas as pd


def build_air_quality_features(
    csv_path: str, target_city: str = "Delhi"
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 1. Filter city and convert Date to datetime
    df = df[df["City"] == target_city].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # 2. Temporarily set Date as index for time-weighted interpolation
    df = df.set_index("Date")

    core_pollutants = ["PM2.5", "PM10", "NO", "NO2", "CO", "SO2", "O3"]
    # Interpolate using time index, then backfill initial untracked points
    df[core_pollutants] = (
        df[core_pollutants].interpolate(method="time", limit=3).bfill()
    )

    # 3. Restore Date as a standard column
    df = df.reset_index()

    # 4. Cyclical calendar encodings (Sine/Cosine for annual & weekly seasonal cycles)
    day_of_year = df["Date"].dt.dayofyear
    day_of_week = df["Date"].dt.dayofweek
    df["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["sin_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["cos_week"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # 5. Lag features (t-1, t-2, t-3, t-7)
    lag_days = [1, 2, 3, 7]
    for pollutant in ["PM2.5", "NO2", "CO"]:
        for lag in lag_days:
            df[f"{pollutant}_lag{lag}"] = df[pollutant].shift(lag)

    # 6. Rolling statistics (Past 7-day & 14-day trends, shifted to prevent leakage)
    for pollutant in ["PM2.5", "NO2"]:
        df[f"{pollutant}_roll_mean_7"] = (
            df[pollutant].shift(1).rolling(7).mean()
        )
        df[f"{pollutant}_roll_std_7"] = df[pollutant].shift(1).rolling(7).std()
        df[f"{pollutant}_roll_mean_14"] = (
            df[pollutant].shift(1).rolling(14).mean()
        )

    # 7. Target definition (Forecast next day: t+1)
    df["Target_PM25_t1"] = df["PM2.5"].shift(-1)
    df["Target_NO2_t1"] = df["NO2"].shift(-1)

    # 8. Drop NaNs introduced by lagging and shifting
    df = df.dropna().reset_index(drop=True)
    return df
