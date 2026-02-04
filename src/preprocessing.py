import pandas as pd
import numpy as np

def classify_region(city_name):
    north_west = ["Lai Châu", "Điện Biên Phủ", "Sơn La", "Lào Cai"]

    north_east = ["Cao Bằng", "Lạng Sơn", "Tuyên Quang", "Thái Nguyên"]

    north = ["Hà Nội", "Hải Phòng", "Hạ Long", "Bắc Ninh", "Hưng Yên", "Ninh Bình", "Việt Trì"]

    north_central = ["Thanh Hóa", "Vinh", "Hà Tĩnh", "Huế"]

    central_coast = ["Đà Nẵng", "Đông Hà", "Quảng Ngãi", "Nha Trang"]

    central_highland = ["Buôn Ma Thuột", "Pleiku", "Đà Lạt"]

    southern = ["Hồ Chí Minh", "Biên Hòa", "Tây Ninh", "Cao Lãnh", "Long Xuyên", "Cần Thơ", "Vĩnh Long", "Cà Mau"]

    if city_name in north_west:
        return "Vùng I. Tây Bắc"
    if city_name in north_east:
        return "Vùng II. Đông Bắc"
    if city_name in north:
        return "Vùng III. Bắc Bộ"
    if city_name in north_central:
        return "Vùng IV. Bắc Trung Bộ"
    if city_name in central_coast:
        return "Vùng V. Nam Trung Bộ"
    if city_name in central_highland:
        return "Vùng VI. Tây Nguyên"
    if city_name in southern:
        return "Vùng VII. Nam Bộ"

def create_feature_temporal_social(df):
    print("Processing Group 1: Temporal & Social...")
    df = df.copy()
    # 1.1 Cyclical Hour
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 1.2 Day Parts
    def get_day_part(h):
        if 5 <= h < 10: return "morning"
        elif 10 <= h < 15: return "midday"
        elif 15 <= h < 18: return "afternoon"
        elif 18 <= h < 23: return "evening"
        else: return "night"
        
    df["day_part"] = df["hour"].apply(get_day_part).astype("category")

    # 1.3 Rush Hour
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # 1.4 Weekend
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # 1.5 Season
    df["month"] = df["timestamp"].dt.month
    
    def month_to_season(m):
        if m in [12, 1, 2]: return "winter"
        elif m in [3, 4, 5]: return "spring"
        elif m in [6, 7, 8]: return "summer"
        else: return "autumn"
        
    df["season"] = df["month"].apply(month_to_season).astype("category")

    # Cyclical Month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Convert City to Category
    if "city" in df.columns:
        df["city"] = df["city"].astype("category")

    return df

def create_feature_physic(df):
    print("Processing Group 2: Physics & Meteo...")
    df = df.copy()
    # 2.1 Wind Vector
    if "wind_speed" in df.columns and "wind_dir" in df.columns:
        wd_rad = df["wind_dir"] * np.pi / 180
        df["wind_x"] = df["wind_speed"] * np.cos(wd_rad)
        df["wind_y"] = df["wind_speed"] * np.sin(wd_rad)
        df = df.drop(columns=["wind_dir"], errors="ignore")

    # 2.2 Cumulative Rain (Washout effect)
    if "rain" in df.columns:
        df["rain_sum_6h"] = df.groupby("city", observed=True)["rain"].shift(1).rolling(6).sum().reset_index(0, drop=True)

    # 2.3 Temperature Difference 24h (Inversion proxy)
    if "temp" in df.columns:
        temp_shifted = df.groupby("city", observed=True)["temp"].shift(1)
        df["temp_diff_24h"] = (
            temp_shifted.rolling(24).max() - temp_shifted.rolling(24).min()
        ).reset_index(0, drop=True)

    # 2.4 Humidity-Temperature Interaction
    if "humidity" in df.columns and "temp" in df.columns:
        df["humid_x_temp"] = df["humidity"] * df["temp"]
    
    return df

def create_feature_history_trend(df):
    print("Processing Group 3: History & Trend...")
    target = "pm2_5"
    df = df.copy()
    # 3.1 Lag Features
    lags = [1, 2, 3, 24]
    for lag in lags:
        df[f"pm25_lag_{lag}h"] = df.groupby("city", observed=True)[target].shift(lag)

    # 3.2 Rolling Statistics
    pm25_shifted = df.groupby("city", observed=True)[target].shift(1)

    # Short-term (6h)
    df["pm25_rm_6h"] = pm25_shifted.rolling(6).mean().reset_index(0, drop=True)
    df["pm25_rs_6h"] = pm25_shifted.rolling(6).std().reset_index(0, drop=True)

    # Long-term (24h)
    df["pm25_rm_24h"] = pm25_shifted.rolling(24).mean().reset_index(0, drop=True)

    # 3.3 Short-term Trend
    df["pm25_trend_1h"] = df["pm25_lag_1h"] - df["pm25_lag_2h"]

    return df

def create_feature_composition(df):
    print("Processing Group 4: Composition...")
    df = df.copy()
    if "pm10" in df.columns and "pm2_5" in df.columns:
        # Coarse Dust
        df["coarse_dust"] = df["pm10"] - df["pm2_5"]
        # PM Ratio
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    # Exogenous Lags
    exo_cols = ["no2", "so2", "co", "o3", "coarse_dust", "pm_ratio", "pm10"]
    for col in exo_cols:
        if col in df.columns:
            df[f"{col}_lag1h"] = df.groupby("city", observed=True)[col].shift(1)
    
    return df


def train_val_test_split(df, train_ratio=0.8, val_ratio=0.1):
    df_model = df.copy()
    # Create Target (Shift -1 hour)
    df_model["target_future"] = df_model.groupby("city", observed=True)["pm2_5"].shift(-1)

    # Drop NaNs created by lags/rolling/shift
    df_model = df_model.dropna()

    # Get unique timestamps
    timestamps = (
        df_model[["timestamp"]]
        .drop_duplicates()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    n_ts = len(timestamps)

    train_ts_end = int(n_ts * train_ratio)
    val_ts_end = int(n_ts * (train_ratio + val_ratio))

    # Define dates for split
    train_cut = timestamps.iloc[train_ts_end]["timestamp"].replace(day=1).normalize()
    val_cut = timestamps.iloc[val_ts_end]["timestamp"].replace(day=1).normalize()

    # Split Data
    train = df_model[df_model["timestamp"] < train_cut]
    val = df_model[
        (df_model["timestamp"] >= train_cut)
        & (df_model["timestamp"] < val_cut)
    ]
    test = df_model[df_model["timestamp"] >= val_cut]

    return train, val, test, train_cut, val_cut
    
