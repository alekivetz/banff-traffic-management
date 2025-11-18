from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# ------------------------
# Directory Setup
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "banff_route_data_preprocessed.csv")
all_data = pd.read_csv(DATA_PATH)

# Clean column names
all_data.columns = [c.strip() for c in all_data.columns]

# Convert route IDs to integers
all_data["route_num"] = all_data["route"].str.replace("Route ", "").astype(int)


# ------------------------
# Health Check
# ------------------------
@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/home")
def home():
    return {
        "message": "Welcome to the Banff Traffic Management API",
        "endpoints": ["/health", "/home", "/v1/predict", "/v2/predict"]
    }


# ------------------------
# Dummy Example
# ------------------------
@app.post("/v1/predict")
def predict_v1(dummy: dict):
    return {"prediction": 5.6, "note": "dummy test"}


# ------------------------
# V2 Input
# ------------------------
class TrafficInput(BaseModel):
    route_id: int
    hour: int


# ------------------------
# Load Correct Model
# ------------------------
def load_route_model(route_id: int):
    filename = f"Route_{route_id}_RandomForest.pkl"
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


# ------------------------
# Build Features (Matches Training)
# ------------------------
def compute_features(route_id: int, hour: int):
    df = all_data[all_data["route_num"] == route_id].copy()

    if df.empty:
        raise ValueError(f"No data found for route {route_id}")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Lags
    df["delay_lag_1"] = df["delay"].shift(1)
    df["mean_travel_time_lag_1"] = df["mean_travel_time"].shift(1)

    df["delay_lag_2"] = df["delay"].shift(2)
    df["mean_travel_time_lag_2"] = df["mean_travel_time"].shift(2)

    # Rolling (6)
    df["delay_roll_mean_6"] = df["delay"].rolling(6).mean()
    df["delay_roll_max_6"] = df["delay"].rolling(6).max()

    df["mean_travel_time_roll_mean_6"] = df["mean_travel_time"].rolling(6).mean()
    df["mean_travel_time_roll_max_6"] = df["mean_travel_time"].rolling(6).max()

    # Rolling (12)
    df["delay_roll_mean_12"] = df["delay"].rolling(12).mean()
    df["delay_roll_max_12"] = df["delay"].rolling(12).max()

    df["mean_travel_time_roll_mean_12"] = df["mean_travel_time"].rolling(12).mean()
    df["mean_travel_time_roll_max_12"] = df["mean_travel_time"].rolling(12).max()

    # select last row matching requested hour
    df_hour = df[df["hour"] == hour].tail(1)

    if df_hour.empty:
        raise ValueError(f"No data for route {route_id} at hour {hour}")

    # Keep only the 19 training columns
    feature_cols = [
        "speed", "mean_travel_time", "hour", "day_of_week", "is_weekend",
        "trend_ flat", "trend_ up",
        "delay_lag_1", "mean_travel_time_lag_1",
        "delay_lag_2", "mean_travel_time_lag_2",
        "delay_roll_mean_6", "delay_roll_max_6",
        "mean_travel_time_roll_mean_6", "mean_travel_time_roll_max_6",
        "delay_roll_mean_12", "delay_roll_max_12",
        "mean_travel_time_roll_mean_12", "mean_travel_time_roll_max_12"
    ]

    return df_hour[feature_cols]


# ------------------------
# Real Prediction
# ------------------------
@app.post("/v2/predict")
def predict_v2(data: TrafficInput):

    X = compute_features(data.route_id, data.hour)
    model = load_route_model(data.route_id)

    pred = model.predict(X)[0]

    return {
        "route_id": data.route_id,
        "hour": data.hour,
        "prediction": float(pred)
    }
