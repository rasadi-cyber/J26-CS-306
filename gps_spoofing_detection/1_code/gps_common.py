import os
import numpy as np
import pandas as pd
import joblib

W = 50       # about 2 seconds at 25 Hz
STEP = 10    # raw files are about 250 Hz; keep every 10th row

FEATURES = ["velocity", "velocity_up", "speed_pos", "res_east", "res_north",
            "res_mag", "res_speed", "alt_diff",
            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
            "linear_acceleration_x", "linear_acceleration_y",
            "linear_acceleration_z"]

def add_features(g):
    g = g.copy()
    t = g["time_usec"] / 1e6
    lat0 = g["latitude_deg"].iloc[0]
    lon0 = g["longitude_deg"].iloc[0]
    x = (g["longitude_deg"] - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (g["latitude_deg"] - lat0) * 110540
    dt = t - t.shift(W)
    g["vx_pos"] = (x - x.shift(W)) / dt
    g["vy_pos"] = (y - y.shift(W)) / dt
    g["speed_pos"] = np.sqrt(g["vx_pos"] ** 2 + g["vy_pos"] ** 2)
    ve = g["velocity_east"].rolling(W).mean()
    vn = g["velocity_north"].rolling(W).mean()
    g["res_east"] = g["vx_pos"] - ve
    g["res_north"] = g["vy_pos"] - vn
    g["res_mag"] = np.sqrt(g["res_east"] ** 2 + g["res_north"] ** 2)
    g["res_speed"] = g["speed_pos"] - g["velocity"].rolling(W).mean()
    g["alt_diff"] = g["altitude"] - g["pressure_altitude"]
    return g

def prepare_flight(df):
    df = df.sort_values("time_usec").reset_index(drop=True)
    df = df.iloc[::STEP].copy().reset_index(drop=True)
    df = add_features(df)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def analyze_flight(csv_path, condition=None, bundle_path="gps_detector.pkl"):
    """Analyse one raw flight CSV (Merged format) and return a result dict."""
    bundle = joblib.load(bundle_path)
    df = prepare_flight(pd.read_csv(csv_path))
    prob = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    score = float(prob.mean())
    thr = bundle["thresholds"]["global"]
    if condition in bundle["thresholds"]:
        thr = bundle["thresholds"][condition]
    detected = score >= thr
    return {
        "component": "GPS Spoofing Detection",
        "flight_file": os.path.basename(csv_path),
        "detected_event": "GPS spoofing" if detected else "Normal flight",
        "confidence_score": round(score, 4),
        "threshold_used": round(float(thr), 4),
        "evidence": {
            "mean_position_velocity_residual": round(float(df["res_mag"].mean()), 4),
            "rows_analysed": int(len(df)),
        },
        "reliability_note": bundle["reliability_note"],
    }