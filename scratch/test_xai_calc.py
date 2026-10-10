"""
scratch/test_xai_calc.py

Scratch script to verify SHAP, LIME, and Topology analysis calculations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import load_demo_pipeline, THRESHOLD_TAU

def compute_xai_shap_lime(demo_pipe, normal_window_raw, disturbed_window, target_sensor):
    model = demo_pipe["model"]
    preprocessor = demo_pipe["preprocessor"]
    adj_matrix = demo_pipe["adj_matrix"]
    mu_sq_h = demo_pipe["mu_sq_h"]
    sensor_cols = demo_pipe["sensor_cols"]
    normal_target_raw = demo_pipe["normal_target_raw"]

    normal_X = preprocessor.transform_features(normal_window_raw, sensor_cols)[np.newaxis, :, :].astype(np.float32)
    normal_Y = preprocessor.transform_features(normal_target_raw, sensor_cols)[np.newaxis, :, :].astype(np.float32)
    disturbed_X = preprocessor.transform_features(disturbed_window, sensor_cols)[np.newaxis, :, :].astype(np.float32)

    def eval_score_batch(X_batch):
        X_batch_f32 = X_batch.astype(np.float32)
        with torch.no_grad():
            preds = model(torch.from_numpy(X_batch_f32), adj_matrix=adj_matrix).numpy()
        # E_dist shape: (N, 12, 43)
        E = (preds - normal_Y) ** 2
        Z = E / mu_sq_h
        # Anomaly score per window is mean over timesteps and sensors at horizon 12 (index 11)
        scores = np.mean(Z[:, 11, :], axis=1)
        return scores

    s_dist = float(eval_score_batch(disturbed_X)[0])
    s_norm = float(eval_score_batch(normal_X)[0])

    # 1. Exact 2-Player Coalition SHAP
    N_sensors = len(sensor_cols)
    X_dist_sub = np.repeat(disturbed_X, N_sensors, axis=0) # (43, 12, 43)
    X_norm_add = np.repeat(normal_X, N_sensors, axis=0)  # (43, 12, 43)

    for k in range(N_sensors):
        X_dist_sub[k, :, k] = normal_X[0, :, k]
        X_norm_add[k, :, k] = disturbed_X[0, :, k]

    scores_dist_sub = eval_score_batch(X_dist_sub) # (43,)
    scores_norm_add = eval_score_batch(X_norm_add) # (43,)

    shap_values = 0.5 * (s_dist - scores_dist_sub) + 0.5 * (scores_norm_add - s_norm)

    shap_list = []
    for k, s_name in enumerate(sensor_cols):
        val = float(shap_values[k])
        direction = "Increased anomaly score" if val >= 0 else "Reduced anomaly score"
        shap_list.append({
            "Sensor": s_name,
            "SHAP Value": val,
            "Abs SHAP": abs(val),
            "Direction": direction
        })

    shap_df = pd.DataFrame(shap_list).sort_values(by="Abs SHAP", ascending=False).reset_index(drop=True)

    # 2. Local Surrogate LIME
    np.random.seed(42)
    N_samples = 50

    # Perturb disturbed_X directly in normalized feature space
    noise = np.random.normal(0.0, 0.5, size=(N_samples, 12, N_sensors)).astype(np.float32)
    # Mask noise to apply randomly to subset of sensors per sample
    mask = (np.random.uniform(0, 1, size=(N_samples, 1, N_sensors)) > 0.5).astype(np.float32)
    X_perturbed = disturbed_X + noise * mask

    scores_pert = eval_score_batch(X_perturbed)

    # Feature diffs at horizon 12 (timestep 11) relative to normal_X
    X_diff = (X_perturbed[:, 11, :] - normal_X[:, 11, :]).astype(np.float32)

    # Weights based on Euclidean distance to disturbed_X
    dists = np.sqrt(np.mean((X_perturbed - disturbed_X) ** 2, axis=(1, 2)))
    weights = np.exp(- (dists ** 2) / (2 * (0.5 ** 2)))

    # Fit Ridge regression
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_diff, scores_pert, sample_weight=weights)

    lime_coefs = ridge.coef_
    lime_list = []
    for k, s_name in enumerate(sensor_cols):
        val = float(lime_coefs[k])
        lime_list.append({
            "Sensor": s_name,
            "LIME Weight": val,
            "Abs LIME": abs(val)
        })

    lime_df = pd.DataFrame(lime_list).sort_values(by="Abs LIME", ascending=False).reset_index(drop=True)

    # Agreement
    shap_top4 = set(shap_df["Sensor"].iloc[:4])
    lime_top4 = set(lime_df["Sensor"].iloc[:4])
    overlap = len(shap_top4.intersection(lime_top4))

    if overlap >= 3:
        agreement = "High"
    elif overlap == 2:
        agreement = "Moderate"
    else:
        agreement = "Low"

    return shap_df, lime_df, agreement, s_dist, s_norm

def test_xai():
    demo_pipe = load_demo_pipeline()
    normal_window_raw = demo_pipe["normal_window_raw"].copy()
    disturbed_window = normal_window_raw.copy()
    target_sensor = "F_PU1"
    disturbed_window.loc[6:, target_sensor] += 4.0 * demo_pipe["sensor_stds"].get(target_sensor, 1.0)

    shap_df, lime_df, agreement, s_dist, s_norm = compute_xai_shap_lime(demo_pipe, normal_window_raw, disturbed_window, target_sensor)
    print("Baseline score:", s_norm, "Disturbed score:", s_dist)
    print("\nSHAP Top 5:")
    print(shap_df.head(5))
    print("\nLIME Top 5:")
    print(lime_df.head(5))
    print("\nAgreement:", agreement)

if __name__ == "__main__":
    test_xai()
