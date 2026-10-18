"""
Diagnostic script to investigate repeated GNN LIME rankings,
test integer feature indexing vs substring parsing,
measure local surrogate R^2, and analyze calibration denominators.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.preprocessing import DataPreprocessor
from dashboard.app import load_demo_pipeline

def run_diagnostic():
    print("=== GNN LIME & CALIBRATION DIAGNOSTIC AUDIT ===")

    demo_pipe = load_demo_pipeline()
    gnn_model = demo_pipe["model"]
    adj_matrix = demo_pipe["adj_matrix"]
    mu_sq_h = demo_pipe["mu_sq_h"]
    sensor_cols = demo_pipe["sensor_cols"]
    preprocessor = demo_pipe["preprocessor"]

    loader = BATADALDatasetLoader()
    df03, _, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv")
    df04, _, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset04.csv")

    features03 = preprocessor.transform_features(df03, sensor_cols)
    features04 = preprocessor.transform_features(df04, sensor_cols)
    X_04, Y_04, _ = preprocessor.create_forecasting_windows(features04, forecast_horizon=12)

    normal_baseline_vec = np.mean(features03, axis=0).astype(np.float32)

    attack_sequences = [
        {"seq_id": 1, "target_comp": "T1", "mid_timestep": 1740},
        {"seq_id": 2, "target_comp": "T2", "mid_timestep": 2360},
        {"seq_id": 3, "target_comp": "V2", "mid_timestep": 2865},
        {"seq_id": 4, "target_comp": "PU4", "mid_timestep": 3511},
        {"seq_id": 5, "target_comp": "T7", "mid_timestep": 3770},
    ]

    from lime.lime_tabular import LimeTabularExplainer

    print("\n--- Diagnostic 1: Tracing GNN LIME execution on each sequence ---")
    for seq in attack_sequences:
        t = seq["mid_timestep"]
        win_idx = t - 12
        X_act = X_04[win_idx:win_idx+1] # (1, 12, 43)
        Y_act = Y_04[win_idx:win_idx+1] # (1, 12, 43)

        with torch.no_grad():
            pred_act = gnn_model(torch.from_numpy(X_act), adj_matrix=adj_matrix).numpy()
        E_act = (pred_act - Y_act) ** 2
        Z_act = E_act / mu_sq_h
        s_act = float(np.mean(Z_act[0, 11, :]))

        def mask_score_fn(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(X_act, B, axis=0)
            for b in range(B):
                for f in range(43):
                    if coalition_masks[b, f] == 0:
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            E = (preds - Y_act) ** 2
            Z = E / mu_sq_h
            return np.mean(Z[:, 11, :], axis=1)

        # Baseline reference deltas
        rng = np.random.RandomState(42)
        ref_deltas = rng.normal(0.0, 0.2, size=(100, 43)).astype(np.float64)

        explainer = LimeTabularExplainer(
            training_data=ref_deltas,
            feature_names=sensor_cols,
            mode="regression",
            random_state=42,
            verbose=False,
        )

        data_row = np.ones(43, dtype=np.float64)
        exp = explainer.explain_instance(
            data_row=data_row,
            predict_fn=lambda x: mask_score_fn(x),
            num_features=43,
            num_samples=120,
        )

        # Method A: Substring parsing as currently done in lime_explainer.py
        raw_list = exp.as_list()
        attrs_substring = {name: 0.0 for name in sensor_cols}
        for condition_str, weight in raw_list:
            matched_sensor = None
            for name in sensor_cols:
                if name in condition_str:
                    matched_sensor = name
                    break
            if matched_sensor:
                attrs_substring[matched_sensor] = float(weight)

        top5_substring = sorted(attrs_substring.keys(), key=lambda k: abs(attrs_substring[k]), reverse=True)[:5]

        # Method B: Direct Integer Feature Indexing via exp.local_exp
        # In regression mode, exp.local_exp has key 1 or 0
        local_key = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
        attrs_integer = {name: 0.0 for name in sensor_cols}
        for feat_idx, weight in exp.local_exp[local_key]:
            attrs_integer[sensor_cols[feat_idx]] = float(weight)

        top5_integer = sorted(attrs_integer.keys(), key=lambda k: abs(attrs_integer[k]), reverse=True)[:5]

        print(f"\nSequence {seq['seq_id']} (t={t}, comp={seq['target_comp']}, Score={s_act:.2f}):")
        print(f"  Surrogate R^2 (exp.score): {exp.score:.6f}")
        print(f"  Method A (Substring Parsing) Top-5 : {top5_substring}")
        print(f"  Method B (Integer Indexing)  Top-5 : {top5_integer}")
        print(f"  Match between Method A and B? : {top5_substring == top5_integer}")
        if top5_substring != top5_integer:
            print("  [DISCREPANCY FOUND]: Substring parsing altered feature attribution!")
            # Show which conditions matched wrong names
            for condition_str, weight in raw_list[:5]:
                for name in sensor_cols:
                    if name in condition_str:
                        # find true name from integer
                        break
                print(f"    Condition: '{condition_str}' -> parsed as '{name}'")

    print("\n--- Diagnostic 2: Seed sensitivity test on Sequence 1 ---")
    t1 = 1740
    win_idx = t1 - 12
    X_act1 = X_04[win_idx:win_idx+1]
    Y_act1 = Y_04[win_idx:win_idx+1]
    def s1_fn(coalition_masks):
        B = len(coalition_masks)
        X_b = np.repeat(X_act1, B, axis=0)
        for b in range(B):
            for f in range(43):
                if coalition_masks[b, f] == 0:
                    X_b[b, :, f] = normal_baseline_vec[f]
        with torch.no_grad():
            preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
        E = (preds - Y_act1) ** 2
        Z = E / mu_sq_h
        return np.mean(Z[:, 11, :], axis=1)

    for test_seed in [42, 123, 999]:
        rng = np.random.RandomState(test_seed)
        ref_deltas = rng.normal(0.0, 0.2, size=(100, 43)).astype(np.float64)
        exp_seed = LimeTabularExplainer(
            training_data=ref_deltas,
            feature_names=sensor_cols,
            mode="regression",
            random_state=test_seed,
            verbose=False,
        ).explain_instance(
            data_row=np.ones(43, dtype=np.float64),
            predict_fn=s1_fn,
            num_features=43,
            num_samples=120,
        )
        local_key = 1 if 1 in exp_seed.local_exp else list(exp_seed.local_exp.keys())[0]
        seed_attrs = {sensor_cols[idx]: weight for idx, weight in exp_seed.local_exp[local_key]}
        seed_top5 = sorted(seed_attrs.keys(), key=lambda k: abs(seed_attrs[k]), reverse=True)[:5]
        print(f"  Seed {test_seed:3d}: Surrogate R^2 = {exp_seed.score:.6f} | Top-5 Integer Indexing: {seed_top5}")

if __name__ == "__main__":
    run_diagnostic()
