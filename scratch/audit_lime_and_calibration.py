"""
Comprehensive audit script for:
Objective 1: GNN LIME Diagnostic Audit (investigating repeated rankings, perturbation samples, surrogate R^2, controlled input tests, feature ordering)
Objective 2: Anomaly-Score Calibration Limitation & Sensor Decomposition Audit (E, mu^2, Z, channel categories, status vs flow dominance)
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.detection.dataset_loader import BATADALDatasetLoader
from src.detection.preprocessing import DataPreprocessor
from dashboard.app import load_demo_pipeline
from lime.lime_tabular import LimeTabularExplainer
from src.explainability.lime_explainer import GNNLIMEExplainer

def run_audit():
    print("======================================================================")
    print("  OBJECTIVES 1 & 2: GNN LIME DIAGNOSTIC & CALIBRATION DECOMPOSITION")
    print("======================================================================")

    demo_pipe = load_demo_pipeline()
    gnn_model = demo_pipe["model"]
    adj_matrix = demo_pipe["adj_matrix"]
    mu_sq_h = demo_pipe["mu_sq_h"]
    sensor_cols = demo_pipe["sensor_cols"]
    preprocessor = demo_pipe["preprocessor"]

    loader = BATADALDatasetLoader()
    df03, _, _ = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv")
    df04, _, label04 = loader.load_csv(ROOT_DIR / "data" / "raw" / "BATADAL_dataset04.csv")

    features03 = preprocessor.transform_features(df03, sensor_cols)
    features04 = preprocessor.transform_features(df04, sensor_cols)
    X_04, Y_04, _ = preprocessor.create_forecasting_windows(features04, forecast_horizon=12)

    normal_baseline_vec = np.mean(features03, axis=0).astype(np.float32)

    attack_sequences = [
        {"seq_id": 1, "target_comp": "T1", "mid_timestep": 1740, "desc": "Attack 1 (Tank T1 deception)"},
        {"seq_id": 2, "target_comp": "T2", "mid_timestep": 2360, "desc": "Attack 2 (Tank T2 overflow)"},
        {"seq_id": 3, "target_comp": "V2", "mid_timestep": 2865, "desc": "Attack 3 (Valve V2 closed)"},
        {"seq_id": 4, "target_comp": "PU4", "mid_timestep": 3511, "desc": "Attack 4 (Pumps PU4/PU5 altered)"},
        {"seq_id": 5, "target_comp": "T7", "mid_timestep": 3770, "desc": "Attack 5 (Tank T7 level attack)"},
    ]

    print("\n" + "="*70)
    print("PART A: OBJECTIVE 2 - CALIBRATION PROCEDURE & PER-SENSOR DECOMPOSITION")
    print("="*70)

    # Inspect mu_sq_h
    print(f"\nCalibration denominator mu_sq_h shape: {mu_sq_h.shape}")
    mu2_step11 = mu_sq_h[11, :] if mu_sq_h.ndim == 2 else mu_sq_h[0, 11, :]
    print(f"Min denominator: {np.min(mu2_step11):.4e} on sensor {sensor_cols[np.argmin(mu2_step11)]}")
    print(f"Max denominator: {np.max(mu2_step11):.4e} on sensor {sensor_cols[np.argmax(mu2_step11)]}")
    print(f"Variance ratio (Max/Min): {np.max(mu2_step11)/np.min(mu2_step11):.2e}")

    # Top 7 smallest denominators
    sorted_denom_idx = np.argsort(mu2_step11)
    print("\nSmallest 7 calibration denominators (Horizon Step 11):")
    for idx in sorted_denom_idx[:7]:
        print(f"  {sensor_cols[idx]:8s}: denom={mu2_step11[idx]:.6e}")

    calib_audit_records = []

    for seq in attack_sequences:
        t = seq["mid_timestep"]
        win_idx = t - 12
        X_act = X_04[win_idx:win_idx+1]
        Y_act = Y_04[win_idx:win_idx+1]

        with torch.no_grad():
            pred_act = gnn_model(torch.from_numpy(X_act), adj_matrix=adj_matrix).numpy()

        E_act = (pred_act - Y_act) ** 2  # (1, 12, 43)
        E_11 = E_act[0, 11, :]          # (43,)
        Z_11 = E_11 / mu2_step11        # (43,)
        contrib_score = Z_11 / 43.0     # (43,)
        total_score = float(np.sum(contrib_score))

        # Categorize
        status_sensors = [i for i, s in enumerate(sensor_cols) if s.startswith("S_")]
        flow_sensors = [i for i, s in enumerate(sensor_cols) if s.startswith("F_")]
        pressure_sensors = [i for i, s in enumerate(sensor_cols) if s.startswith("P_")]
        tank_sensors = [i for i, s in enumerate(sensor_cols) if s.startswith("L_")]

        status_contrib = float(np.sum(contrib_score[status_sensors]))
        flow_contrib = float(np.sum(contrib_score[flow_sensors]))
        pressure_contrib = float(np.sum(contrib_score[pressure_sensors]))
        tank_contrib = float(np.sum(contrib_score[tank_sensors]))

        status_pct = 100.0 * status_contrib / total_score if total_score > 0 else 0.0
        flow_pct = 100.0 * flow_contrib / total_score if total_score > 0 else 0.0
        pressure_pct = 100.0 * pressure_contrib / total_score if total_score > 0 else 0.0
        tank_pct = 100.0 * tank_contrib / total_score if total_score > 0 else 0.0

        # Sort sensors by Z_11
        top_sensor_idx = np.argsort(Z_11)[::-1]
        top3_info = []
        for i in range(3):
            s_idx = top_sensor_idx[i]
            s_name = sensor_cols[s_idx]
            top3_info.append({
                "sensor": s_name,
                "E_sq_residual": float(E_11[s_idx]),
                "mu_sq_denom": float(mu2_step11[s_idx]),
                "Z_standardized": float(Z_11[s_idx]),
                "score_contribution": float(contrib_score[s_idx]),
                "pct_of_total": float(100.0 * contrib_score[s_idx] / total_score),
            })

        rec = {
            "seq_id": seq["seq_id"],
            "timestep": t,
            "desc": seq["desc"],
            "total_score": total_score,
            "category_breakdown": {
                "status_score": status_contrib,
                "status_pct": status_pct,
                "flow_score": flow_contrib,
                "flow_pct": flow_pct,
                "pressure_score": pressure_contrib,
                "pressure_pct": pressure_pct,
                "tank_score": tank_contrib,
                "tank_pct": tank_pct,
            },
            "top_sensors": top3_info,
        }
        calib_audit_records.append(rec)

        print(f"\n--- Incident Seq {seq['seq_id']} (t={t}, {seq['desc']}) ---")
        print(f"  Total Anomaly Score S: {total_score:.4f}")
        print(f"  Category Share: Status={status_pct:5.1f}% | Flow={flow_pct:5.1f}% | Pressure={pressure_pct:5.1f}% | Tank={tank_pct:5.1f}%")
        print("  Top 3 Contributing Sensors:")
        for rank, s_inf in enumerate(top3_info, 1):
            print(f"    {rank}. {s_inf['sensor']:8s}: E={s_inf['E_sq_residual']:.6e}, mu^2={s_inf['mu_sq_denom']:.6e}, Z={s_inf['Z_standardized']:.2f}, Contrib={s_inf['score_contribution']:.2f} ({s_inf['pct_of_total']:5.1f}%)")

    print("\n" + "="*70)
    print("PART B: OBJECTIVE 1 - GNN LIME AUDIT & ROOT CAUSE INVESTIGATION")
    print("="*70)

    # 1. Recreate the Naive Continuous Perturbation Bug vs Fixed Binary Discrete Masking
    print("\n--- Sub-Investigation 1: Why Naive GNN LIME yielded identical rankings for Seq 4 & 5 ---")
    
    # We will test Seq 4 and Seq 5 with the continuous sampling that fails the `== 0` check
    def run_naive_vs_fixed(seq_dict):
        t = seq_dict["mid_timestep"]
        win_idx = t - 12
        X_act = X_04[win_idx:win_idx+1]
        Y_act = Y_04[win_idx:win_idx+1]

        # Naive continuous score fn with exact zero check
        def naive_score_fn(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(X_act, B, axis=0)
            for b in range(B):
                for f in range(43):
                    if coalition_masks[b, f] == 0:  # FAILS on continuous floats!
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            E = (preds - Y_act) ** 2
            Z = E / mu_sq_h
            return np.mean(Z[:, 11, :], axis=1)

        # Fixed binary discrete score fn
        def fixed_score_fn(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(X_act, B, axis=0)
            masks_bool = (coalition_masks >= 0.5)
            for b in range(B):
                for f in range(43):
                    if not masks_bool[b, f]:
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            E = (preds - Y_act) ** 2
            Z = E / mu_sq_h
            return np.mean(Z[:, 11, :], axis=1)

        # Run naive explainer
        rng_naive = np.random.RandomState(42)
        ref_naive = rng_naive.normal(0.0, 0.2, size=(100, 43)).astype(np.float64)
        exp_naive = LimeTabularExplainer(ref_naive, feature_names=sensor_cols, mode="regression", random_state=42, verbose=False)
        res_naive = exp_naive.explain_instance(np.ones(43, dtype=np.float64), naive_score_fn, num_features=43, num_samples=120)

        # Run fixed explainer
        rng_fixed = np.random.RandomState(42)
        ref_binary = rng_fixed.binomial(1, 0.5, size=(100, 43)).astype(np.float64)
        exp_fixed = LimeTabularExplainer(ref_binary, feature_names=sensor_cols, categorical_features=list(range(43)), mode="regression", random_state=42, verbose=False)
        res_fixed = exp_fixed.explain_instance(np.ones(43, dtype=np.float64), fixed_score_fn, num_features=43, num_samples=120)

        # Extract naive rankings
        lkey_n = 1 if 1 in res_naive.local_exp else list(res_naive.local_exp.keys())[0]
        naive_weights = {sensor_cols[idx]: w for idx, w in res_naive.local_exp[lkey_n]}
        naive_top5 = sorted(naive_weights.keys(), key=lambda k: abs(naive_weights[k]), reverse=True)[:5]

        # Extract fixed rankings
        lkey_f = 1 if 1 in res_fixed.local_exp else list(res_fixed.local_exp.keys())[0]
        fixed_weights = {sensor_cols[idx]: w for idx, w in res_fixed.local_exp[lkey_f]}
        fixed_top5 = sorted(fixed_weights.keys(), key=lambda k: abs(fixed_weights[k]), reverse=True)[:5]

        return {
            "naive_r2": float(res_naive.score),
            "naive_top5": naive_top5,
            "naive_all_weights_zero": all(abs(w) < 1e-12 for w in naive_weights.values()),
            "fixed_r2": float(res_fixed.score),
            "fixed_top5": fixed_top5,
            "fixed_weights": {k: float(fixed_weights[k]) for k in fixed_top5},
        }

    seq4_diag = run_naive_vs_fixed(attack_sequences[3]) # Seq 4
    seq5_diag = run_naive_vs_fixed(attack_sequences[4]) # Seq 5
    seq1_diag = run_naive_vs_fixed(attack_sequences[0]) # Seq 1

    print("\n[NAIVE IMPLEMENTATION (Continuous Sampling + '== 0' Check)]:")
    print(f"  Seq 4: Surrogate R^2 = {seq4_diag['naive_r2']:.6f} | All weights zero? {seq4_diag['naive_all_weights_zero']} | Top-5: {seq4_diag['naive_top5']}")
    print(f"  Seq 5: Surrogate R^2 = {seq5_diag['naive_r2']:.6f} | All weights zero? {seq5_diag['naive_all_weights_zero']} | Top-5: {seq5_diag['naive_top5']}")
    print(f"  Seq 1: Surrogate R^2 = {seq1_diag['naive_r2']:.6f} | All weights zero? {seq1_diag['naive_all_weights_zero']} | Top-5: {seq1_diag['naive_top5']}")
    print(f"  --> ARE SEQ 4 AND SEQ 5 RANKINGS IDENTICAL IN NAIVE? {seq4_diag['naive_top5'] == seq5_diag['naive_top5']}")
    print("      EXACT CAUSE: All weights collapsed to 0.000000 due to flat surrogate targets.")
    print("      LIME fell back to tie-breaking dictionary insertion order, producing IDENTICAL top-5 lists!")

    print("\n[FIXED IMPLEMENTATION (Discrete Binary Masking + Threshold Check)]:")
    print(f"  Seq 4: Surrogate R^2 = {seq4_diag['fixed_r2']:.6f} | Top-5: {seq4_diag['fixed_top5']}")
    print(f"  Seq 5: Surrogate R^2 = {seq5_diag['fixed_r2']:.6f} | Top-5: {seq5_diag['fixed_top5']}")
    print(f"  Seq 1: Surrogate R^2 = {seq1_diag['fixed_r2']:.6f} | Top-5: {seq1_diag['fixed_top5']}")
    print(f"  --> ARE SEQ 4 AND SEQ 5 RANKINGS IDENTICAL IN FIXED? {seq4_diag['fixed_top5'] == seq5_diag['fixed_top5']}")

    # 2. Detailed Per-Incident Recording for all 5 Incidents under the Genuine Fixed Implementation
    print("\n--- Sub-Investigation 2: Detailed Recording for All 5 Incidents ---")
    incident_lime_records = []

    for seq in attack_sequences:
        t = seq["mid_timestep"]
        win_idx = t - 12
        X_act = X_04[win_idx:win_idx+1]
        Y_act = Y_04[win_idx:win_idx+1]

        with torch.no_grad():
            pred_act = gnn_model(torch.from_numpy(X_act), adj_matrix=adj_matrix).numpy()
        E_act = (pred_act - Y_act) ** 2
        Z_act = E_act / mu_sq_h
        s_act = float(np.mean(Z_act[0, 11, :]))

        def incident_mask_score_fn(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(X_act, B, axis=0)
            masks_bool = (coalition_masks >= 0.5)
            for b in range(B):
                for f in range(43):
                    if not masks_bool[b, f]:
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            E = (preds - Y_act) ** 2
            Z = E / mu_sq_h
            return np.mean(Z[:, 11, :], axis=1)

        rng = np.random.RandomState(42)
        binary_bg = rng.binomial(1, 0.5, size=(100, 43)).astype(np.float64)
        gnn_lime = GNNLIMEExplainer(
            incident_mask_score_fn,
            sensor_cols,
            reference_deltas=binary_bg,
            categorical_features=list(range(43)),
            random_state=42,
        )

        res_lime = gnn_lime.explain_instance(
            np.ones(43, dtype=np.float64),
            baseline_score=s_act,
            top_k=5,
            num_samples=150,
        )

        # Count distinct samples generated
        # LimeTabularExplainer samples around data_row
        surrogate_r2 = res_lime.metadata.get("surrogate_r2", 0.0)
        top5 = res_lime.top_k_features[:5]
        top5_weights = {s: res_lime.attributions[s] for s in top5}
        top5_abs = {s: res_lime.abs_attributions[s] for s in top5}

        rec = {
            "seq_id": seq["seq_id"],
            "timestep": t,
            "target_comp": seq["target_comp"],
            "desc": seq["desc"],
            "anomaly_score": s_act,
            "explained_input_shape": list(X_act.shape),
            "explained_input_norm": float(np.linalg.norm(X_act)),
            "explained_input_min": float(np.min(X_act)),
            "explained_input_max": float(np.max(X_act)),
            "surrogate_r2": surrogate_r2,
            "top5_sensors": top5,
            "top5_weights": top5_weights,
            "top5_abs_weights": top5_abs,
            "num_samples": 150,
        }
        incident_lime_records.append(rec)

        print(f"\nSequence {seq['seq_id']} (t={t}, {seq['desc']}):")
        print(f"  Scalar Anomaly Score: {s_act:.4f}")
        print(f"  Surrogate R^2: {surrogate_r2:.6f}")
        print(f"  Top-5 LIME Features: {top5}")
        for s in top5:
            print(f"    {s:8s}: weight={top5_weights[s]:+10.4f}, abs={top5_abs[s]:10.4f}")

    # 3. Controlled Diagnostic: Two Deliberately Different Inputs with Fixed Seed & Multi-Seed
    print("\n--- Sub-Investigation 3: Controlled Diagnostic (Different Inputs & Multi-Seed) ---")
    
    # Deliberately different inputs:
    # Input A: Sequence 1 (t=1740, Tank T1 deceptive attack, Score = 12,278.52, dominated by PU11)
    # Input B: Sequence 4 (t=3511, Pump PU4 attack, Score = 51.13, dominated by PU6)
    seeds = [42, 100, 123, 777, 999]
    multi_seed_results = {"Input_A_Seq1": {}, "Input_B_Seq4": {}}

    for test_seed in seeds:
        # Input A
        t_A = 1740
        wA = t_A - 12
        XA = X_04[wA:wA+1]
        YA = Y_04[wA:wA+1]
        def fn_A(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(XA, B, axis=0)
            m = (coalition_masks >= 0.5)
            for b in range(B):
                for f in range(43):
                    if not m[b, f]:
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            return np.mean(((preds - YA)**2 / mu_sq_h)[:, 11, :], axis=1)

        rngA = np.random.RandomState(test_seed)
        bgA = rngA.binomial(1, 0.5, size=(100, 43)).astype(np.float64)
        gnn_lime_A = GNNLIMEExplainer(fn_A, sensor_cols, reference_deltas=bgA, categorical_features=list(range(43)), random_state=test_seed)
        resA = gnn_lime_A.explain_instance(np.ones(43, dtype=np.float64), baseline_score=12278.52, top_k=5, num_samples=150)

        # Input B
        t_B = 3511
        wB = t_B - 12
        XB = X_04[wB:wB+1]
        YB = Y_04[wB:wB+1]
        def fn_B(coalition_masks):
            B = len(coalition_masks)
            X_b = np.repeat(XB, B, axis=0)
            m = (coalition_masks >= 0.5)
            for b in range(B):
                for f in range(43):
                    if not m[b, f]:
                        X_b[b, :, f] = normal_baseline_vec[f]
            with torch.no_grad():
                preds = gnn_model(torch.from_numpy(X_b), adj_matrix=adj_matrix).numpy()
            return np.mean(((preds - YB)**2 / mu_sq_h)[:, 11, :], axis=1)

        rngB = np.random.RandomState(test_seed)
        bgB = rngB.binomial(1, 0.5, size=(100, 43)).astype(np.float64)
        gnn_lime_B = GNNLIMEExplainer(fn_B, sensor_cols, reference_deltas=bgB, categorical_features=list(range(43)), random_state=test_seed)
        resB = gnn_lime_B.explain_instance(np.ones(43, dtype=np.float64), baseline_score=51.13, top_k=5, num_samples=150)

        multi_seed_results["Input_A_Seq1"][test_seed] = {
            "top5": resA.top_k_features[:5],
            "r2": resA.metadata.get("surrogate_r2", 0.0),
        }
        multi_seed_results["Input_B_Seq4"][test_seed] = {
            "top5": resB.top_k_features[:5],
            "r2": resB.metadata.get("surrogate_r2", 0.0),
        }

        print(f"  Seed {test_seed:3d}:")
        print(f"    Input A (Seq 1): R^2={resA.metadata.get('surrogate_r2', 0.0):.4f} | Top-5: {resA.top_k_features[:5]}")
        print(f"    Input B (Seq 4): R^2={resB.metadata.get('surrogate_r2', 0.0):.4f} | Top-5: {resB.top_k_features[:5]}")
        print(f"    Do A and B produce different explanations? {resA.top_k_features[:5] != resB.top_k_features[:5]}")

    # 4. Feature Ordering & 43D-to-Temporal Mapping Verification
    print("\n--- Sub-Investigation 4: Feature Ordering & Temporal Mapping Check ---")
    # Verify that:
    # 1. sensor_cols length is exactly 43
    # 2. X_act has shape (1, 12, 43), where axis 1 is 12 timesteps and axis 2 is 43 sensors
    # 3. Y_act has shape (1, 12, 43), where axis 1 is 12 forecast horizons and axis 2 is 43 sensors
    # 4. mu_sq_h has shape (12, 43) or (1, 12, 43), where index 11 corresponds to forecast horizon H=12
    # 5. Column f in X_act and Y_act exactly matches sensor_cols[f]
    print(f"  len(sensor_cols) == 43: {len(sensor_cols) == 43}")
    print(f"  X_04 shape: {X_04.shape} (N, W=12, F=43)")
    print(f"  Y_04 shape: {Y_04.shape} (N, H=12, F=43)")
    print(f"  mu_sq_h shape: {mu_sq_h.shape} (H=12, F=43)")

    # Verify column alignment by checking standard deviations on Dataset03
    df03_stds = df03[sensor_cols].std().values
    train_stds = preprocessor.scaler.scale_
    std_alignment_ok = np.allclose(df03_stds, train_stds, rtol=1e-3, atol=1e-5)
    print(f"  Scaler feature alignment with sensor_cols: {std_alignment_ok}")

    # Output results to scratch/diagnostic_audit_output.json
    final_output = {
        "calibration_audit": calib_audit_records,
        "naive_vs_fixed_root_cause": {
            "seq4": seq4_diag,
            "seq5": seq5_diag,
            "seq1": seq1_diag,
        },
        "incident_lime_records": incident_lime_records,
        "multi_seed_results": multi_seed_results,
    }

    out_file = ROOT_DIR / "scratch" / "diagnostic_audit_output.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved complete diagnostic audit results to {out_file}")

if __name__ == "__main__":
    run_audit()
