"""
run_real_training.py

Executes ONE real training run of the Spatio-Temporal GNN Forecasting Anomaly Detection Model
on BATADAL data according to the approved redesign.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from src.detection.train import train_model

def main():
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=================================================================")
    print("      REAL BATADAL FORECASTING ANOMALY DETECTION PIPELINE       ")
    print("=================================================================")
    print(f"Project Root: {Path.cwd()}")
    print(f"Train File:   {config['detection']['train_path']} (Normal Baseline)")
    print(f"Val File:     {config['detection']['val_path']} (Labeled Validation)")
    print(f"Test File:    {config['detection']['test_path']} [HELD OUT INFERENCE]")
    print(f"Topology INP: {config['paths']['inp_file']}")
    print("-----------------------------------------------------------------")

    start_time = time.time()
    res = train_model(config)
    elapsed_time = time.time() - start_time

    model = res["model"]
    val_metrics = res["validation_metrics"]
    history = res["history"]
    calibrated_threshold = res["calibrated_threshold"]
    num_sensors = res["num_sensors"]
    adj_nonzero = res["adjacency_nonzero_count"]

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n-----------------------------------------------------------------")
    print("                     TRAINING PIPELINE LOG                       ")
    print("-----------------------------------------------------------------")
    print(f"Model Architecture:          {model.__class__.__name__} (Predictive Forecasting)")
    print(f"Trainable Parameter Count:   {num_params:,}")
    print(f"Mapped Sensors:              {num_sensors}/43")
    print(f"Graph Adjacency Entries:     {adj_nonzero} non-zero entries (<=3-hop weighted)")
    print(f"Training Time:               {elapsed_time:.2f} seconds")
    print(f"Total Epochs:                {len(history['train_loss'])}")
    print(f"Saved Checkpoint:            {res['checkpoint_path']}")
    print(f"Calibrated Threshold (99.5%): {calibrated_threshold:.6f}")

    print("\n-----------------------------------------------------------------")
    print("           EPOCH-BY-EPOCH MSE LOSS (TRAIN vs CALIBRATION)        ")
    print("-----------------------------------------------------------------")
    for ep, (t_loss, v_loss) in enumerate(zip(history["train_loss"], history["val_loss"]), 1):
        print(f"Epoch {ep:2d}/{len(history['train_loss'])} | Train MSE Loss: {t_loss:.6f} | Calib MSE Loss: {v_loss:.6f}")

    print("\n-----------------------------------------------------------------")
    print("             VALIDATION EVALUATION METRICS                       ")
    print("             (Evaluated on BATADAL_dataset04.csv)                ")
    print("-----------------------------------------------------------------")
    for k, v in val_metrics.items():
        if k == "confusion_matrix":
            print(f"  {k:<25}: {v}")
        elif isinstance(v, float):
            print(f"  {k:<25}: {v:.6f}")
        else:
            print(f"  {k:<25}: {v}")

    report = {
        "pipeline": "X-Resilience-WDS Spatio-Temporal GNN Forecasting Anomaly Detection",
        "dataset_files": {
            "train": config["detection"]["train_path"],
            "validation": config["detection"]["val_path"],
            "test": config["detection"]["test_path"],
        },
        "model_architecture": model.__class__.__name__,
        "num_trainable_parameters": num_params,
        "mapped_sensors_count": num_sensors,
        "adjacency_nonzero_count": adj_nonzero,
        "training_time_seconds": round(elapsed_time, 2),
        "epochs": len(history["train_loss"]),
        "checkpoint_path": res["checkpoint_path"],
        "calibrated_threshold": calibrated_threshold,
        "epoch_losses": [
            {"epoch": ep, "train_mse_loss": round(float(t), 6), "calib_mse_loss": round(float(v), 6)}
            for ep, (t, v) in enumerate(zip(history["train_loss"], history["val_loss"]), 1)
        ],
        "validation_metrics": val_metrics,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "training_execution_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n-----------------------------------------------------------------")
    print(f"Report saved to: {report_file.resolve()}")
    print("=================================================================")

if __name__ == "__main__":
    main()
