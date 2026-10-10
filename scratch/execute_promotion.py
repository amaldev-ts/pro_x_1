"""
scratch/execute_promotion.py

Atomic Promotion Verification Script for H=12 SpatioTemporalGNN Candidate Checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

models_dir = ROOT_DIR / "models"
backups_dir = models_dir / "backups"
backup_path = backups_dir / "detection_gnn_pre_h12_promotion_20260914_192026.pt"

prod_path = models_dir / "detection_gnn.pt"
cand_path = models_dir / "detection_gnn_h12_candidate.pt"
cfg_path = ROOT_DIR / "configs" / "config.yaml"
ds03_path = ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv"
ds04_path = ROOT_DIR / "data" / "raw" / "BATADAL_dataset04.csv"
test_path = ROOT_DIR / "data" / "raw" / "BATADAL_test_dataset.csv"

def get_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_and_finish_promotion():
    print("=" * 80)
    print("PHASE 1 & 2: VERIFY PRE-PROMOTION BACKUP & PROMOTION CHECKPOINT")
    print("=" * 80)

    exp_cand_hash = "8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667"
    exp_prod_hash = "056284e983baf988d962e4d56d7ea7528c4638173f081c418ec9f6ce907d9e64"

    assert backup_path.exists(), f"Backup file missing: {backup_path}"
    backup_hash = get_hash(backup_path)
    assert backup_hash == exp_prod_hash, f"Backup hash mismatch: {backup_hash} != {exp_prod_hash}"
    print(f"-> Pre-promotion Backup VERIFIED: {backup_path}")
    print(f"   Original Production SHA256: {backup_hash} (100% MATCH)")

    # Verify atomic replacement
    prod_hash_post = get_hash(prod_path)
    cand_hash_post = get_hash(cand_path)
    cfg_hash_post = get_hash(cfg_path)
    ds03_hash_post = get_hash(ds03_path)
    ds04_hash_post = get_hash(ds04_path)
    test_hash_post = get_hash(test_path)

    print(f"-> Promoted Production SHA256 : {prod_hash_post}")
    print(f"-> Candidate SHA256           : {cand_hash_post}")
    print(f"-> Config SHA256              : {cfg_hash_post}")

    assert prod_hash_post == exp_cand_hash, f"Production post hash mismatch: {prod_hash_post}"
    assert cand_hash_post == exp_cand_hash, f"Candidate post hash mismatch: {cand_hash_post}"
    print("-> Post-promotion SHA256 hashes VERIFIED 100% MATCH.")

    print("\n" + "=" * 80)
    print("PHASE 4: PRODUCTION CHECKPOINT LOAD TEST")
    print("=" * 80)

    from src.detection.graph_model import SpatioTemporalGNN

    checkpoint = torch.load(prod_path, weights_only=False)
    print(f"Loaded checkpoint keys: {list(checkpoint.keys())}")
    forecast_horizon = checkpoint.get("forecast_horizon", 12)
    print(f"Forecast Horizon H: {forecast_horizon}")
    assert forecast_horizon == 12, f"Expected H=12, got H={forecast_horizon}"

    model = SpatioTemporalGNN(in_features=43, window_size=12, hidden_dim=64, forecast_horizon=12)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(2, 12, 43)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Input Shape  : {dummy_input.shape} (Batch=2, History W=12, Features F=43)")
    print(f"Model Output Shape : {output.shape} (Expected: torch.Size([2, 12, 43]))")
    assert output.shape == (2, 12, 43), f"Unexpected output shape: {output.shape}"
    print("-> Production Checkpoint Load Test PASSED 100%.")

    summary = {
        "promotion_timestamp": "20260914_192026",
        "backup_path": str(backup_path),
        "original_production_sha256": exp_prod_hash,
        "candidate_sha256": exp_cand_hash,
        "final_production_sha256": prod_hash_post,
        "backup_sha256": backup_hash,
        "candidate_post_promotion_sha256": cand_hash_post,
        "config_sha256": cfg_hash_post,
        "raw_dataset03_sha256": ds03_hash_post,
        "raw_dataset04_sha256": ds04_hash_post,
        "raw_test_dataset_sha256": test_hash_post,
        "checkpoint_load_result": "PASS (Loaded H=12 model successfully)",
        "model_architecture": "SpatioTemporalGNN",
        "forecast_horizon": 12,
        "threshold_tau": 3.013382,
        "status": "PROMOTION SUCCESSFUL"
    }

    report_dir = ROOT_DIR / "results" / "production_readiness"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "h12_promotion_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PROMOTION VERIFIED AND SUMMARY CREATED!")
    print("=" * 80)
    return summary

if __name__ == "__main__":
    verify_and_finish_promotion()
