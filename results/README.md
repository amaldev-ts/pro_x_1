# X-Resilience-WDS Results & Research Artifact Index

This directory contains all experimental output files, validation reports, ablation study JSONs, test inference deliverables, publication figures, and production-readiness audit reports for the `X-Resilience-WDS` spatio-temporal GNN anomaly detection system.

---

## Directory Overview

```text
results/
├── final_figures/                  # High-resolution publication figures (fig1 - fig6)
├── test_inference/                 # Held-out test set evaluation deliverables
│   └── corrected/                  # Monotonic timeline corrected test inference results
├── production_readiness/           # Production readiness review, live dashboard acceptance, & promotion reports
└── [Ablation & Experiment Reports] # Benchmark JSONs, CSVs, and ablation summary figures
```

---

## 1. Primary Benchmark & Final Summary Reports

- `final_results_summary.json` & `final_results_summary.md`: Consolidated technical summary of final $H=12$ model architecture, Dataset04 validation metrics, sequence-level performance, and held-out test set findings.
- `final_h12_validation_report.json`: Detailed Dataset04 ground-truth validation results, confusion matrix, threshold tuning curves, and per-sequence detection delays.
- `final_h12_training_report.json`: Execution log, epoch loss trajectories, and training metadata for the final $H=12$ model checkpoint.
- `calibrated_threshold.json`: Operational anomaly threshold ($\tau_{99.8} = 3.013382$) calibrated on the Dataset03 normal operational partition.

---

## 2. Held-Out Test Inference Deliverables (`results/test_inference/corrected/`)

- `test_anomaly_scores.csv`: Per-window anomaly scores ($2,066$ evaluated $H=12$ windows), anomaly flags, UI log display scores, and top sensor drivers.
- `test_anomaly_episodes.csv`: $17$ contiguous anomaly episodes detailing start/end timestamps, durations, peak/mean scores, raw severity, and operational log severity.
- `test_sensor_attribution.csv`: Per-timestep residual contributions across all $43$ sensors.
- `test_inference_report.json`: Summary metadata for test set inference.

---

## 3. Production Readiness & Promotion Reports (`results/production_readiness/`)

- `production_readiness_report.md` & `.json`: Comprehensive production-readiness review evaluating data integrity, timestamp parsing, PU3 near-zero baseline variance, and dashboard readiness.
- `live_dashboard_acceptance.md` & `.json`: Live server acceptance test report for the Streamlit dashboard (`dashboard/app.py`).
- `h12_promotion_report.md` & `.json`: Promotion audit report documenting pre/post SHA-256 checkpoint verification, timestamped backup creation, and load testing.
- `pu3_extreme_score_analysis.csv`: Baseline variance analysis of pump PU3 flow (`F_PU3`) and state (`S_PU3`).
- `severity_analysis.csv`: Comparison of linear raw severity vs log-scaled operational display severity across test anomaly episodes.
- `dashboard_readiness_checklist.csv`: Feature completeness checklist for the Phase 8 dashboard.

---

## 4. Controlled Experiments & Ablation Studies

### A. Capacity-Controlled Experiment ($H=1$ vs $H=12$)
- `capacity_control_report.json`, `capacity_control_summary.csv`, `capacity_control_metrics.csv`
- `capacity_control_f1.png`, `capacity_control_fpr.png`, `capacity_control_seq3.png`
- *Findings*: Parameter-matched comparison ($95,164$ vs $95,236$ parameters) proving $H=12$ yields superior recall and sequence-level detection through multi-horizon forecasting rather than parameter capacity.

### B. Multi-Horizon Horizon Error Ablation
- `multihorizon_ablation_report.json`, `multihorizon_metrics.csv`, `multihorizon_horizon_error.csv`
- `multihorizon_training_curves.png`, `multihorizon_error_growth.png`, `multihorizon_sequence3_error_growth.png`
- *Findings*: Evaluation of forecasting error growth across horizons $H=1, 3, 6, 12, 24$.

### C. Multi-Seed Robustness Evaluation
- `multiseed_robustness_report.json`, `multiseed_summary.csv`, `multiseed_robustness_metrics.csv`
- `multiseed_training_stability.png`, `multiseed_f1_comparison.png`, `multiseed_fpr_comparison.png`, `multiseed_seq3_comparison.png`
- *Findings*: 5-seed statistical distribution confirming stability of F1 ($0.6578 \pm 0.012$) and ROC-AUC ($0.9685 \pm 0.003$).

### D. Threshold & Normalization Ablations
- `h12_threshold_ablation_report.json`, `h12_threshold_ablation.csv`, `h12_threshold_comparison.png`
- `sensor_normalized_ablation_report.json`
- `anomaly_score_ablation_report.json`

### E. Sequence 3 & Topology Attribution Deep Dives
- `sequence3_sensor_attribution_report.json`, `sequence3_sensor_attribution.csv`, `sequence3_temporal_analysis.csv`
- `seq3_top_sensor_trajectories.png`, `seq3_instantaneous_vs_rolling.png`, `seq3_elevated_sensors_over_time.png`
- `topology_regional_ablation_report.json`, `topology_regional_scores.csv`, `topology_region_definition.json`
- `ctown_topology.png`, `ctown_topology_dma.png`, `ctown_topology_louvain.png`

---

## 5. Publication Figures (`results/final_figures/`)

High-resolution publication figures generated for academic paper submission. See [`results/final_figures/README.md`](final_figures/README.md) for individual figure descriptions and scientific interpretation notes.
