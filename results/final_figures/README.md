# Publication Figure Index & Interpretation Guide

This directory contains high-resolution publication figures generated for the `X-Resilience-WDS` research paper and technical documentation.

---

## Figure Index

| Figure Filename | Title / Topic | Dataset / Source Experiment | Key Interpretation & Scientific Caveats |
|---|---|---|---|
| **`fig1_dataset04_validation.png`** | Dataset04 Ground-Truth Validation | `BATADAL_dataset04.csv` Supervised Evaluation | Displays time-series anomaly predictions, PR curve (PR-AUC = $0.6769$), and confusion matrix (F1 = $0.6578$, Recall = $0.7900$). *Caveat*: Evaluated on labeled simulation benchmark data. |
| **`fig2_h1_vs_h12_horizon.png`** | Forecast Horizon Comparison ($H=1$ vs $H=12$) | Multi-Horizon Horizon Error Ablation | Demonstrates forecasting error growth and detection performance across horizons $H=1, 3, 6, 12, 24$. Highlights $H=12$ balance between context window and drift. |
| **`fig3_capacity_controlled_comparison.png`** | Capacity-Controlled Horizon Experiment | Parameter-Matched Multi-Seed Experiment | Compares $H=1$ ($95,164$ params) vs $H=12$ ($95,236$ params) across 5 random seeds ($99.75\%$ parameter match). Proves $H=12$ recall gains stem from multi-step forecasting rather than model capacity. *Caveat*: $H=12$ has higher FPR ($0.0341$ vs $0.0195$). |
| **`fig4_test_anomaly_timeline.png`** | Held-Out Test Anomaly Timeline | `BATADAL_test_dataset.csv` Unsupervised Inference | Visualizes raw and log-scaled anomaly scores across $2,066$ test windows (`2017-01-04` $\to$ `2017-03-31`) with threshold $\tau_{99.8} = 3.013382$ and $17$ detected episodes. *Caveat*: Test dataset is unlabeled; points represent statistical anomaly detections. |
| **`fig5_sensor_attribution.png`** | Sequence 3 & Test Sensor Attribution | Sequence 3 Attribution Deep Dive | Heatmap breakdown of top sensor residual drivers (`L_T2`, `P_J422`, `F_PU3`) mapped to physical C-Town network components. *Caveat*: Sensor attribution indicates statistical residual magnitude, NOT proven physical attack causality. |
| **`fig6_high_severity_episode.png`** | High-Severity Anomaly Episode Profile | Test Episode EP-12 Analysis | Detailed timeline of EP-12 (`2017-02-11 14:00` $\to$ `2017-02-13 18:00`, 53 h) showing pump PU3 score amplification ($\sim 1.13 \times 10^{10}$). *Caveat*: PU3 was inactive in normal calibration data; high scores represent valid mathematical state-transition z-score amplification. |

---

## Publication Usage Notes

- All figures are saved as 300 DPI PNGs formatted for two-column academic paper layouts.
- Figures must be cited in text alongside non-alarmist terminology guidelines.
