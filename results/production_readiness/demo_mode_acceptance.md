# X-Resilience-WDS: Presentation Demo Mode Acceptance Report

This document records the design, implementation, interactive test verification, data isolation audit, and SHA-256 integrity verification for the **Presentation Demo Mode** on the `X-Resilience-WDS` project.

---

## 1. Demo Mode Architecture & Overview

- **Purpose**: Provides a live presentation demonstration tab ("🎬 Live Demonstration / What-If Simulation") inside the Streamlit dashboard (`dashboard/app.py`).
- **Model Execution**: Operates strictly in-memory using the promoted production $H=12$ SpatioTemporalGNN model (`models/detection_gnn.pt`) in read-only evaluation mode.
- **Operational Threshold**: $\tau_{99.8} = 3.013382$ (Derived strictly from Dataset03 normal calibration).
- **Data Isolation Policy**: In-memory array transformations only. Zero modifications or writes to `data/raw/`, `results/test_inference/`, `models/`, or `configs/config.yaml`.
- **Scientific Terminology Notice**: Uses non-alarmist scientific terminology ("Demonstration Anomaly", "Simulated Operational Disturbance", "Statistical Anomaly Detected", "Affected Sensor/Component"). Never asserts unverified cyberattack mechanisms.

---

## 2. Monitored Sensor & Component Categories (43 Total Sensors)

All 43 monitored sensors were discovered and categorized directly from `data/raw/BATADAL_dataset03.csv` and C-Town topology mapping:

| Component Category | Total Monitored Sensors | Available in Detector | Demo Supported | Example Monitored Sensors |
| :--- | :---: | :---: | :---: | :--- |
| **Junction Pressure** | 12 | **YES** | **YES** | `P_J280`, `P_J269`, `P_J300`, `P_J256`, `P_J289` |
| **Tank Level** | 7 | **YES** | **YES** | `L_T1`, `L_T2`, `L_T3`, `L_T4`, `L_T5`, `L_T6`, `L_T7` |
| **Pump Flow** | 11 | **YES** | **YES** | `F_PU1`, `F_PU2`, `F_PU3`, `F_PU4`, `F_PU5` ... `F_PU11` |
| **Pump Status** | 11 | **YES** | **YES** | `S_PU1`, `S_PU2`, `S_PU3`, `S_PU4`, `S_PU5` ... `S_PU11` |
| **Valve Flow** | 1 | **YES** | **YES** | `F_V2` |
| **Valve Status** | 1 | **YES** | **YES** | `S_V2` |

---

## 3. Live Demonstration Test Results Across 6 Component Categories

Interactive simulations were performed on the promoted $H=12$ production model across all 6 categories:

| Component Category | Target Sensor | Disturbance Type | Baseline Score | Simulated Score | Operational Threshold | Model Response | Top Residual Attribution |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Junction Pressure** | `P_J280` | Sudden Increase | $0.6110$ | **$1.1432$** | $3.013382$ | 🟢 Below Threshold | `L_T3` ($12.1\times$), `S_PU8` ($4.7\times$) |
| **Tank Level** | `L_T1` | Level Decrease | $0.6110$ | **$1.4848$** | $3.013382$ | 🟢 Below Threshold | `S_PU5` ($11.0\times$), `L_T3` ($9.3\times$) |
| **Pump Flow** | `F_PU6` | Flow Increase | $0.6110$ | **$0.8662$** | $3.013382$ | 🟢 Below Threshold | `L_T5` ($6.5\times$), `S_PU11` ($4.9\times$) |
| **Pump Status** | `S_PU3` | State Transition (OFF $\rightarrow$ ON) | $0.6110$ | **$0.5079$** | $3.013382$ | 🟢 Below Threshold | `F_PU3` ($3.4\times$), `S_PU11` ($2.1\times$) |
| **Valve Flow** | `F_V2` | Flow Decrease | $0.6110$ | **$0.5445$** | $3.013382$ | 🟢 Below Threshold | `F_PU3` ($3.8\times$), `S_PU11` ($2.8\times$) |
| **Valve Status** | `S_V2` | State Transition (CLOSED $\rightarrow$ OPEN) | $0.6110$ | **$0.6683$** | $3.013382$ | 🟢 Below Threshold | `S_PU11` ($4.5\times$), `F_PU11` ($4.3\times$) |

---

## 4. Unsensed Physical Topology Component Handling

- **EPANET Node Inspection Mode**: Allows interactive inspection of physical EPANET network components (e.g. junction `J511`).
- **Scientific Guardrail Warning**:
  > ⚠️ **UNSENSED PHYSICAL TOPOLOGY COMPONENT DETECTED**
  > Physical component `J511` is NOT directly observed by the 43-sensor detector. A direct anomaly score cannot be claimed for this component.
- **Topological Association**: If indirect/topological sensor associations exist, they are explicitly labeled as `Topology-Associated Monitored Sensors (Indirect/topological association only — direct detection is NOT claimed)`.

---

## 5. Maintenance of Existing 6 Dashboard Tabs

All 6 existing research tabs remain **100% UNCHANGED and OPERATIONAL**:
1. **Executive Overview**: 2,066 Evaluated Windows | 156 Anomalous Windows (7.55%) | 17 Contiguous Episodes | $\tau = 3.013382$
2. **Anomaly Timeline**: 2,066 monotonic hourly timesteps | Log10 display working
3. **Episode Explorer**: 17 episodes | Zero `ValueError` formatting issues | EP-12, EP-11, EP-20, EP-06 verified
4. **Sensor Attribution**: PU3 near-zero baseline variance warning active
5. **Topology Mapping**: C-Town EPANET network node mapping active
6. **Ground-Truth Validation**: Dataset04 metrics preserved (Precision 0.5635, Recall 0.7900, F1 0.6578, ROC-AUC 0.9685, PR-AUC 0.6769, FPR 0.0341)

---

## 6. Pytest Automated Test Verification

Full test suite execution:
```bash
.venv\Scripts\python.exe -m pytest -v
```

**Result**: **53 / 53 PASSED (100% Pass Rate)**

Added Demo Mode tests:
- `test_demo_uses_locked_threshold` (PASSED)
- `test_demo_pipeline_sensor_categories_discoverable` (PASSED)
- `test_demo_synthetic_injection_in_memory_only` (PASSED)

---

## 7. SHA-256 File Integrity Audit

| Asset Description | File Path | Expected SHA-256 Hash | Verified SHA-256 Hash | Integrity |
| :--- | :--- | :--- | :--- | :---: |
| **Production Model** | `models/detection_gnn.pt` | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | **PASS** |
| **Candidate Model** | `models/detection_gnn_h12_candidate.pt` | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | `8e85de2b871ff49bb0894f2eb97e3d9fd6087151e8f0ba7ee35006f890a93667` | **PASS** |
| **Rollback Checkpoint** | `models/backups/detection_gnn_pre_h12_promotion_20260914_192026.pt` | `056284e983baf988d962e4d56d7ea7528c4638173f081c418ec9f6ce907d9e64` | `056284e983baf988d962e4d56d7ea7528c4638173f081c418ec9f6ce907d9e64` | **PASS** |
| **Config File** | `configs/config.yaml` | `fe88ac78daf2fef10c476d91bc35e361f586d35b5d8de7c1f6a27fc925160900` | `fe88ac78daf2fef10c476d91bc35e361f586d35b5d8de7c1f6a27fc925160900` | **PASS** |
| **Raw Dataset 03** | `data/raw/BATADAL_dataset03.csv` | `8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b` | `8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b` | **PASS** |
| **Raw Dataset 04** | `data/raw/BATADAL_dataset04.csv` | `4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b` | `4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b` | **PASS** |
| **Raw Test Dataset** | `data/raw/BATADAL_test_dataset.csv` | `184dfc1581df52193b2acb79164f7ce094f2bb00b1bc2e3c2e70050095065fa5` | `184dfc1581df52193b2acb79164f7ce094f2bb00b1bc2e3c2e70050095065fa5` | **PASS** |

---

## 8. Final Presentation Demo Status

**FINAL DEMO STATUS**: **READY**
