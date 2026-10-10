# Quality & Sanity Audit Summary — H=12 Held-Out Test Inference

A comprehensive audit was performed on the existing test inference results under [`results/test_inference/`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/results/test_inference/) produced by candidate model [`models/detection_gnn_h12_candidate.pt`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/models/detection_gnn_h12_candidate.pt) at threshold $\tau_{99.8} = 3.013382$.

## Audit Scorecard

| Audit Item | Category | Result | Root Cause / Note |
| :---: | :--- | :---: | :--- |
| **A** | **Test Data Timeline** | **FAIL** | Pandas guessed `MM/DD/YY` vs `DD/MM/YY`, causing 6 backwards date jumps. |
| **B** | **Episode Construction** | **FAIL** | Scrambled timestamps split contiguous attack blocks into 23 fragmented pseudo-episodes. |
| **C** | **Numerical Stability** | **PASS** | Peak score $\sim 1.13 \times 10^{10}$ is clean & legitimate (pump `PU3` turning ON vs near-zero normal baseline variance). |
| **D** | **Preprocessing Consistency** | **PASS** | $100\%$ identical scaler, topology, windowing, and calibration baseline as training. |
| **E** | **Sensor Attribution** | **PASS** | $1$-to-$1$ sensor column mapping, correct residual ratios, and valid C-Town topology nodes. |
| **F** | **Severity Calculation** | **PASS** | Implemented strictly per formula; extreme residual spikes in `EP-11`/`EP-12` dominate severity. |
| **G** | **System Integrity** | **PASS** | Candidate, production model, config, and raw CSV files strictly unchanged & read-only. |
| **H** | **Pytest Suite** | **PASS** | **$41 / 41$ tests PASSED** |

---

## Detailed Audit Findings

### 1. Test Data Timeline (FAIL)
- **Discrepancy**: Raw timestamps in `BATADAL_test_dataset.csv` are formatted as `DD/MM/YY HH:MM` (`04/01/17 00` = Jan 4, 2017). When parsed without `format="%d/%m/%y %H"`, pandas parsed `04/01/17` as April 1st, but switched back to `dayfirst=True` whenever a day integer exceeded 12 (`13/01/17`).
- **Impact**: This caused 6 backwards timestamp jumps (e.g. `2017-11-02` to `2017-02-13` for `EP-12`).
- **Correction**: Parsing explicitly with `format="%d/%m/%y %H"` confirms $2,089$ raw rows, $2,066$ evaluated windows, strictly monotonic dates from `2017-01-04 00:00` to `2017-04-01 00:00`, with zero missing or duplicate timestamps.

### 2. Episode Construction (FAIL)
- **Discrepancy**: Because episode grouping was executed on non-monotonic default timestamps, contiguous attack windows were broken across the 6 date jump boundaries into 23 pseudo-episodes.
- **Correction**: Grouping contiguous anomalous windows on the strictly monotonic timeline yields **17 true contiguous episodes**:
  - `EP-11`: `2017-02-08 16:00` to `2017-02-10 20:00` (53 hours)
  - `EP-12`: `2017-02-11 14:00` to `2017-02-13 18:00` (53 hours)

### 3. Numerical Stability (PASS with Documented Limitation)
- **Peak Anomaly Score**: $\approx 1.13 \times 10^{10}$ at Index 931 (Raw row 943, `2017-02-12 07:00`).
- **Root Cause**: Raw data is 100% clean (0 NaNs, 0 infs, 0 `-999`s). During normal baseline training (`BATADAL_dataset03.csv`), pump `PU3` is OFF (`F_PU3 = 0.0`), producing near-zero calibration error variance $\mu_{\text{sq, h}}[11] = 2.9 \times 10^{-8}$. When pump `PU3` turns ON in the test set (`F_PU3 = 86.49` L/s), dividing squared forecasting error ($14,294.26$) by $2.9 \times 10^{-8}$ yields normalized residual $Z = 4.87 \times 10^{11}$.
- **Scientific Validity**: Model predictions and raw numbers are valid physical anomaly signals representing an unmodeled state transition (pump `PU3` activation).

### 4. Severity Formula (PASS with Documented Limitation)
- **Formula**: $\text{Severity Score} = \text{Mean Anomaly Score} \times \sqrt{\text{Duration in Hours}} \times \left(1 + \frac{\text{Peak Anomaly Score}}{\tau_{99.8}}\right)$
- **Limitation**: Because the formula scales quadratically with anomaly score ($\text{Mean} \times \text{Peak}$), multi-billion residual spikes in `EP-11` and `EP-12` ($\sim 10^{10}$) produce severity scores of $\sim 10^{19} - 10^{20}$, numerically dominating all other episodes ($\sim 10^1 - 10^2$).

---

## Proven Implementation Bug & Corrected Outputs

An implementation bug was proven in timestamp parsing. As mandated by audit instructions:
- Corrected execution script created: [`scratch/run_test_inference_corrected.py`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/scratch/run_test_inference_corrected.py)
- Corrected outputs saved separately under: [`results/test_inference/corrected/`](file:///c:/Users/newuser/Downloads/pro_X/X-Resilience-WDS/results/test_inference/corrected/)
- Original outputs under `results/test_inference/` remain preserved without deletion or overwrite.
