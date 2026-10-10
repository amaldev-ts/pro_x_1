# Final Held-Out Test Inference Summary

Execution of Phase 5 held-out test inference on `BATADAL_test_dataset.csv` using frozen candidate model (`models/detection_gnn_h12_candidate.pt`) and locked operational threshold $\tau_{99.8} = 3.013382$.

## Key Findings

- **Total Test Timesteps / Windows**: 2,066
- **Detected Anomalous Windows**: **156** (7.55%)
- **Total Anomaly Episodes Detected**: **23**
- **Maximum Anomaly Score**: **11329609728.0000**
- **Mean Anomaly Score**: **212974832.0000**

## Detected Anomaly Episodes (Ranked by Severity)

| Rank | Episode ID | Start Timestamp | End Timestamp | Duration (h) | Max Score | Mean Score | Severity Score | Dominant Sensors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `EP-12` | `2017-11-02 14:00` | `2017-02-13 18:00` | 53.0h | 11329609728.00 | 4336120832.00 | **118686157875098140672.00** | `F_PU3 (Link Flow): 186379796480.0x` |
| 2 | `EP-11` | `2017-08-02 16:00` | `2017-10-02 20:00` | 53.0h | 7384827904.00 | 3965879808.00 | **70756065014497681408.00** | `F_PU3 (Link Flow): 170476797952.0x` |
| 3 | `EP-20` | `2017-07-03 13:00` | `2017-07-03 14:00` | 2.0h | 40.30 | 39.18 | **796.39** | `S_PU6 (Link Status): 891.1x` |
| 4 | `EP-06` | `2017-01-30 01:00` | `2017-01-30 18:00` | 18.0h | 8.44 | 5.37 | **86.64** | `L_T2 (Tank Level): 33.7x` |
| 5 | `EP-16` | `2017-02-25 16:00` | `2017-02-25 18:00` | 3.0h | 6.79 | 5.44 | **30.67** | `P_J289 (Junction Pressure): 77.2x` |
| 6 | `EP-18` | `2017-02-26 09:00` | `2017-02-26 10:00` | 2.0h | 5.35 | 5.24 | **20.57** | `P_J280 (Junction Pressure): 36.2x` |
| 7 | `EP-21` | `2017-08-03 02:00` | `2017-08-03 07:00` | 6.0h | 3.96 | 3.61 | **20.47** | `F_PU1 (Link Flow): 16.6x` |
| 8 | `EP-07` | `2017-01-30 20:00` | `2017-01-30 21:00` | 2.0h | 4.87 | 4.80 | **17.76** | `P_J289 (Junction Pressure): 36.3x` |
| 9 | `EP-08` | `2017-01-31 05:00` | `2017-01-31 07:00` | 3.0h | 3.57 | 3.30 | **12.48** | `L_T2 (Tank Level): 27.3x` |
| 10 | `EP-05` | `2017-01-29 22:00` | `2017-01-29 22:00` | 1.0h | 3.78 | 3.78 | **8.51** | `P_J289 (Junction Pressure): 38.1x` |
| 11 | `EP-22` | `2017-03-24 12:00` | `2017-03-24 12:00` | 1.0h | 3.77 | 3.77 | **8.50** | `P_J269 (Junction Pressure): 23.3x` |
| 12 | `EP-17` | `2017-02-25 21:00` | `2017-02-25 21:00` | 1.0h | 3.77 | 3.77 | **8.47** | `P_J289 (Junction Pressure): 52.1x` |
| 13 | `EP-19` | `2017-04-03 13:00` | `2017-04-03 13:00` | 1.0h | 3.65 | 3.65 | **8.07** | `P_J280 (Junction Pressure): 35.6x` |
| 14 | `EP-02` | `2017-09-01 14:00` | `2017-09-01 14:00` | 1.0h | 3.55 | 3.55 | **7.73** | `P_J280 (Junction Pressure): 26.7x` |
| 15 | `EP-15` | `2017-02-24 14:00` | `2017-02-24 14:00` | 1.0h | 3.50 | 3.50 | **7.57** | `P_J280 (Junction Pressure): 37.3x` |
| 16 | `EP-09` | `2017-01-31 11:00` | `2017-01-31 11:00` | 1.0h | 3.47 | 3.47 | **7.46** | `P_J280 (Junction Pressure): 30.8x` |
| 17 | `EP-10` | `2017-01-02 07:00` | `2017-01-02 07:00` | 1.0h | 3.42 | 3.42 | **7.31** | `P_J422 (Junction Pressure): 19.2x` |
| 18 | `EP-14` | `2017-02-24 05:00` | `2017-02-24 05:00` | 1.0h | 3.38 | 3.38 | **7.17** | `P_J269 (Junction Pressure): 17.3x` |
| 19 | `EP-03` | `2017-01-16 19:00` | `2017-01-16 19:00` | 1.0h | 3.20 | 3.20 | **6.61** | `P_J256 (Junction Pressure): 26.9x` |
| 20 | `EP-23` | `2017-03-25 04:00` | `2017-03-25 04:00` | 1.0h | 3.10 | 3.10 | **6.28** | `L_T2 (Tank Level): 21.9x` |
| 21 | `EP-04` | `2017-01-28 22:00` | `2017-01-28 22:00` | 1.0h | 3.06 | 3.06 | **6.18** | `P_J307 (Junction Pressure): 15.5x` |
| 22 | `EP-01` | `2017-05-01 07:00` | `2017-05-01 07:00` | 1.0h | 3.03 | 3.03 | **6.09** | `L_T1 (Tank Level): 35.3x` |
| 23 | `EP-13` | `2017-02-20 12:00` | `2017-02-20 12:00` | 1.0h | 3.03 | 3.03 | **6.06** | `P_J280 (Junction Pressure): 29.7x` |

## Topology & Attribution Context

- **Link Flow & Status Sensors** (`F_PU*`, `S_PU*`) exhibit the largest relative magnitude jumps during broad operational disruptions.
- **Junction Pressure Sensors** (`P_J*`) and **Tank Levels** (`L_T*`) provide localized hydraulic spatial signals during specific localized episodes.

## Integrity Verification

- Candidate model SHA256 pre vs post: **UNCHANGED**
- Production model `models/detection_gnn.pt`: **UNTOUCHED**
- Config `configs/config.yaml`: **UNCHANGED**
- Raw data `data/raw/*`: **UNCHANGED**
- Test set labels: **UNACCESSED / UNTOUCHED**
