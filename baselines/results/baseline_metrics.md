Measured by baselines/run_baselines.py (parameters tuned on the training split only).

Chosen parameters: {"B1_mode_aware_z": {"k": 8, "persist_min": 10}, "B2_cusum": {"kappa": 2.0, "h": 120}, "B3_isolation_forest": {"train_quantile": 0.995, "persist_min": 10}}

| Baseline | Split | Composite | Event F1 | Recall | Precision | False + excess alerts / asset-day | Timeliness | Fault-type recall | Class recall | Median lead to deadline (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_scada_alarms | train | 27.6 | 0.31 | 0.23 | 0.46 | 0.14 | 0.22 | 0.04 | 0.23 | -0.0 |
| B0_scada_alarms | validation | 23.1 | 0.20 | 0.11 | 1.00 | 0.00 | 0.11 | 0.11 | 0.11 | - |
| B1_mode_aware_z | train | 62.0 | 0.73 | 0.69 | 0.78 | 0.10 | 0.57 | 0.38 | 0.58 | 31.9 |
| B1_mode_aware_z | validation | 72.9 | 0.82 | 0.78 | 0.88 | 0.07 | 0.78 | 0.44 | 0.78 | 5.5 |
| B2_cusum | train | 50.3 | 0.47 | 0.85 | 0.33 | 0.92 | 0.77 | 0.19 | 0.81 | 10.0 |
| B2_cusum | validation | 60.0 | 0.51 | 1.00 | 0.35 | 1.21 | 0.90 | 0.44 | 0.89 | 10.6 |
| B3_isolation_forest | train | 41.5 | 0.50 | 0.38 | 0.71 | 0.08 | 0.34 | 0.15 | 0.38 | 108.7 |
| B3_isolation_forest | validation | 45.4 | 0.44 | 0.44 | 0.44 | 0.36 | 0.44 | 0.33 | 0.44 | -0.1 |
