Measured by baselines/run_baselines.py (parameters tuned on the training split only).

Chosen parameters: {"B1_mode_aware_z": {"k": 8, "persist_min": 10}, "B2_cusum": {"kappa": 2.0, "h": 120}, "B3_isolation_forest": {"train_quantile": 0.995, "persist_min": 10}}

| Baseline | Split | Composite | Event F1 | Recall | Precision | False + excess alerts / asset-day | Timeliness | Fault-type recall | Class recall | Median lead to deadline (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0_scada_alarms | train | 33.2 | 0.39 | 0.33 | 0.46 | 0.25 | 0.31 | 0.06 | 0.33 | -0.0 |
| B0_scada_alarms | validation | 37.8 | 0.40 | 0.25 | 1.00 | 0.00 | 0.25 | 0.25 | 0.25 | - |
| B1_mode_aware_z | train | 60.2 | 0.73 | 0.61 | 0.92 | 0.04 | 0.50 | 0.39 | 0.50 | 36.0 |
| B1_mode_aware_z | validation | 60.8 | 0.67 | 0.50 | 1.00 | 0.00 | 0.50 | 0.50 | 0.50 | 10.7 |
| B2_cusum | train | 55.1 | 0.55 | 0.78 | 0.42 | 0.68 | 0.73 | 0.28 | 0.78 | 10.1 |
| B2_cusum | validation | 66.0 | 0.62 | 1.00 | 0.44 | 0.71 | 0.78 | 0.50 | 1.00 | 20.8 |
| B3_isolation_forest | train | 32.2 | 0.36 | 0.22 | 1.00 | 0.00 | 0.22 | 0.11 | 0.22 | 108.6 |
| B3_isolation_forest | validation | 37.8 | 0.40 | 0.25 | 1.00 | 0.00 | 0.25 | 0.25 | 0.25 | - |
