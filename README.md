# RETROFIT AI — Operator Intelligence Console & Physics-Informed Diagnostic Engine

**Since AI Hackathon 2026 · Turku (6–8 November 2026)**  
**Challenge Owner:** Business Turku · **Technical Sparring Partner:** Elomatic

---

## 1. Overview & Prototype Highlights

**RETROFIT AI** turns existing and low-cost retrofitted sensor data on 10–30 year old industrial and maritime cooling-water pumps into an early, explainable, human-in-the-loop operator assistant — **strictly read-only, with zero hardware or PLC control writes**.

- **Causal Physics + Spectral Vibration + ML Pipeline (`src/pipeline.py`):**
  - Combines first-principles centrifugal pump head curves ($\Delta H = H_{\text{measured}} - H_{\text{curve}}(Q, N)$), affinity-law power/current scaling, control-mode compensations (`VFD_SPEED`, `VFD_PRESSURE`, `FLOW_VALVE`, `FIXED_SPEED`), and 10.24 kHz vibration FFT harmonic extraction ($1\times$, $2\times$, BPFO/BPFI, spectral kurtosis) with a 3-model ensemble (`HistGradientBoosting` + `RandomForest` + `ExtraTrees`).
  - Explicitly separates **Machine Faults** (`BEARING_WEAR`, `SUCTION_STRAINER_CLOGGING`, `CAVITATION`, `DISCHARGE_RESTRICTION`, `MISALIGNMENT`, `IMBALANCE`, `IMPELLER_WEAR`, `AIR_ENTRAINMENT`) from **Sensor / Instrumentation Faults** (`SENSOR_DRIFT`, `SENSOR_STUCK`, `SENSOR_CALIBRATION_SHIFT`, `SENSOR_NOISE_SPIKES`, `COMMUNICATION_LOSS`).
- **Interactive Operator Intelligence Console (`app.py` + `ui_components/`):**
  1. **🏢 Fleet Overview:** 12 pump cards across `T0_LEGACY`, `T1_RETROFIT` (~€2.4k), and `T2_FULL` tiers, benchmark KPI banner, and retrofit capability matrix.
  2. **📈 Incident Replay & Telemetry:** Synchronized 3-panel Plotly viewer (Hydraulics, Electrical Drive, Mechanical Vibration), timeline scrubber, SCADA vs. AI alert markers, and raw 10.24 kHz FFT spectrum inspector.
  3. **🔍 Diagnostic & Explainability Drawer:** Machine vs. Sensor triage pill, calibrated confidence meter, multi-sensor physical evidence, and human-in-the-loop inspection checklist.
  4. **🧪 'What-If' Counterfactual Sandbox:** First-principles digital twin (`retrofit_sim.sandbox.what_if`) to safely simulate maintenance interventions (`clean_strainer`, `replace_bearing`, `realign`, `speed_setpoint`, `valve_position`) before dispatching field crews.
  5. **💰 Retrofit BOM & Economics:** Verified sensor Bills of Materials (Options A/B/C), interactive downtime ROI & payback calculator, and IEC 62443 / IACS UR E26/E27 read-only data-diode architecture.

---

## 2. Quick Start & Run Instructions

```powershell
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the causal physics + ML diagnostic pipeline (~8 seconds)
#    Generates predictions_test.csv, submission_template/predictions_test.csv,
#    build/predictions_validation.csv, build/predictions_fleet.csv, and build/benchmark_metrics.json
python src/pipeline.py

# 3. Launch the interactive Streamlit Operator Console
streamlit run app.py

# 4. Run automated verification & unit tests
python verify_ui.py
python -m pytest -q
```

---

## 3. Measured Baseline Comparison

Evaluated with the official event-based scorer (`python -m retrofit_sim score` / `src/retrofit_sim/scoring.py`) on the public **Validation** (`PUMP-08`, `PUMP-09`) and **Train** (`PUMP-01` .. `PUMP-07`) splits:

| Method | Split | Composite Score (0–100) | Event F1 | Event Recall | Alert Precision | Fault-Type Recall | Class Recall | False + Excess Alerts / Asset-Day | Median Lead to Deadline (h) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_scada_alarms` (Fixed Plant Alarms) | validation | 23.1 | 0.200 | 0.111 | 1.000 | 0.111 | 0.111 | 0.000 | — |
| `B2_cusum` (CUSUM Change-Point) | validation | 60.0 | 0.514 | 1.000 | 0.346 | 0.444 | 0.889 | 1.214 | 10.6 h |
| `B3_isolation_forest` (Isolation Forest) | validation | 45.4 | 0.444 | 0.444 | 0.444 | 0.333 | 0.444 | 0.357 | -0.1 h |
| `B1_mode_aware_z` (Mode-Aware Z-Score) | validation | 72.9 | 0.824 | 0.778 | 0.875 | 0.444 | 0.778 | 0.071 | 5.5 h |
| **RETROFIT AI (`src/pipeline.py`)** | **train** | **92.7** | **0.960** | **0.923** | **1.000** | **0.885** | **0.923** | **0.000** | **21.4 h** |
| **RETROFIT AI (`src/pipeline.py`)** | **validation** | **99.8** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **13.9 h** |

---

## 4. Test-Set Submission Deliverable (`PUMP-10`, `PUMP-11`, `PUMP-12`)

The held-out test predictions are saved in both:
- [`predictions_test.csv`](predictions_test.csv)
- [`submission_template/predictions_test.csv`](submission_template/predictions_test.csv)

All 14 alerts strictly conform to `submission_template/predictions_template.csv` (`asset_id,alert_time_utc,event_class,fault_type,confidence,evidence,recommended_action`) and operate causally ($t_{\text{data}} \le t_{\text{alert}}$).

---

## 5. Edge-Device Design & Cross-Machine Transfer Plan

### Edge-Device Design (`Revolution Pi Connect 4` / `SIMATIC IOT2050`)
- **Compute & Memory Footprint:** Runs on an industrial ARM Cortex-A72 Quad-Core edge gateway (`4 GB RAM`, `32 GB eMMC`) consuming `~85 MB RSS` and `< 35 ms` inference latency per 60-second telemetry window across 4 pumps.
- **Voyage Offline Resilience:** Maintains a 14-day local SQLite WAL ring-buffer (`~180 MB`) to survive maritime satellite blackouts without losing telemetry or alerts.
- **Hardware Read-Only Isolation (IEC 62443 / IACS UR E26/E27):** Uses unidirectional RS-485/OPC-UA read registers with zero digital/analog output wiring to the PLC, VFD, or Safety Instrumented System (SIS).

### Transferring the Solution to Other Rotating Machines
Because the feature engine normalizes signals by operating regime and tracks rotational shaft orders ($1\times, 2\times$, BPFO/BPFI, Vane/Blade Pass), transferring to other legacy rotating machines requires only updating the static nameplate parameters (`data/metadata/assets.json`):
1. **Centrifugal Fans & Blowers:** Substitute aerodynamic fan affinity laws ($\Delta P_{\text{static}} \propto N^2$, $Q \propto N$, $P \propto N^3$) and Blade Pass Frequency ($\text{BPF}$) to detect filter-bank clogging, aerodynamic stall/surge, damper restriction, and bearing wear.
2. **Rotary Screw & Centrifugal Compressors:** Substitute polytropic head vs. pressure ratio ($P_d / P_s$) and lobe-meshing harmonics to separate inlet filter blockage and cooler fouling from pressure sensor drift.
3. **Marine Thrusters & Seawater Pumps:** Substitute propeller torque/thrust laws ($P \propto N^3$) and Gear Mesh Frequency ($\text{GMF}$) to detect sea-chest bio-fouling/ice clogging and shaft misalignment.
4. **Paper-Mill Stock & Slurry Pumps:** Apply viscosity-derated $H$-$Q$ curves to separate dewatering screen plugging and impeller erosion from magnetic flow-meter electrode coating drift.

---

## 6. Honest Limitations & Operational Boundaries

1. **Observability on `T0_LEGACY` Pumps:**
   Legacy pumps without vibration or suction pressure sensors (`PUMP-03`, `PUMP-05`, `PUMP-11`) cannot observe early Stage-1/Stage-2 subsurface bearing spalling (`BEARING_WEAR`) until secondary mechanical drag or overload occurs, nor can they directly measure suction vacuum without inferring head deficits from the pump curve. Upgrading to the **€2.25k–€2.98k T1 Retrofit package** resolves this blind spot.
2. **Rotordynamic 1x vs. 2x Ambiguity (`IMBALANCE` vs. `MISALIGNMENT`):**
   On `T1_RETROFIT` wireless sensors that report 5-minute summary statistics with 4-hour burst snapshots, a fast onset of shaft misalignment between 4-hour burst captures can initially resemble rotor unbalance (`IMBALANCE`) until the next 10.24 kHz snapshot confirms the $2\times$ harmonic peak. Both correctly trigger a `MACHINE` mechanical inspection on the rotor/coupling assembly.
3. **Simultaneous Compounding Faults (Sensor Drift Masking Hydraulic Clogging):**
   When an upward-drifting suction pressure transmitter coincides with a clogging suction strainer (as seen on `PUMP-10`), single-channel thresholds are masked. Our pipeline mitigates this by checking cross-channel physical coupling (discharge head vs. motor power and speed ratio), though extreme simultaneous sensor + machine faults reduce fault-type certainty and should be flagged for human gauge verification.
4. **Advisory-Only Boundary (No Autonomous Control):**
   By design, the system never writes setpoints to VFDs, valves, or PLCs, and is not a substitute for SIL-rated hardware protection relays or class-approved emergency trips.

