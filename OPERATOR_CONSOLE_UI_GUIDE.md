# Retrofit AI — Operator Console UI & Systems Documentation
**Hackathon:** Since AI Hackathon 2026 · Turku (6–8 November 2026)  
**Challenge Owner:** Business Turku · **Technical Sparring Partner:** Elomatic  
**Status:** Complete & Production-Ready  

---

## 1. Executive Summary & Architecture

The **Retrofit AI Operator Intelligence Console** is a local, high-performance, interactive Streamlit application designed as an **operator advisory companion** for 10–30 year old industrial and maritime cooling-water pumps.

In strict alignment with the challenge brief and **IEC 62443 / NIST SP 800-82** cyber-physical security standards, the system is **100% read-only**: it listens to real-time and replay telemetry, identifies developing machine and sensor degradation early, presents explainable evidence, evaluates counterfactual maintenance interventions via a digital twin simulator, and **never commands or writes to equipment**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT OPERATOR CONSOLE                      │
│                                (app.py)                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
[ Module 1: Fleet Overview ]   [ Module 2: Replay Viewer ]    [ Module 3: Diagnostics ]
• 12 Pump Asset Cards          • 3-Row Synchronized Plotly    • Machine vs Sensor Triage
• T0, T1, T2 Tier Badging      • Hydraulics / Drive / Vib     • Calibrated Confidence
• Capability Matrix            • 6 Famous Incident Presets    • Physical Evidence Matrix
• Live Filter Controls         • FFT Harmonic Spectrum (10kHz)• Action Checklist & Cautions
                                    │                               │
                                    ▼                               ▼
                       [ Module 4: "What-If" Twin ]    [ Module 5: BOM & Economics ]
                       • retrofit_sim.sandbox.what_if  • Option A & B Hardware BOM
                       • Test Clean Strainer / Bearings • Interactive ROI / Payback
                       • Cavitation NPSH & Overload    • IEC 62443 Diode Diagram
```

---

## 2. File Structure & Component Map

All UI code has been written modularly under `ui_components/` with clean separation of concerns:

```
c:\Users\aleks\Downloads\businessturku\
├── app.py                             # Main Streamlit dashboard & sidebar router
├── verify_ui.py                       # Automated regression verification script
├── OPERATOR_CONSOLE_UI_GUIDE.md       # This comprehensive documentation file
└── ui_components/
    ├── __init__.py                    # Package initializer
    ├── data_loader.py                 # Cached ingestion pipeline (@st.cache_data)
    ├── fleet_view.py                  # Module 1: Fleet Overview & Sensor Tier Dashboard
    ├── telemetry_view.py              # Module 2: Synchronized Plotly Replay & FFT Inspector
    ├── diagnostic_drawer.py           # Module 3: Operator Explainability & Diagnostic Drawer
    ├── what_if_view.py                # Module 4: Digital Sandbox "What-If" Counterfactual Twin
    └── retrofit_economics.py          # Module 5: Hardware BOM, ROI Model & Cyber Safety
```

---

## 3. Detailed Module Breakdown

### Module 1: Fleet Overview & Sensor Tier Dashboard (`ui_components/fleet_view.py`)
- **Fleet KPI Ribbon:** Displays monitored pump count (12 assets across RoPax ferries and chemical/food/paper plants), breakdown across instrumentation tiers, and total ground-truth incident events.
- **12 Interactive Pump Cards:**
  - Badges for **Instrumentation Tier**:
    - `T0_LEGACY` (Yellow): Minimal baseline telemetry (motor current, discharge pressure).
    - `T1_RETROFIT` (Blue): Low-cost retrofit adding wireless vibration/temperature and suction pressure (~€2.4k).
    - `T2_FULL` (Green): Comprehensive sensor suite with flow meter, winding PT100, and wired IO-Link.
  - Badges for **Benchmark Split**: `train` (grey), `validation` (cyan), `test` (pink).
  - Quick-glance specifications: Rated flow ($m^3/h$), rated head ($m$), motor kW, rated RPM, fluid type, and commissioned year.
  - **"Inspect Asset ➜" Action Button:** Instantly sets the global session state and navigates to the Replay and Diagnostics views.
- **Dynamic Filter Controls:** Multi-select filtering by Split, Instrumentation Tier, Equipment Profile (`SHIP_SW_COOLING`, `SHIP_LT_FRESHWATER`, `INDUSTRIAL_CW`, `PROCESS_TRANSFER`), and Site.
- **Retrofit Capability Matrix:** Side-by-side comparison table explaining sensor visibility, sample rates, and failure mode detection capabilities across T0, T1, and T2.

---

### Module 2: Interactive Incident Replay & Multi-Signal Telemetry (`ui_components/telemetry_view.py`)
- **6 One-Click Famous Incident Bookmarks:**
  1. `🔴 PUMP-06`: Chemical Plant D — Suction strainer clogging developing into violent cavitation (Oct 8–10, 2025).
  2. `🟡 PUMP-04`: Food Plant C — Stage-2 bearing wear crest-factor spike followed by discharge restriction (June 16–22, 2025).
  3. `🔵 PUMP-01`: RoPax Ferry A — Seawater strainer clog + wireless gateway comms loss (Aug 11–16, 2025).
  4. `⚪ PUMP-03`: RoPax Ferry A — T0 legacy pump: severe discharge restriction vs calibration shift (March 2–7, 2026).
  5. `🧪 PUMP-08`: RoPax Ferry B (Validation) — Strainer clogging & bearing wear caught early (Sept 8–14, 2025).
  6. `🧪 PUMP-09`: Food Plant C (Validation) — Shaft misalignment & discharge restriction (Feb 9–15, 2026).
- **Synchronized 3-Row Plotly Subplots:**
  - **Subplot 1 (Hydraulics & Pressure):** Discharge pressure ($kPa$), Suction pressure ($kPa$), and volumetric flow ($m^3/h$).
  - **Subplot 2 (Drive & Electrical):** Motor speed ($RPM$), Current ($A$), and Input power ($kW$).
  - **Subplot 3 (Mechanical Vibration):** Vibration velocity RMS ($mm/s$), Acceleration Crest Factor (peak/RMS), Bearing temperature ($^\circ C$), and the configurable **ISO 10816 alarm threshold limit** line.
- **Visual Overlays & Time Scrubber:**
  - Shaded bands for Ground Truth Incidents (`MACHINE` = amber, `SENSOR` = deep sky blue).
  - Red dotted vertical markers for plant **SCADA threshold alarms**.
  - Purple dash-dot markers for **AI anomaly alerts** loaded from model predictions.
  - Interactive **Black Vertical Cursor** tied directly to the scrubber slider, updating the real-time telemetry snapshot banner and the diagnostic drawer.
- **Raw 10.24 kHz Vibration Waveform & FFT Harmonic Spectrum Inspector:**
  - Reads raw acceleration bursts from `data/full/vibration_snapshots.parquet`.
  - Plots the first 100 ms of the raw acceleration waveform ($g$ vs $ms$).
  - Computes the FFT frequency spectrum ($0–1500\ Hz$) using NumPy `rfft`.
  - Automatically calculates and overlays vertical harmonic reference lines:
    - $1X$ Shaft Running Frequency ($f_r = \text{rpm} / 60$)
    - $2X$ Shaft Misalignment Harmonic
    - $BPFO$ (Ball Pass Frequency Outer Race) = $1X \times \text{bearing\_bpfo\_order}$
    - $BPFI$ (Ball Pass Frequency Inner Race) = $1X \times \text{bearing\_bpfi\_order}$
    - $VPF$ (Vane Pass Frequency) = $1X \times \text{impeller\_vanes}$
  - Explains why high-frequency defect peaks allow detecting bearing wear days before temperature or velocity rises.

---

### Module 3: Operator Explainability & Diagnostic Drawer (`ui_components/diagnostic_drawer.py`)
- **Triage Status Banner:**
  - Distinct colored status pills:
    - `🚨 MACHINE FAULT DETECTED` (Orange/Amber)
    - `📡 SENSOR / INSTRUMENTATION FAULT` (Electric Blue/Cyan)
    - `✅ NORMAL HEALTHY OPERATION` (Green)
- **Calibrated Confidence Meter:**
  - Visual confidence percentage bar (e.g. `88% Confidence`).
  - Transparent handling of ambiguous signals: classifies as `UNKNOWN` when confidence is $< 65\%$ to avoid misclassification penalties.
- **Physical Consistency Evidence Matrix:**
  - Cross-validates physical coupling against centrifugal pump affinity laws:
    - *Example (Machine):* Suction pressure dropped 38 kPa while speed increased to 72% $\rightarrow$ verifies physical strainer restriction rather than transmitter drift.
    - *Example (Sensor):* Single channel frozen for 30 minutes while load and ambient vary $\rightarrow$ flags `SENSOR_STUCK`.
- **Actionable Human-in-the-Loop Checklist:**
  - Primary recommended inspection / maintenance task.
  - Operational safety caution (e.g. *"Do NOT override VFD speed setpoint above 80% to avoid cavitation damage"*).
  - Operator interactive verification checkboxes and mock "Log Work Order" workflow.
  - One-click shortcut to test proposed interventions in the Digital Sandbox.

---

### Module 4: Digital Sandbox "What-If" Counterfactual Twin (`ui_components/what_if_view.py`)
- Directly integrates `retrofit_sim.sandbox.what_if`.
- **Operator Proposed Interventions:**
  - `clean_strainer` (Restores inlet flow resistance; labor ~3h, parts ~€50).
  - `replace_bearing` (Restores vibration baseline; labor ~12h, parts ~€900).
  - `realign` (Laser realignment; labor ~6h, parts ~€150).
  - `speed_setpoint` (Adjusts VFD speed ratio between $0.40$ and $1.10$).
  - `valve_position` (Adjusts throttling position between $0\%$ and $100\%$).
- **Latent Condition Hypotheses:** Sliders for hypothesized physical degradation (`clog`, `bearing_wear`, `restriction`, `misalignment`).
- **First-Principles Evaluation:**
  - Safety validation check:
    - `ACCEPTED`: Interventions within allowable hydraulic and motor envelopes.
    - `REJECTED`: Fails physical safety constraints (e.g. NPSH cavitation margin $< 1.3\ m$, motor load factor $> 100\%$, speed step $> 20\%$).
  - Before vs After comparison metric cards and comprehensive data table:
    - Flow delivery ($m^3/h$) & Head ($m$)
    - NPSH cavitation margin ($m$)
    - Power consumption ($kW$) & daily energy cost (€/day)
    - Vibration RMS velocity ($mm/s$) & Bearing temperature ($^\circ C$)
  - Work-order direct cost estimation (stop duration, labor hours, total cost in EUR).

---

### Module 5: Retrofit Feasibility, Hardware BOM & Economics (`ui_components/retrofit_economics.py`)
- **Verified Hardware Bill of Materials (from `RETROFIT_FEASIBILITY.md`):**
  - **Option B: Modbus Minimum (~€2.1k–€2.4k hardware / ~€4.5k installed):**
    - 2x Banner QM30VT2-QP Modbus RTU dual vibration/temperature probes (€320/ea)
    - 2x WIKA A-10 4–20 mA pressure transmitters (€145/ea)
    - 1x RS PRO PT100 4-wire RTD probe (€108)
    - 1x Carlo Gavazzi EM24 Modbus energy meter (€430)
    - 1x Siemens SIMATIC IOT2050 industrial edge PC (€602)
  - **Option A: Wired IO-Link Digital (~€2.8k–€3.1k hardware / ~€5.2k installed):**
    - 2x ifm electronic VVB001 IO-Link vibration probes (€530/ea)
    - 1x ifm AL1350 IO-Link master module (€373)
    - 2x Danfoss MBS 3100 marine class-approved pressure transmitters (€288/ea)
    - 1x Revolution Pi Connect 4 industrial edge computer (€497)
  - **Option C: Clamp-on Ultrasonic Flow Meter (+€380 TUF-2000M class).**
- **Interactive Economic ROI & Payback Model:**
  - Real-time sliders for failure rate, recall rate, unplanned downtime cost (€/hr), repair costs, and technician alert verification costs.
  - Live calculation of **Gross Avoided Loss**, **Net Annual Benefit (€/yr)**, **Payback Period in Months** ($< 12$ months), and **5-Year ROI (%)**.
- **Cyber-Physical Read-Only Safety Architecture:**
  - ASCII and structured block diagram demonstrating strict unidirectional data flow.
  - Documents compliance with **IEC 62443-3-3** (Zone & Conduit separation), **NIST SP 800-82r3** (DMZ data diode), **IACS UR E26/E27** (maritime cyber resilience), and **IEC 61511** (Safety Instrumented System independence).

---

### Ingestion & Caching Layer (`ui_components/data_loader.py`)
- Uses Streamlit's `@st.cache_data` for near-instant page reloads.
- Automatic tiered fallback: Prefers `data/full/` (`features_1min.parquet` 10.4MB, `vibration_snapshots.parquet` 14.2MB), seamlessly falling back to `data/starter/` (`telemetry.csv.gz`).
- Ingests model predictions (`predictions_B1_mode_aware_z_validation.csv` or test submissions) to enable immediate visual benchmarking against ground truth.

---

## 4. Verification & Testing

An automated regression script [verify_ui.py](file:///c:/Users/aleks/Downloads/businessturku/verify_ui.py) tests all underlying data loaders, physics sandbox calls, and diagnostic logic:

```powershell
python verify_ui.py
```

### Verification Output:
```
=== RETROFIT AI UI VERIFICATION ===
Testing data_loader...
  ✓ Loaded 12 assets (splits: {'train': 7, 'test': 3, 'validation': 2})
  ✓ Loaded 36 ground-truth events across train/validation
  ✓ Loaded 51 SCADA alarms
  ✓ Loaded 1-min features for PUMP-06 (10080 rows)
  ✓ Loaded 42 raw vibration snapshots for PUMP-04

Testing 'What-If' digital twin sandbox...
  ✓ Simulation accepted: True
  ✓ Baseline flow: 59.8 m³/h -> Proposed: 60.1 m³/h
  ✓ Intervention cost: €275.0

Testing diagnostic triage logic...
  ✓ Diagnostic Triage: MACHINE -> CAVITATION (Confidence: 85%)
  ✓ Action: Immediately throttle discharge valve or reduce speed to restore...

🎉 ALL TESTS PASSED! UI components and digital twin are fully operational.
```

---

## 5. How to Setup & Run the Application

### 5.1 Prerequisites
- **Supported Python Version:** Python 3.10 – 3.13 (Python 3.12 recommended).
  *(Avoid Python 3.15+ alpha releases, as binary wheels for scientific libraries like Pandas/Scikit-learn are not yet published for them).*
- If Python 3.12 is not yet installed on Windows:
  ```powershell
  winget install Python.Python.3.12
  ```

---

### 5.2 Step-by-Step Installation (from scratch)

Run these commands in PowerShell from the project root directory (`c:\Users\aleks\Downloads\businessturku`):

```powershell
# 1. (Optional) Allow script execution in current PowerShell session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Create a clean virtual environment using Python 3.12 (or 3.13)
py -3.12 -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Install UI dependencies (pinning <1.38 avoids Windows MAX_PATH issue)
pip install "streamlit<1.38" plotly
```

---

### 5.3 Automated Verification (Optional but Recommended)

Before launching the web app, run the automated self-test to verify that all data loaders, digital twin simulations, and diagnostic logic execute cleanly:

```powershell
python verify_ui.py
```
*(You should see `🎉 ALL TESTS PASSED! UI components and digital twin are fully operational`).*

---

### 5.4 Launching the Streamlit Console

```powershell
streamlit run app.py
```
*(Or alternatively: `python -m streamlit run app.py`)*

The application will start the local server and automatically open your default browser at:
👉 **`http://localhost:8501`**

---

### 5.5 Troubleshooting Common Windows Issues

| Issue / Error | Cause | Quick Fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell default execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `meson setup ERROR: Could not parse vswhere.exe` | Pip tried building Pandas on Python 3.15 without C++ compiler | Use Python 3.12 (`py -3.12 -m venv .venv`), which has pre-compiled binary wheels |
| `[WinError 206] Filename or extension is too long` | Windows 260-character `MAX_PATH` limit | Pin Streamlit: `pip install "streamlit<1.38" plotly` |
| `Fatal Python error: Failed to import encodings` | Stale `VIRTUAL_ENV` variable from deleted venv | Open a new terminal tab or run `Remove-Item env:VIRTUAL_ENV` |

---

## 6. Alignment with Jury Scoring Criteria

| Scoring Criterion | Weight | How the UI Addresses It |
|---|---|---|
| **Working Demo & Clarity** | **10 %** | Polished, intuitive industrial UI with instant 1-click incident bookmarks, synchronized multi-axis charts, and smooth timeline scrubbing. |
| **Business Usefulness** | **25 %** | Clear triage, actionable checklists, work-order integration, and real-time downtime avoidance economics. |
| **Technical Validity** | **25 %** | Synchronized multi-physics signals, FFT harmonic spectrum analysis ($1X, 2X, BPFO, BPFI$), and direct integration with model predictions. |
| **Reliability & Safety** | **20 %** | Physical read-only architecture badge, separation of sensor vs machine faults, calibrated confidence, and transparent `UNKNOWN` handling. |
| **Retrofit Feasibility** | **20 %** | Detailed €2.4k T1 BOM breakdown, vendor part numbers, payback calculator, and IEC 62443 cyber-physical compliance documentation. |

