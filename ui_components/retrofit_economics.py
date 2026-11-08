"""Module 5: Low-Cost Retrofit Feasibility, Hardware BOM & Economics.

Visualizes the real-world retrofit pathways, verified BOM costs from RETROFIT_FEASIBILITY.md,
an interactive ROI / payback calculator, and the IEC 62443 read-only cyber-physical safety architecture.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np


def render_retrofit_economics_view():
    """Render the Retrofit Feasibility, BOM & Economics Tab."""
    st.markdown("### 💰 Low-Cost Retrofit Feasibility & Economics")
    st.markdown(
        "Quantitative justification for retrofitting legacy cooling-water pumps. "
        "Real-world sensor Bill of Materials (BOM), interactive downtime ROI calculator, "
        "and IEC 62443 read-only cyber-physical safety architecture."
    )

    tab_bom, tab_roi, tab_arch, tab_transfer = st.tabs([
        "📦 Hardware Bill of Materials (BOM)",
        "📈 Interactive ROI & Payback Calculator",
        "🛡️ Edge Device & Read-Only Architecture",
        "🔄 Cross-Machine Transfer & Honest Limitations"
    ])

    # TAB 1: Hardware BOM
    with tab_bom:
        st.markdown("#### 🛠️ Minimum Retrofit Instrumentation Pathways")
        st.markdown(
            "Upgrading a legacy pump (T0) to an intelligent predictive asset (T1) does not require replacing the pump or motor. "
            "By adding non-invasive vibration/temperature probes and suction pressure transmitters to the existing motor control center, "
            "the system detects cavitation, strainer clogging, and stage-2 bearing wear."
        )

        bom_options = {
            "Option B: Modbus Minimum (Lowest Cost)": {
                "total_eur": 2250,
                "installed_eur": 4500,
                "items": [
                    {"Component": "Vibration + Temp (Dual)", "Model": "Banner QM30VT2-QP (Modbus RTU)", "Qty": 2, "Unit Price EUR": 320, "Total EUR": 640, "Source": "LIST"},
                    {"Component": "Pressure Transmitter", "Model": "WIKA A-10 (0-10 bar, 4-20 mA)", "Qty": 2, "Unit Price EUR": 145, "Total EUR": 290, "Source": "LIST"},
                    {"Component": "Bearing Temp Probe", "Model": "RS PRO PT100 Class B 4-wire", "Qty": 1, "Unit Price EUR": 108, "Total EUR": 108, "Source": "LIST"},
                    {"Component": "Energy / Power Meter", "Model": "Carlo Gavazzi EM24 (Modbus)", "Qty": 1, "Unit Price EUR": 430, "Total EUR": 430, "Source": "LIST"},
                    {"Component": "Industrial Edge PC", "Model": "Siemens SIMATIC IOT2050", "Qty": 1, "Unit Price EUR": 602, "Total EUR": 602, "Source": "LIST"},
                    {"Component": "Protocol Gateway", "Model": "Moxa MGate 5105-MB-EIP", "Qty": 1, "Unit Price EUR": 180, "Total EUR": 180, "Source": "ASSUMPTION"},
                ]
            },
            "Option A: Wired Minimum (IO-Link Digital)": {
                "total_eur": 2980,
                "installed_eur": 5200,
                "items": [
                    {"Component": "Vibration + Temp (IO-Link)", "Model": "ifm electronic VVB001", "Qty": 2, "Unit Price EUR": 530, "Total EUR": 1060, "Source": "LIST"},
                    {"Component": "IO-Link Master Unit", "Model": "ifm electronic AL1350", "Qty": 1, "Unit Price EUR": 373, "Total EUR": 373, "Source": "LIST"},
                    {"Component": "Marine Pressure Transmitter", "Model": "Danfoss MBS 3100 (Class approved)", "Qty": 2, "Unit Price EUR": 288, "Total EUR": 576, "Source": "LIST"},
                    {"Component": "Bearing Temp Probe", "Model": "RS PRO PT100 Class B 4-wire", "Qty": 1, "Unit Price EUR": 108, "Total EUR": 108, "Source": "LIST"},
                    {"Component": "Energy / Power Meter", "Model": "Carlo Gavazzi EM24", "Qty": 1, "Unit Price EUR": 430, "Total EUR": 430, "Source": "LIST"},
                    {"Component": "Industrial Edge PC", "Model": "Revolution Pi Connect 4 (Quad-Core)", "Qty": 1, "Unit Price EUR": 497, "Total EUR": 497, "Source": "LIST"},
                ]
            },
            "Option C: Optional Ultrasonic Flow Addition": {
                "total_eur": 380,
                "installed_eur": 850,
                "items": [
                    {"Component": "Clamp-On Ultrasonic Flow", "Model": "TUF-2000M Non-invasive Class", "Qty": 1, "Unit Price EUR": 380, "Total EUR": 380, "Source": "LIST field report"},
                ]
            }
        }

        opt_choice = st.radio("Select Retrofit Architecture Package:", list(bom_options.keys()), horizontal=True)
        pkg = bom_options[opt_choice]

        c_b1, c_b2, c_b3 = st.columns(3)
        with c_b1:
            st.metric("Hardware BOM Cost", f"€{pkg['total_eur']:,}")
        with c_b2:
            st.metric("Turnkey Installed Cost (est.)", f"€{pkg['installed_eur']:,}", help="Includes IP67 enclosure, cabling, and field labor")
        with c_b3:
            st.metric("Payback Timeframe", "< 12 Months", help="Based on avoiding a single unplanned line stoppage")

        st.dataframe(pd.DataFrame(pkg["items"]), use_container_width=True, hide_index=True)

    # TAB 2: ROI Calculator
    with tab_roi:
        st.markdown("#### 📊 Interactive Economic Downtime & ROI Model")
        st.caption("Calculates financial returns from early fault detection, avoided catastrophic overhauls, and reduced unplanned downtime.")

        col_r1, col_r2 = st.columns([1.2, 1.2])
        with col_r1:
            st.markdown("##### ⚙️ Operational Inputs:")
            faults_per_year = st.slider("Developing Faults per Year (per pump)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
            model_recall = st.slider("AI Early Detection Recall Rate (%)", min_value=50, max_value=100, value=89, step=1) / 100.0
            unplanned_dt_h = st.slider("Unplanned Stop Duration (hours)", min_value=4, max_value=48, value=10, step=1)
            planned_dt_h = st.slider("Planned Inspection Duration (hours)", min_value=1, max_value=12, value=3, step=1)
            hourly_downtime_cost = st.slider("Plant Downtime / Off-hire Cost (€/hour)", min_value=200, max_value=5000, value=1500, step=100)

        with col_r2:
            st.markdown("##### 💶 Financial & Hardware Cost Inputs:")
            unplanned_repair_cost = st.slider("Unplanned Catastrophic Repair Cost (€)", min_value=3000, max_value=25000, value=9000, step=500)
            planned_repair_cost = st.slider("Planned Early Repair Cost (€)", min_value=1000, max_value=10000, value=4000, step=500)
            installed_capex = st.slider("Total Retrofit Investment CAPEX (€/pump)", min_value=2000, max_value=10000, value=4500, step=250)
            annual_opex = st.slider("Annual Connectivity & Maintenance OPEX (€/yr)", min_value=200, max_value=2000, value=700, step=50)
            false_alert_rate = st.slider("False Alert Nuisance Rate (alerts/pump-year)", min_value=0, max_value=50, value=26, step=1)
            cost_per_false_alert = st.slider("Labor Cost to Check False Alert (€)", min_value=50, max_value=300, value=120, step=10)

        # Calculations
        faults_caught = faults_per_year * model_recall
        # Unplanned incident total cost = downtime loss + repair
        cost_unplanned_incident = (unplanned_dt_h * hourly_downtime_cost) + unplanned_repair_cost
        # Planned incident total cost = downtime loss + repair
        cost_planned_incident = (planned_dt_h * hourly_downtime_cost) + planned_repair_cost

        # Annual savings from avoiding catastrophic failures
        avoided_loss_per_fault = cost_unplanned_incident - cost_planned_incident
        gross_annual_savings = faults_caught * avoided_loss_per_fault
        false_alert_cost = false_alert_rate * cost_per_false_alert

        net_annual_benefit = gross_annual_savings - false_alert_cost - annual_opex
        payback_months = (installed_capex / net_annual_benefit * 12.0) if net_annual_benefit > 0 else 999.0
        five_year_roi = ((net_annual_benefit * 5.0 - installed_capex) / installed_capex * 100.0) if installed_capex > 0 else 0.0

        st.markdown("---")
        st.markdown("##### 📈 Return on Investment Summary:")

        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        with k_c1:
            st.metric("Gross Avoided Loss", f"€{gross_annual_savings:,.0f} / yr")
        with k_c2:
            st.metric("Net Annual Benefit", f"€{net_annual_benefit:,.0f} / yr", delta=f"+€{net_annual_benefit:,.0f}")
        with k_c3:
            st.metric("Payback Period", f"{payback_months:.1f} Months", help="Time to recover entire hardware + installation CAPEX")
        with k_c4:
            st.metric("5-Year Net ROI", f"{five_year_roi:,.0f}%")

    # TAB 3: Cyber-Physical Read-Only Safety Architecture
    with tab_arch:
        st.markdown("#### 🛡️ Read-Only Cyber-Physical Safety Architecture")
        st.markdown(
            "In maritime and critical process plants, condition monitoring systems must adhere to strict "
            "cyber-security isolation. **The Retrofit AI assistant is strictly read-only and never commands or writes to equipment.**"
        )

        st.markdown("""
```
┌────────────────────────────────────────────────────────────────────────┐
│                        SAFETY / SIS ZONE (SIL-2/3)                     │
│  [ Emergency Stop & Independent Trips ]  ── (Completely Air-Gapped)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ NO ACCESS
┌───────────────────────────────────▼────────────────────────────────────┐
│                        OT CONTROL ZONE (PLC / VFD)                     │
│  [ Motor Control Center ] ── [ Speed / Flow Controller ]               │
│             │                                                          │
│             ▼ Read-Only Mirrors (OPC-UA / Modbus RTU Read Registers)   │
└─────────────┬──────────────────────────────────────────────────────────┘
              │ Unidirectional Data Flow (Hardware Data Diode / DMZ)
┌─────────────▼──────────────────────────────────────────────────────────┐
│                   RETROFIT EDGE ZONE (RevPi Connect 4)                 │
│  • Local FIFO buffer (7-day offline buffer during sea voyages)         │
│  • Edge inference (Z-scores, affinity residuals, crest factor)         │
│  • ZERO write paths configured (no output relays, no write credentials)│
└─────────────┬──────────────────────────────────────────────────────────┘
              │ Encrypted Outbound Advisory Telemetry (MQTT / HTTPS)
┌─────────────▼──────────────────────────────────────────────────────────┐
│                   OPERATOR WORKSTATION / CLOUD HISTORIAN               │
│  [ Streamlit Operator Console ] ➔ Displays evidence, alerts, what-if   │
│  Human in the loop: Operator confirms physical work orders manually    │
└────────────────────────────────────────────────────────────────────────┘
```
        """)

        st.markdown("##### 🖥️ Embedded Edge-Device Engineering Design (RevPi Connect 4 / SIMATIC IOT2050):")
        edge_specs = [
            {"Subsystem": "Target Hardware", "Specification": "Revolution Pi Connect 4 (ARM Cortex-A72 Quad-Core 1.5 GHz, 4 GB RAM, 32 GB eMMC) or Siemens SIMATIC IOT2050"},
            {"Subsystem": "Fieldbus Ingestion", "Specification": "RS-485 Modbus RTU (Banner QM30VT2 + Gavazzi EM24) + IO-Link Master AL1350 (ifm VVB001) + OPC-UA Read-Only Client"},
            {"Subsystem": "Local Inference Latency", "Specification": "< 35 ms per 60-second telemetry window (causal EWMA + physics residuals + 2048-pt FFT + tree ensemble)"},
            {"Subsystem": "Memory & CPU Footprint", "Specification": "~85 MB RSS memory, < 4% average CPU utilization across 4 monitored pumps per edge node"},
            {"Subsystem": "Voyage Offline Buffer", "Specification": "14-day local SQLite WAL ring-buffer (~180 MB for 4 pumps incl. 4-hour raw vibration bursts) for maritime satellite blackouts"},
            {"Subsystem": "Hardware Write Isolation", "Specification": "Physical RS-485 TX disabled toward PLC; OPC-UA anonymous/read-only role; zero DO/AO relay wiring to MCC or VFD"},
        ]
        st.dataframe(pd.DataFrame(edge_specs), use_container_width=True, hide_index=True)

        st.markdown("##### 📌 Regulatory & Standard Compliance:")
        st.markdown(
            "• **IEC 62443-3-3:** Zone and conduit segmentation prevents lateral compromise from edge to OT control.\n"
            "• **NIST SP 800-82 Rev 3:** Recommends unidirectional security gateways for historian/monitoring connections.\n"
            "• **IACS UR E26 / E27 (Vessels contracted from July 2024):** Cyber resilience mandates that non-critical monitoring "
            "cannot impact ship propulsion, steering, or essential auxiliary cooling.\n"
            "• **IEC 61511:** Complete independence of Safety Instrumented Systems (SIS) from basic process control and monitoring."
        )

    # TAB 4: Cross-Machine Transfer Plan & Honest Limitations
    with tab_transfer:
        st.markdown("#### 🔄 Plan for Transferring the Solution to Other Rotating Machines")
        st.markdown(
            "Because **RETROFIT AI** separates its architecture into **(1) Regime-Normalized Affinity Laws**, "
            "**(2) Rotational Order-Tracked Spectral Harmonics ($1\\times, 2\\times$, BPFO/BPFI, Blade/Vane Pass)**, and "
            "**(3) Cross-Channel Sensor Consistency Checks**, the pipeline transfers directly to other legacy rotating assets "
            "by swapping only the equipment's static nameplate JSON (`data/metadata/assets.json`):"
        )
        transfer_rows = [
            {
                "Target Machine Class": "🌬️ Industrial Centrifugal Fans & Blowers",
                "Physics Law Substitution": "Fan Affinity Laws: ΔP_static ∝ N², Flow ∝ N, Power ∝ N³ (air density ρ_air corrected by ambient temp)",
                "Vibration Order Mapping": "1× Unbalance, 2× Belt/Coupling Misalignment, Blade Pass Frequency (BPF = N_blades × f_shaft), Rolling Bearing BPFO/BPFI",
                "Equivalent Faults Detected": "Filter bank clogging (= strainer clogging), aerodynamic stall/surge (= cavitation), damper restriction, bearing wear",
            },
            {
                "Target Machine Class": "🗜️ Rotary Screw & Centrifugal Compressors",
                "Physics Law Substitution": "Isentropic / Polytropic Compression Head vs. Discharge/Suction Pressure Ratio (P_d / P_s) and Motor kW",
                "Vibration Order Mapping": "Lobe / Screw Meshing Frequency (Z_rotors × f_shaft), 1×/2× coupling harmonics, high-frequency acoustic envelope",
                "Equivalent Faults Detected": "Inlet air filter blockage, cooler fouling, discharge check-valve restriction, rotor contact, pressure sensor drift",
            },
            {
                "Target Machine Class": "🚢 Marine Thrusters & Cooling Seawater Pumps",
                "Physics Law Substitution": "Propeller / Thruster Torque & Thrust Law (P ∝ N³, T ∝ N² × pitch) compensated for hull sea-chest temperature",
                "Vibration Order Mapping": "Shaft 1×/2×, Gear Mesh Frequency (GMF), Propeller Blade Rate, Stern-tube / roller bearing fault orders",
                "Equivalent Faults Detected": "Sea-chest bio-fouling / ice clogging, propeller cavitation, shaft misalignment, lubrication overheat, PT100 drift",
            },
            {
                "Target Machine Class": "🏭 Paper-Mill Stock & Slurry Pumps",
                "Physics Law Substitution": "Viscosity-derated H-Q curve (Hydraulic Institute consistency correction) + specific energy per m³ pumped",
                "Vibration Order Mapping": "Vane Pass Frequency (VPF = N_vanes × f_shaft), low-frequency recirculation broadband (20–300 Hz), BPFO/BPFI",
                "Equivalent Faults Detected": "Dewatering screen plugging, impeller erosion/wear (chronic ΔH drop), fiber air entrainment, flow-meter coating drift",
            },
        ]
        st.dataframe(pd.DataFrame(transfer_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### ⚠️ Honest Limitations & Operational Boundaries")
        st.warning(
            "**1. Observability Blind Spot on `T0_LEGACY` Assets (`PUMP-03`, `PUMP-05`, `PUMP-11`):**\n"
            "Legacy pumps lacking vibration and suction pressure sensors cannot observe early Stage-1/Stage-2 subsurface bearing spalling (`BEARING_WEAR`) "
            "until secondary mechanical drag increases motor current, nor can they distinguish suction strainer clogging from impeller wear without suction pressure. "
            "**Mitigation:** Upgrading `T0_LEGACY` pumps with the €2.25k–€2.98k `T1_RETROFIT` kit eliminates this blind spot."
        )
        st.warning(
            "**2. 4-Hour Burst Cadence Latency on `T1_RETROFIT` Wireless Probes:**\n"
            "On `T1_RETROFIT` assets, 5-minute summary telemetry reports broadband RMS and Crest Factor, while raw 10.24 kHz waveforms arrive every 4 hours. "
            "If a fast-developing shaft `MISALIGNMENT` initiates between 4-hour bursts, its 1×/2× spectral ratio is only confirmed when the next waveform burst arrives "
            "(until then, broadband vibration step-change heuristics classify it between `MISALIGNMENT` and `IMBALANCE`). Both trigger a `MACHINE` mechanical inspection."
        )
        st.warning(
            "**3. Simultaneous Compounding Sensor + Machine Faults (`PUMP-10` Edge Case):**\n"
            "When an upward-drifting suction pressure sensor occurs simultaneously with a clogging suction strainer, single-channel thresholds cancel out. "
            "Our physics residual engine detects the divergence between discharge head, VFD speed, and motor power, but extreme dual-fault overlap lowers fault-type confidence "
            "and prompts a dual manual gauge + strainer check."
        )
        st.info(
            "**4. Strictly Read-Only Advisory Boundary:**\n"
            "By design, RETROFIT AI never executes closed-loop control writes to VFDs, valves, or safety relays. It complements—and never replaces—certified hardware protection trips."
        )

