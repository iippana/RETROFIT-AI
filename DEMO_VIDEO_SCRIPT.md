# 🎬 RETROFIT AI — 3 to 5 Minute Demo Video Script

**Target Duration:** ~3 min 45 sec (fits comfortably inside the 3–5 minute window)  
**What to Have Open Before Recording:**
- Your browser open to `http://localhost:8501` on the **🏢 Fleet Overview** tab.
- Full-screen (`F11`) or maximized browser window so the charts and metrics look sharp.

---

## ⏱️ 0:00 – 0:45 | Part 1: The Problem, Architecture & Benchmark Results
**📍 Screen Action:**
1. Start on **🏢 Fleet Overview** (`http://localhost:8501`).
2. Highlight the top KPI row (`12 Pumps`, `T0 Legacy`, `T1 Retrofit`, `T2 Full`, `50 AI Alerts`) and the **Validation Composite Score (`99.8 / 100`, `+26.9 vs B1 Baseline`)**.
3. Scroll slightly down to show the **📏 Measured Comparison Against Provided Baselines (B0, B1, B2, B3)** table and the **12 Pump Cards** (specifically pointing out the locked test pumps `PUMP-10`, `PUMP-11`, `PUMP-12`).

**🗣️ What to Say:**
> "Hi, this is **RETROFIT AI**—our physics-informed diagnostic engine and read-only operator console for 10-to-30-year-old industrial and maritime cooling-water pumps, built for the Business Turku and Elomatic challenge.
>
> Legacy pumps suffer from two opposite problems: fixed SCADA alarms trigger too late—after cavitation or bearing damage is already severe—while naive statistical thresholds flood operators with false alarms during normal duty switches or sensor drift.
>
> Our causal pipeline in `src/pipeline.py` combines **centrifugal pump affinity laws**, **10.24 kHz vibration FFT order tracking**, and a **600-tree ensemble** to explicitly separate physical **Machine Faults** from **Sensor & Instrumentation Faults**.
>
> On the official event-based benchmark scorer, **RETROFIT AI achieves a 99.8 out of 100 composite score on the validation split**—outperforming the strongest provided baseline `B1` by **+26.9 points**. It catches **100% of events with zero false alerts**, **100% exact fault-type accuracy**, **100% Machine-versus-Sensor class separation**, and delivers a **13.9-hour median early-warning lead time** before the maintenance deadline. Across the blinded test pumps (`PUMP-10`, `PUMP-11`, `PUMP-12`), it generates 14 causal alerts in `predictions_test.csv`."

---

## ⏱️ 0:45 – 1:45 | Part 2: Synchronized Incident Replay & 10.24 kHz FFT Spectrum
**📍 Screen Action:**
1. Click **📈 Incident Replay & Telemetry** in the left sidebar.
2. Click the **`🟡 PUMP-04`** bookmark button (or select `PUMP-04` from the dropdown).
3. Point to the 3 synchronized telemetry panels:
   - **Row 1:** Ground-Truth windows (`MACHINE` in amber, `SENSOR` in blue) and the gold **Active Inspection Cursor**.
   - **Row 2:** Bright green **AI Diagnostic Alerts (`🧠 AI: ...`)** firing hours before plant failure.
   - **Row 3:** Mechanical vibration and temperature vs. the red ISO 10816 alarm limit.
4. Scroll down to the **10.24 kHz Raw Vibration Waveform & FFT Spectrum** section at the bottom of the page and point to the rotational harmonic badges (`1X Shaft`, `2X Misalign`, `BPFO`, `BPFI`, `VPF`).

**🗣️ What to Say:**
> "Let’s jump into **Incident Replay & Telemetry** on `PUMP-04`. Here the operator sees three synchronized physical domains: Hydraulics & Pressures on top, Electrical Drive & Speed in the middle, and Mechanical Vibration & Temperature on the bottom.
>
> Notice how our green **AI Diagnostic Alerts** in Row 2 fire early in the degradation progression—long before the red SCADA threshold alarm in Row 3 is ever breached.
>
> Equally important, look at the blue `SENSOR_NOISE_SPIKES` window in the middle of the week: when a single transmitter spikes or drifts without corresponding changes in motor power or hydraulic head, our affinity-law cross-check tags it as a **SENSOR** fault rather than waking up a mechanical crew.
>
> Scrolling down, for `T1_RETROFIT` and `T2_FULL` pumps, operators can inspect the raw **10.24 kHz vibration burst waveform and Hanning-windowed FFT spectrum**, with automatic markers for $1\times$ unbalance, $2\times$ misalignment, vane-pass frequency, and rolling-element bearing raceway frequencies (`BPFO` and `BPFI`)."

---

## ⏱️ 1:45 – 2:30 | Part 3: Explainable Triage & Human-in-the-Loop Checklist
**📍 Screen Action:**
1. Click the **`🔴 PUMP-06`** bookmark at the top of the Telemetry page (or select `PUMP-06`), then click **🔍 Diagnostic & Evidence Drawer** in the left sidebar.
2. Point out the **🚨 MACHINE FAULT DETECTED** badge (`SUCTION_STRAINER_CLOGGING`), the **AI Confidence meter**, the **Diagnostic Evidence & Physical Consistency** bullets on the left, and the **Recommended Operator Action** + **Human-in-the-Loop Checklist** on the right.
3. Tick the 3 operator verification checkboxes, then click **Open Counterfactual 'What-If' Simulation ➜** at the bottom right.

**🗣️ What to Say:**
> "Selecting `PUMP-06` and opening the **Diagnostic & Explainability Drawer**, the operator gets an immediate triage decision: **MACHINE FAULT — SUCTION STRAINER CLOGGING** with calibrated confidence and human-readable physical evidence.
>
> It explains *why* this is a physical machine fault and not a bad sensor: suction pressure dropped into vacuum while VFD speed and motor power climbed together to maintain flow demand.
>
> On the right, the operator receives a concrete recommended action, a safety caution against increasing VFD speed into cavitation, and a human-in-the-loop verification checklist. Before dispatching a maintenance crew, the operator can click **Open Counterfactual 'What-If' Simulation**."

---

## ⏱️ 2:30 – 3:15 | Part 4: Counterfactual 'What-If' Digital Twin Sandbox
**📍 Screen Action:**
1. You are now in **🧪 'What-If' Counterfactual Sandbox** (with `PUMP-06` and `clean_strainer` pre-selected).
2. Click the primary red button **🚀 Run Digital Twin Simulation**.
3. Point to the green **✅ INTERVENTION ACCEPTED** banner, the 4 KPI delta cards (`Flow Delivery +9.5 m³/h`, `NPSH Cavitation Margin`, `Vibration RMS`, `Energy Cost`), the before-vs-after table, and the **Estimated Work-Order Economics (`1.5 hours stop`, `€305 total cost`)**.

**🗣️ What to Say:**
> "Inside the **'What-If' Counterfactual Sandbox**, we wrap the first-principles hydrodynamic digital twin so the operator can safely test an intervention—such as cleaning the suction strainer, replacing a bearing, laser-realigning the shaft, or adjusting VFD speed—before touching the real plant.
>
> Clicking **Run Digital Twin Simulation** confirms that cleaning `PUMP-06`’s strainer is physically feasible and safe: it restores flow delivery by nearly 10 cubic meters per hour, lifts the NPSH cavitation margin back into the safe zone, reduces vibration and daily energy cost, and estimates the work order at 1.5 hours of stop time and 305 euros."

---

## ⏱️ 3:15 – 4:00 | Part 5: Low-Cost Retrofit BOM, Edge Design, Transferability & Honest Limitations
**📍 Screen Action:**
1. Click **💰 Retrofit BOM & Economics** in the left sidebar.
2. In **Tab 1 (`📦 Hardware Bill of Materials`)**, toggle between **Option B (`€2,250` Modbus Minimum)** and **Option A (`€2,980` IO-Link Digital)**.
3. Click **Tab 2 (`📈 Interactive ROI & Payback Calculator`)** and show the `< 12 Month` payback (`~2.9 months` default).
4. Click **Tab 3 (`🛡️ Edge Device & Read-Only Architecture`)** to show the **Data-Diode Read-Only diagram** and the **Embedded Edge-Device specs (`RevPi Connect 4` / `SIMATIC IOT2050`, `< 35 ms` latency, 14-day SQLite offline buffer)**.
5. Click **Tab 4 (`🔄 Cross-Machine Transfer & Honest Limitations`)** to show the transfer table (Fans, Compressors, Marine Thrusters, Stock Pumps) and the 4 yellow warning boxes for **Honest Limitations**.

**🗣️ What to Say:**
> "Finally, **Retrofit BOM & Economics** justifies the real-world deployment.
>
> In **Tab 1**, we show how upgrading a blind `T0_LEGACY` pump to `T1_RETROFIT` requires only **2,250 euros in off-the-shelf hardware**—adding clamp-on vibration/temperature probes and a suction pressure transmitter without replacing the pump or motor.
>
> **Tab 2** provides an interactive ROI calculator showing that avoiding even a single unplanned stoppage pays back the entire retrofit in **under 4 months**.
>
> **Tab 3** details our **IEC 62443 and IACS UR E26 compliant read-only edge architecture** running on a Revolution Pi Connect 4 or Siemens IOT2050: inference takes under 35 milliseconds per window, uses 85 megabytes of RAM, buffers 14 days of telemetry offline during sea voyages, and has **zero physical write paths** to the PLC or VFD.
>
> And in **Tab 4**, we lay out our **Cross-Machine Transfer Plan**—showing how swapping the static nameplate JSON adapts our affinity-law and shaft-order engine to centrifugal fans, compressors, marine thrusters, and paper-mill stock pumps—followed by our **Honest Limitations**: namely that `T0_LEGACY` pumps without vibration sensors cannot see early Stage-2 bearing spalling until upgraded to `T1_RETROFIT`, that `T1` wireless burst cadence has a 4-hour full-FFT confirmation window, and that simultaneous sensor drift plus strainer clogging requires manual gauge verification.
>
> Thank you!"

