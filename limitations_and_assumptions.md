# Limitations, Assumptions, and Research Integrity Statement

**Contribution:** When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition  
**Conference Submission:** IEEE SENSORS@25 (Cairo, Egypt • December 14–16, 2026)  
**Author:** Prince Chovatiya  

---

## 1. Explicit Methodological Assumptions

1. **Stationary Waist Alignment:**
   The experiments rely on the waist-mounted placement of the Samsung Galaxy S II smartphone used in the benchmark UCI HAR dataset. The coordinate system assumes the device’s vertical and lateral axes approximately correspond to the participant's body planes. If a user’s phone is free-floating in a pocket and changes orientation relative to gravity during motion, coordinate transform calibration (e.g., dynamic gravity alignment) is required before these findings apply directly.

2. **Controlled Mathematical Simulation of Sensor Failure:**
   Sensor dropouts were simulated by setting raw physical channels to zero (silent sensor disconnection or hardware shutdown), while single-axis failures were simulated by flatlining individual channels. In actual hardware, sensor failures can manifest as floating analog-to-digital converter (ADC) voltages, stuck-at-rail saturation ($+V_{max}$ or $-V_{min}$), or intermittent high-frequency noise spikes.

3. **Physical Hardware Triad Coupling vs. Mathematical Channel Ablation:**
   In physical MEMS IMU hardware, an accelerometer contains three physical sensing axes measuring total acceleration (dynamic movement plus static gravitational vector $1g$). In the UCI HAR dataset, 'body acceleration' is a software-derived signal calculated via a 0.3 Hz Butterworth filter. A true physical hardware failure (such as an ADC channel disconnect or sensing beam fracture) kills both total acceleration and derived body acceleration on that physical axis simultaneously. Our work explicitly distinguishes true physical IMU hardware axis failure (where coupled channels fail together) from mathematical single-channel ablation (which isolates classifier sensitivity to specific filtered frequency components).

3. **Gaussian Additive Noise Model:**
   The noise sweeps utilized zero-mean additive Gaussian noise ($\sigma \in [0.0, 2.0]$) added to the standardized raw signal arrays. While thermal noise and amplifier jitter in MEMS sensors follow Gaussian distributions, environmental noise such as loose phone rattling in a pocket or walking on vibrating surfaces (e.g., subways) often exhibits non-Gaussian, impulsive, or structured frequency characteristics.

4. **Energy Consumption Analogy:**
   The energy savings of duty-cycling the gyroscope are modeled based on published manufacturer datasheets (e.g., TDK InvenSense MPU-6050: ~3.6 mA active gyroscope current vs. ~0.5 mA active accelerometer current). Actual battery savings in a commercial smartphone also depend on operating system wake-lock overhead, bus transfer intervals (I2C vs. SPI DMA transfers), and application processor sleep states.

---

## 2. Boundaries of Generalization

* **Not a Clinical Diagnostic Tool:**
  This research evaluates sensing reliability for consumer fitness, personal activity logging, and ubiquitous context awareness. It makes no claims regarding clinical diagnostics, fall detection in frail populations, or gait analysis for neurological pathologies (such as Parkinson’s disease or stroke recovery).
* **Population Specificity:**
  The 30 study volunteers were healthy adults aged 19 to 48 performing scripted activities under experimenter observation. Movement signatures of pediatric, frail elderly, or mobility-impaired individuals may show different baseline frequencies and higher reliance on angular stabilization.
* **Device Homogeneity:**
  Data was collected from a single smartphone model. Inter-device hardware variability (sensor bias drift, factory calibration offsets, sensitivity tolerances across different MEMS suppliers) was not varied within this dataset.

---

## 3. What This Work Does NOT Claim

1. **We do NOT claim a novel deep neural architecture:**
   The novelty lies in the systematic empirical characterization of sensor failure modes, the quantification of functional asymmetry, and the formulation of an adaptive fallback strategy for low-power edge sensing.
2. **We do NOT claim 100% fault tolerance:**
   When the accelerometer is destroyed or heavily corrupted ($\sigma > 0.50$), all models fail. Our fallback mechanism specifically addresses the high-probability real-world scenario of gyroscope duty-cycling, where the accelerometer remains operational.
3. **We do NOT claim that gyroscopes are redundant:**
   Gyroscopes provide critical discriminative boundaries for dynamic gait separation (specifically walking vs. walking upstairs/downstairs). When the gyroscope is quiet, WALKING F1 drops from 0.8460 to 0.5257. Gyroscopes are essential when high-precision dynamic tracking is required, but can be safely disabled during stationary states.

---

## 4. Reproducibility & Research Integrity Pledge

* Every reported metric (Macro-F1, Balanced Accuracy, per-class F1, and confusion matrix cell) was directly calculated by executing the accompanying Python scripts on the real public UCI HAR dataset.
* No numbers, curves, or results were fabricated, hallucinated, or extrapolated from outside sources.
* All code, scripts, raw outputs, and plotting routines are openly packaged in this repository for independent peer replication.
