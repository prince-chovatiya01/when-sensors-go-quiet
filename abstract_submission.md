# IEEE SENSORS@25 Abstract Submission

**Title:** When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition  
**Author:** Prince Chovatiya  
**Affiliation:** Independent Researcher  
**Contact Email:** pchovatiya@ieee.org  
**Conference:** IEEE SENSORS@25 (25th Anniversary Milestone Event, Cairo, Egypt, Dec 14–16, 2026)  
**Track / Session:** Interactive Presentation Session (Frontier Sensing & Edge AI)  
**Official Documents:**
- DOCX: [SENSOR25_Abstract_Submission.docx](file:///c:/Users/Prince%20Chovatiya/OneDrive/Desktop/conference%20paper/SENSOR25_Abstract_Submission.docx)
- PDF: [SENSOR25_Abstract_Submission.pdf](file:///c:/Users/Prince%20Chovatiya/OneDrive/Desktop/conference%20paper/SENSOR25_Abstract_Submission.pdf)

---

### Background & Motivation
Human Activity Recognition (HAR) using mobile inertial sensors is fundamental to digital health, gait kinematics, and context-aware mobile computing. While state-of-the-art models routinely report classification accuracies exceeding 90% under pristine laboratory conditions, real-world deployments suffer from significant sensing volatility. Mobile operating systems aggressively duty-cycle or suspend high-drain inertial measurement units—most notably gyroscopes, which consume 5–10× more active power than MEMS accelerometers—to extend device battery life. Furthermore, wearable devices encounter mechanical shocks, transient bus disconnects, sensor clipping, and thermal noise during continuous monitoring. Existing HAR literature overwhelmingly presumes complete, pristine multimodal sensor streams. Consequently, there is an urgent requirement to rigorously characterize the failure envelope of HAR models when sensor modalities or physical axes drop out, and to establish whether lightweight adaptive fallback mechanisms can safeguard against catastrophic failure.

### Novelty & Contribution
This contribution provides a systematic, controlled empirical investigation of sensor degradation across the full 9-channel inertial domain (triaxial body acceleration, triaxial angular velocity, and triaxial total acceleration). Unlike prior studies that restrict evaluation to synthetic feature masking or unimodal data, our contributions are three-fold:
1. **Quantification of Functional Asymmetry:** We demonstrate that while accelerometer loss causes catastrophic model failure (F1 retention < 6%), gyroscope loss preserves over 88% of baseline Macro-F1 across classifiers;
2. **Physical Hardware vs. Signal-Ablation Distinction:** We decouple mathematical signal-channel sensitivity from physical IMU triad failures, proving that physical Accelerometer X-axis failure (destroying gravitational orientation) is the primary single-point vulnerability (Macro-F1 drops from 0.8816 to 0.0514 in Random Forest), whereas single gyroscope axis failures are benign; and
3. **Zero-Cost Adaptive Fallback:** We show that dynamically rerouting inferences to a dedicated accelerometer-trained model during gyroscope silence recovers up to +9.20% Macro-F1 over naive zero-padding.

### Current State of Work
The experimental framework was implemented and executed on the benchmark UCI Smartphone Human Activity Recognition dataset (30 subjects, 10,299 standardized 2.56-second windows at 50 Hz, preserving the strict 70/30 subject-independent partition to eliminate data leakage). A leak-free feature extraction pipeline derived 93 time- and frequency-domain kinematic features from raw signals. Under clean multimodal sensing, models achieved Macro-F1 scores of 0.8816 (Random Forest), 0.8968 (Logistic Regression), and 0.9117 (Linear SVM). Under silent gyroscope dropout, Random Forest Macro-F1 was 0.7764 (88.07% retention), with WALKING exhibiting the highest vulnerability (F1 dropping from 0.8460 to 0.5257 due to stair confusion), while LAYING remained 100% immune (F1 = 1.000). Switching to a dedicated accelerometer fallback model restored Macro-F1 to 0.8117 (RF, +3.53% gain) and 0.8437 (LR, +9.20% gain). Gaussian noise sweeps ($\sigma = 0.0$ to $2.0$) confirmed that rotational gyroscope noise causes a mild, logarithmic decline (F1 = 0.7731 at $\sigma = 2.0$), whereas accelerometer noise causes catastrophic failure beyond $\sigma = 0.10$. All metrics, confusion matrices, and figures are fully verified.

### Future Work & Planned Outcomes
Planned next steps prior to and following SENSORS@25 include:
1. Extending the empirical evaluation to heterogeneous mobile sensor chipsets (e.g., InvenSense MPU-6050 vs. Bosch BMI160) to characterize silicon-specific drift and noise floors;
2. Benchmarking on-device dynamic power consumption on an ARM Cortex-M4 microcontroller to measure energy savings from selective gyroscope sleep; and
3. Packaging the adaptive fallback architecture into an open-source, edge-inference micro-library for resource-constrained wearables.

### Future Work & Planned Outcomes for the Interactive Session
During the Interactive Presentation Session at SENSORS@25 in Cairo, we will demonstrate a live interactive software prototype on the designated conference digital LED screen. Attendees will interactively toggle sensor channels, inject live Gaussian noise, simulate individual physical axis failures, and observe real-time classification changes alongside instantaneous adaptive fallback recovery. Our objective is to foster substantive technical dialogue with sensor designers, edge-AI engineers, and IEEE Sensors Council researchers on standardizing fault-resilience specifications for next-generation pervasive sensing.

---

### Quantitative Benchmark Summary (Verified Experimental Results)

| Operating Condition | Sensor Modality | RF Macro-F1 | LR Macro-F1 | SVM Macro-F1 | RF Retention | Operational Context |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline (Full)** | Multimodal (9-Ch) | **0.8816** | **0.8968** | **0.9117** | **100.0%** | Pristine lab baseline (Acc + Gyro) |
| **Clean Accelerometer-Only** | Acc Triads (6-Ch) | 0.8117 | 0.8437 | 0.8604 | 92.07% | Gyro permanently powered off (ultra-low power) |
| **Clean Gyroscope-Only** | Gyro Triad (3-Ch) | 0.7917 | 0.7090 | 0.6992 | 89.80% | Acc disconnected or clipped |
| **Silent Gyroscope (0-Input)** | Multimodal Model | 0.7764 | 0.7517 | 0.8076 | 88.07% | Unhandled runtime gyro shutdown |
| **Adaptive Acc Fallback** | Dedicated Acc Model | **0.8117** | **0.8437** | **0.8604** | **92.07%** | **+3.53% to +9.20% gain over silent zero-fill** |
| **Physical Acc X Failure** | Multimodal Model | 0.0514 | 0.1730 | 0.2153 | 5.83% | Gravity orientation forward-vector collapsed |
| **Physical Gyro X Failure** | Multimodal Model | 0.7321 | 0.8441 | 0.7916 | 83.05% | Angular pitch rate lost |
| **Additive Noise ($\sigma=0.5$, Gyro)** | Multimodal Model | 0.8491 | 0.8520 | 0.8710 | 96.31% | High rotational jitter / vibration |
| **Additive Noise ($\sigma=0.5$, Acc)** | Multimodal Model | 0.1969 | 0.1812 | 0.1945 | 22.33% | Severe linear impact / sensor noise |

---

### Key References
1. D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in *Proc. 21st Eur. Symp. Artif. Neural Netw. (ESANN)*, Bruges, Belgium, 2013, pp. 437–442.
2. IEEE Sensors Council, "Interactive Presentation Guidelines & Milestone Conference Program," *IEEE SENSORS@25*, Cairo, Egypt, Dec. 2026. Available: https://sensors25.ieee-sensors.org/
3. S. Zhang et al., "Deep learning for sensor-based human activity recognition: A survey," *ACM Computing Surveys*, vol. 55, no. 2, pp. 1–37, 2022.
4. Y. Chen and Y. Xue, "A deep learning approach to human activity recognition based on single accelerometer," in *Proc. IEEE Int. Conf. Syst., Man, Cybern.*, 2015, pp. 1488–1492.
