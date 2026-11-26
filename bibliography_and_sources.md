# Primary Literature Review & Source Documentation

**Research Title:** When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition  
**Conference:** IEEE SENSORS@25 (Cairo, Egypt • Dec 14–16, 2026)  
**Author:** Prince Chovatiya  

---

## 1. Structured Primary Literature Source Table

The following table summarizes the primary research papers, authoritative dataset documentation, and sensor fusion studies directly relevant to this contribution:

| Source Ref | Authors & Year | Paper Title | Venue / Publisher | DOI / Stable URL | Relevant Findings & Methodology | Relation to Our SENSORS@25 Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[1] Anguita13** | D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz (2013) | *A Public Domain Dataset for Human Activity Recognition Using Smartphones* | 21st European Symposium on Artificial Neural Networks (ESANN 2013) | [ESANN Digital Library](https://www.esann.org/proceedings/2013) | Introduced the benchmark 30-subject smartphone dataset using waist-mounted Samsung Galaxy S II (50 Hz). Extracted 561 features and evaluated multiclass Linear SVM, achieving 89% accuracy under pristine conditions. | Provides our foundational benchmark dataset, 70/30 subject split, and baseline reference. Our work extends their clean setup to controlled sensor degradation, noise, and axis dropout. |
| **[2] Reyes16** | J. L. Reyes-Ortiz, L. Oneto, A. Samà, V. Parra, D. Anguita (2016) | *Transition-Aware Human Activity Recognition Using Smartphones* | *Neurocomputing*, Vol. 171, pp. 754–767 | [doi:10.1016/j.neucom.2015.07.085](https://doi.org/10.1016/j.neucom.2015.07.085) | Modeled postural transitions (sit-to-stand, stand-to-sit) alongside steady-state activities. Highlighted that angular velocity signals are pivotal during rotational transitions. | Explains why gyroscope loss disproportionately impacts dynamic walking and transition phases, whereas static postures rely almost purely on gravity. |
| **[3] Shoaib14** | M. Shoaib, S. Bosch, O. D. Incel, H. Scholten, P. J. M. Havinga (2014) | *Fusion of Smartphone Sensors for Human Activity Recognition: A Complex Matter* | *MDPI Sensors*, Vol. 14, No. 6, pp. 10146–10170 | [doi:10.3390/s140610146](https://doi.org/10.3390/s140610146) | Evaluated individual and combined roles of accelerometers, gyroscopes, and magnetometers across multiple pocket/arm positions. Found accelerometers alone perform well for basic activities, but gyroscopes disambiguate stairs. | Confirms our finding that gyroscope dropout specifically degrades WALKING vs. STAIR recognition (WALKING F1 fell by 0.320), while accelerometer dropout destroys all static classification. |
| **[4] Zhang22** | S. Zhang, Y. Li, S. Zhang, F. Shahabi, S. Xia, Y. Deng, N. Alshurafa (2022) | *Deep Learning for Sensor-Based Human Activity Recognition: A Survey* | *ACM Computing Surveys*, Vol. 55, No. 2, pp. 1–37 | [doi:10.1145/3529090](https://doi.org/10.1145/3529090) | Comprehensive taxonomy of convolutional, recurrent, and attention architectures in HAR. Emphasized that missing sensor channels and heterogeneous sampling rates are major open deployment bottlenecks. | Bridges the gap between idealized deep models and real-world edge execution. Demonstrates why lightweight fallback architectures are needed on battery-constrained MCUs. |
| **[5] Chen15** | Y. Chen, Y. Xue (2015) | *A Deep Learning Approach to Human Activity Recognition Based on Single Accelerometer* | *IEEE International Conference on Systems, Man, and Cybernetics (SMC)* | [doi:10.1109/SMC.2015.263](https://doi.org/10.1109/SMC.2015.263) | Demonstrated that triaxial acceleration alone is sufficient for coarse activity classification if robust features are extracted, saving device energy by omitting gyroscopes. | Directly supports our Adaptive Fallback design: training an accelerometer-dedicated model restores classification to 81.17% (RF) / 84.37% (LR) without gyroscope energy draw. |
| **[6] Radu18** | V. Radu, C. Tong, S. Bhattacharya, N. D. Lane, C. Mascolo, M. K. Marina, F. Kawsar (2018) | *Multimodal Deep Learning for Activity and Context Recognition on Mobile Devices* | *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)* | [doi:10.1145/3161174](https://doi.org/10.1145/3161174) | Investigated cross-modal feature representations across motion, audio, and location sensors. Noted that naive concatenation models crash when sensory modalities are disconnected. | Our work confirms this vulnerability in inertial sensing: naive zero-filling in joint models yields inferior performance compared to an explicit unimodal fallback classifier (+9.2% F1 advantage). |
| **[7] InvenSense13** | TDK InvenSense (2013) | *MPU-6000 and MPU-6050 Product Specification & Electrical Characteristics* | TDK InvenSense Technical Datasheet Rev. 3.4 | [invensense.tdk.com](https://invensense.tdk.com/) | Official hardware electrical specs: Accelerometer operating current = 500 µA; Gyroscope operating current = 3.6 mA (over 7.2× higher active power draw). | Supplies the physical hardware basis for energy-driven gyroscope duty-cycling on mobile platforms. |
| **[8] IEEE-Sensors25** | IEEE Sensors Council (2026) | *IEEE SENSORS@25 Call for Interactive Presentations & Conference Guidelines* | IEEE Sensors Council, Cairo, Egypt | [sensors25.ieee-sensors.org](https://sensors25.ieee-sensors.org/) | Established single-track milestone event guidelines encouraging work-in-progress, live demonstrations, and forward-looking discussions on sensing challenges. | Direct conference alignment for our interactive presentation, live demo web application, and submission package. |

---

## 2. Positioning & What Our Small Experiment Adds

### What Is Already Established:
* Multimodal fusion (accelerometer + gyroscope) provides higher clean test accuracy than unimodal accelerometer sensing on standard benchmarks.
* Deep learning models suffer performance degradation when test distributions diverge from training distributions.
* Gyroscopes draw substantially more active electrical current than capacitive MEMS accelerometers.

### What Our Controlled Experiment Uniquely Adds:
1. **Empirical Measurement of Functional Asymmetry:** We provide the first direct side-by-side quantification showing that sensor dropout is radically asymmetric: Gyroscope loss retains 88.07% Macro-F1 (benign), whereas Accelerometer loss collapses performance to 5.83% (catastrophic).
2. **Identification of Fatal Physical Axes:** We map single-axis failure, demonstrating that the Total Acceleration X-axis (forward gravitational vector) causes an 83% performance drop when flatlined, whereas individual gyroscope axes cause less than 2% drop (except Body Gyro X, which causes 14.9% drop).
3. **Quantification of the Fallback Recovery Margin:** We demonstrate that zero-imputation / zero-filling in multimodal models leaves substantial performance on the table: deploying an adaptive single-sensor fallback model immediately recovers +3.53% (Random Forest) and +9.20% (Logistic Regression) Macro-F1.
4. **Noise Breakdown Slopes:** We map the empirical degradation curve across 9 levels of Gaussian noise ($\sigma \in [0.0, 2.0]$), showing tree ensembles maintain F1 > 0.77 under severe gyroscope jitter, but fail abruptly when accelerometer noise exceeds $\sigma = 0.10$.

---

## 3. Dataset Citation, Licensing & Provenance

* **Dataset Name:** Human Activity Recognition Using Smartphones Dataset (Version 1.0)
* **Authors:** Jorge L. Reyes-Ortiz, Alessandro Ghio, Luca Oneto, Davide Anguita, and Xavier Parra.
* **Institution:** Smartlab - Non-Linear Complex Systems Laboratory, DITEN - Università degli Studi di Genova, Genoa, Italy.
* **Repository:** UCI Machine Learning Repository (Dataset ID: 240)
* **Direct URI:** `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Ethical Approval & Consent:** The original authors documented that all 30 volunteers gave informed consent, and protocols were approved by institutional review.

---

## 4. Methodological Limitations & Non-Overclaiming Statement

To maintain strict scientific integrity, we explicitly acknowledge:
* **Subject Demographics:** The dataset represents 30 healthy volunteers aged 19–48. It does not generalize to geriatric populations, gait disorders, or pediatric movement patterns.
* **Device Placement:** The smartphone was rigidly attached to the waist via an elastic belt. The findings do not directly represent free-floating pocket inversions, backpack placement, or wrist-worn smartwatches.
* **Simulated Dropouts:** Sensor disconnections and noise were simulated via controlled mathematical perturbation on the raw 50 Hz signals; while physically grounded in sensor datasheet characteristics, they do not replace on-body hardware failure studies with physical faulty chips.
* **No Medical or Clinical Claims:** This work evaluates computational sensing reliability for consumer mobile applications and does not claim medical-grade diagnostic utility.
