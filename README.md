# IEEE SENSORS@25 Research Contribution & Reproducibility Package

**Title:** When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition  
**Conference:** IEEE SENSORS@25 (25th Anniversary Milestone Event, Cairo, Egypt • Dec 14–16, 2026)  
**Author:** Prince Chovatiya  
**Affiliation:** Independent Researcher  
**Session / Track:** Interactive Presentation Session (Frontier Sensing & Edge AI)  

---

## 📁 Repository Structure & Deliverables Inventory

```
conference paper/
│
├── SENSOR25_Abstract_Submission.docx   # Official IEEE SENSORS@25 submission-ready Abstract (Template preserved)
├── SENSOR25_Abstract_Submission.pdf    # Matching compiled PDF abstract (2-page format)
├── SENSOR25_Abstract_Template.docx     # Source conference template downloaded from website
├── abstract_submission.md              # Markdown formatted abstract and benchmark summary
│
├── presentation_IEEE_SENSORS25.pptx    # Official 8-slide editable PowerPoint deck (16:9 Widescreen)
├── presentation_IEEE_SENSORS25_display.pdf # Official digital display presentation file (0.61 MB < 25 MB limit)
├── presentation_slides/                # High-resolution slide images (Slide1.JPG to Slide8.JPG)
├── presentation_companion.md           # 60s elevator pitch, 5-min script, and 9 technical Q&As
│
├── run_experiments.py                  # Master standalone Python script executing all experiments (~42s)
├── reproduce_sensors25_experiments.ipynb # Fully executable Jupyter Notebook with inline visualizations
│
├── create_abstract_doc.py              # Generator script for SENSORS@25 docx & pdf abstract
├── create_presentation_deck.py         # Generator script for SENSORS@25 PPTX, PDF, and slide images
├── create_notebook.py                  # Generator script for the Jupyter notebook
│
├── bibliography_and_sources.md         # Structured primary literature review, DOIs, and positioning
├── limitations_and_assumptions.md      # Methodological assumptions and scientific integrity statement
├── email_to_chairs.md                  # Inquiry draft for deadline clarification & late consideration
│
├── dataset/                            # UCI HAR Dataset (raw inertial signals & ground-truth labels)
│   └── UCI HAR Dataset/
│       ├── train/ (Inertial Signals, X_train, y_train, subject_train)
│       └── test/  (Inertial Signals, X_test, y_test, subject_test)
│
├── results/                            # Actual computed CSV metric tables & confusion matrices
│   ├── clean_baselines.csv             # Clean model comparison across sensor modalities
│   ├── modality_dropout_comparison.csv # Silent Gyro vs Silent Acc vs Adaptive Fallback
│   ├── physical_axis_dropout.csv       # True physical IMU hardware axis degradation
│   ├── simulated_channel_ablation.csv  # Mathematical single-channel ablation breakdown
│   ├── single_axis_dropout.csv         # Consolidated single-axis dropout metrics
│   ├── noise_robustness_sweeps.csv     # 9-level additive Gaussian noise sweeps
│   ├── per_activity_vulnerability.csv  # Class-level F1 breakdown across conditions
│   └── confusion_matrices.npz          # NumPy arrays of all evaluated confusion matrices
│
├── figures/                            # Publication-quality 300 DPI figures
│   ├── fig1_clean_baselines.png        # Bar chart of clean baselines across modalities
│   ├── fig2_noise_robustness.png       # Robustness degradation curves under Gaussian noise
│   ├── fig3_dropout_and_fallback.png   # Modality failure & adaptive fallback advantage
│   ├── fig4_single_axis_dropout.png    # 2-panel plot: Physical IMU failure vs channel ablation
│   ├── fig5_per_activity_vulnerability.png # Activity vulnerability breakdown bar plot
│   └── fig6_confusion_matrix_comparison.png # 3-panel confusion matrix comparison
│
└── interactive_demo/                   # Interactive demonstration web app for conference LED screen
    ├── index.html                      # Interactive demonstration user interface
    ├── style.css                       # Responsive executive styling (Dark Navy & Warm Gold)
    └── app.js                          # Dynamic charts, real-time noise slider, and fallback simulator
```

---

## 🚀 Reproduction Instructions

### 1. Requirements & Environment
The code was executed and verified on Python 3.13. Standard dependencies:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn python-pptx python-docx
```

### 2. Run the Master Experimental Pipeline
To re-run all experiments, recalculate all metrics, and regenerate all figures:
```bash
python run_experiments.py
```
*Total execution time:* ~42 seconds on standard CPU.

### 3. Run the Interactive Jupyter Notebook
Launch Jupyter:
```bash
jupyter notebook reproduce_sensors25_experiments.ipynb
```
Step through each cell to inspect raw signal shapes, verify 93-dimensional feature extraction, and render inline plots.

### 4. Regenerate Submission Documents & Presentation
```bash
# Regenerate the official SENSORS@25 Abstract document (DOCX and PDF):
python create_abstract_doc.py

# Regenerate the PowerPoint presentation, PDF display file, and slide images:
python create_presentation_deck.py
```

### 5. Launch the Interactive Demo Web App
Open `interactive_demo/index.html` in any modern web browser or start a local HTTP server:
```bash
python -m http.server 8080
# Open http://localhost:8080/interactive_demo/index.html
```

---

## 📊 Summary of Verified Experimental Findings

* **Clean Multimodal Baseline:** Multimodal fusion achieves **0.8816** (Random Forest), **0.8968** (Logistic Regression), and **0.9117** (Linear SVM) Macro-F1.
* **Severe Modality Asymmetry:**
  * When the **Gyroscope** goes silent (0-input): Random Forest retains **88.07%** of clean performance (F1 = 0.7764).
  * When the **Accelerometer** goes silent (0-input): The model suffers near-total collapse, retaining only **5.83%** (F1 = 0.0514), because the static 1g gravitational reference vector is eliminated.
* **The Adaptive Fallback Advantage:** Switching dynamically to an Accelerometer-Trained Fallback model recovers **+3.53%** (Random Forest, F1 = 0.8117) to **+9.20%** (Logistic Regression, F1 = 0.8437) Macro-F1 over naive zero-imputed joint models.
* **Physical Hardware Coupling vs. Channel Ablation:**
  * When physical **Accelerometer Axis X** fails (destroying both total acceleration and dynamic body acceleration), Macro-F1 drops from 0.8816 to 0.0514 (collapse).
  * In contrast, physical **Gyroscope Axis Failures** are resilient: Gyro X retains 0.7321, Gyro Y retains 0.8606, and Gyro Z retains 0.8627.
* **Activity Vulnerability:** **WALKING** is the most vulnerable activity to gyroscope loss (F1 drops by **0.320** due to confusion with stairs), whereas **LAYING** is **100% immune** (F1 = 1.0000).
