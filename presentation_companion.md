# Presentation Companion: IEEE SENSORS@25

**Paper / Presentation Title:** When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition  
**Author:** Prince Chovatiya  
**Venue:** IEEE SENSORS@25, Giza Palace, Cairo, Egypt (Dec 14–16, 2026)  
**Session:** Interactive Presentation Session (LED Screen Display & Networking Dialogue)  
**Deliverable Link:** [presentation_IEEE_SENSORS25.pptx](file:///c:/Users/Prince%20Chovatiya/OneDrive/Desktop/conference%20paper/presentation_IEEE_SENSORS25.pptx)

---

## Part 1: The 60-Second Elevator Pitch

> *"Good afternoon. In Human Activity Recognition, our models routinely hit 90%+ accuracy in pristine lab conditions assuming both accelerometers and gyroscopes are always available. But on actual smartphones and wearables in the wild, sensors go quiet.*
>
> *Gyroscopes consume 5 to 10 times more battery than accelerometers, so mobile operating systems aggressively throttle or shut them down. Hardware glitches cause sensor axes to flatline, and pocket friction adds significant noise. When a multimodal model suddenly loses a sensor, what actually happens?*
>
> *In our controlled study on 10,299 standardized windows, we uncovered a striking functional asymmetry: if the gyroscope goes silent, the model retains 88% of its baseline performance—and by switching to an adaptive accelerometer-trained fallback model, we recover up to +9.2% accuracy over naive zero-padding. But if the accelerometer fails, the model suffers near-total collapse down to an F1 of 0.05 because the gravitational reference frame is wiped out.*
>
> *We’ve built an interactive demonstrator here today where you can drag noise sliders, kill individual sensor axes, and see exactly where models break—and how zero-cost architectural fallback protects edge devices."*

---

## Part 2: Slide-by-Slide 5-Minute Speaking Script

### Slide 1: Title and Core Thesis (0:00 – 0:40)
* **Visual:** Dark navy title slide with milestone badge and core thesis callout box.
* **Spoken Script:**
  > *"Welcome everyone. I’m Prince Chovatiya, presenting 'When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition.'*
  > *At this 25th anniversary milestone of IEEE SENSORS, we want to look beyond idealized benchmarks. Our core finding is simple yet critical for real-world IoT: multimodal sensor fusion conceals a massive structural asymmetry. When a gyroscope goes quiet, models can gracefully adapt; but when an accelerometer fails or is severely corrupted, the entire classification engine disintegrates. Today, I'll walk you through our empirical evidence and show how an adaptive fallback protocol recovers up to 9.2% classification accuracy at zero hardware cost."*

### Slide 2: Why Unreliable Sensor Measurements Matter (0:40 – 1:20)
* **Visual:** Three white structured cards covering Energy Asymmetry, Sensor Noise & Drift, and the Missing Modality Blindspot.
* **Spoken Script:**
  > *"Why does this matter so urgently for edge sensing? First, energy physics: a typical MEMS gyroscope draws around 3 to 6 milliamps, compared to less than 0.5 milliamps for an accelerometer. Android and iOS governors routinely throttle or sleep the gyroscope during background monitoring to prevent battery exhaustion.*
  > *Second, physical noise and driver timeouts frequently clip or desynchronize channels.*
  > *Yet, traditional HAR algorithms assume a fixed, immutable tensor shape is always fed to the classifier. When an axis or sensor drops, production code typically passes zeros or stale readings. We asked: how deep is that blindspot?"*

### Slide 3: Research Questions & Contributions (1:20 – 2:00)
* **Visual:** Split layout comparing four precise RQs on the left with our four distinct contributions on the right.
* **Spoken Script:**
  > *"We structured our investigation around four precise research questions: First, is sensor degradation symmetric between linear acceleration and angular velocity? Second, at what exact noise severity does fusion fail? Third, which individual axis is the fatal single point of failure? And fourth, can an adaptive fallback model outperform naive zero-imputation?*
  > *To answer this, our contribution evaluates 9 raw physical sensor streams directly—avoiding synthetic feature masks—and maps out the exact vulnerability profile across activities and sensor configurations."*

### Slide 4: Experimental Setup & Feature Extraction (2:00 – 2:40)
* **Visual:** UCI HAR dataset protocol (30 subjects, 70/30 subject split) and 93 kinematic feature specifications.
* **Spoken Script:**
  > *"For reproducibility, we leveraged the canonical UCI Human Activity Recognition dataset: 30 subjects, 10,299 windows sampled at 50 Hz, preserving the strict subject-independent 70/30 split to guarantee zero data leakage.*
  > *Instead of relying on black-box deep embeddings, we extracted 93 interpretable time- and frequency-domain features from the 9 raw physical channels—including statistical moments, FFT energy, dominant frequencies, and spatial Signal Magnitude Area.*
  > *We then benchmarked three lightweight, edge-compatible models: Random Forest, Logistic Regression, and Linear SVM."*

### Slide 5: Clean Baselines & Modality Comparison (2:40 – 3:20)
* **Visual:** Bar plot of clean baselines (Fig. 1) and quantitative metric cards.
* **Spoken Script:**
  > *"Looking at our clean baseline results on Slide 5: under full multimodal fusion, Linear SVM achieves 0.9117 Macro-F1, Logistic Regression hits 0.8968, and Random Forest achieves 0.8816.*
  > *Now notice what happens when we train strictly on single modalities: Accelerometer-only models retain over 92% to 94% of total multimodal accuracy (0.8604 on SVM).*
  > *However, gyroscope-only models drop sharply in linear classifiers down to ~70%, while Random Forest achieves 0.7917 because it can untangle complex non-linear rotational dynamics. This establishes that the accelerometer is the indispensable structural anchor."*

### Slide 6: Modality Dropout & Adaptive Fallback (3:20 – 4:00)
* **Visual:** Modality failure chart (Fig. 3) illustrating clean vs silent gyro vs adaptive fallback vs silent accelerometer.
* **Spoken Script:**
  > *"Slide 6 reveals our headline discovery: the extreme asymmetry of sensor dropout.*
  > *When a multimodal model experiences a silent gyroscope failure—meaning zeros in the gyro channels—Random Forest F1 drops from 0.8816 to 0.7764, retaining 88% of baseline performance.*
  > *But look at the orange bar on the right: if the accelerometer drops out, performance collapses catastrophically to 0.0514—pure random guessing! Why? Because the accelerometer provides the 1g gravity vector that anchors static postures like Sitting, Standing, and Laying.*
  > *Crucially, look at the green bar: when the gyroscope is lost, if our system dynamically invokes an Accelerometer-Trained Fallback model, Macro-F1 immediately rebounds to 0.8117 for RF, and jumps by +9.20% for Logistic Regression over naive zero-input padding. That is a pure software architectural gain with zero hardware overhead."*

### Slide 7: Noise Robustness & Per-Activity Vulnerability (4:00 – 4:35)
* **Visual:** Noise robustness curves (Fig. 2) and comparative confusion matrices (Fig. 6).
* **Spoken Script:**
  > *"On Slide 7, we examine noise resilience and per-activity vulnerabilities.*
  > *Under additive Gaussian noise sweeps, Random Forest exhibits remarkable immunity to gyroscope noise: even at high noise levels (sigma = 2.0), it maintains an F1 of 0.7731. But accelerometer noise beyond sigma = 0.10 immediately breaks the model.*
  > *Furthermore, our activity analysis reveals that WALKING is the single most vulnerable activity to gyroscope loss, with F1 plummeting from 0.8460 to 0.5257 as the model confuses walking with stair climbing. In contrast, LAYING is 100% immune because pitch and roll gravity components are entirely static."*

### Slide 8: Interactive Demo, Limitations & SENSORS@25 Dialogue (4:35 – 5:00)
* **Visual:** Demo interface overview, honest limitations, and forward-looking discussion questions.
* **Spoken Script:**
  > *"To bring this alive, we’ve deployed an interactive demonstrator right here on the screen where you can toggle modalities, inject sensor jitter, and watch the confusion matrix shift in real time.*
  > *To be scientifically transparent: this study is a controlled baseline on waist-mounted smartphones in healthy adults. Our ongoing work is validating this on an ARM Cortex-M4 microcontroller with physical InvenSense IMUs to measure exact battery savings in milliwatt-hours.*
  > *As we celebrate 25 years of IEEE SENSORS, we propose this question to the community: should our benchmarks require all multi-sensor papers to report a standardized 'Dropout Resilience Metric' alongside clean accuracy? Thank you, and I look forward to your questions."*

---

## Part 3: Anticipated Technical Questions & Scientifically Rigorous Answers

### Q1: "Why did you use the classic UCI HAR dataset rather than collecting your own new dataset with modern sensors?"
**Honest & Defensible Answer:**
> *"That was a deliberate methodological decision. In experimental reliability research, introducing a brand-new, unvalidated dataset creates confounding variables—you cannot easily separate sensor noise effects from idiosyncrasies in participant recruitment or uncalibrated custom hardware.
> The UCI HAR dataset is the gold standard benchmark in human activity recognition, cited over 4,000 times, with verified ground truth from synchronized video recordings and a strict, leak-free 70/30 subject split.
> By testing controlled noise and channel dropout on this universally recognized benchmark, our findings can be directly replicated, audited, and compared against hundreds of published baselines across the literature."*

### Q2: "Why didn't you train a deep Transformer or 1D-CNN rather than Random Forest and Logistic Regression?"
**Honest & Defensible Answer:**
> *"Our primary focus is resource-constrained edge computing on ultra-low-power microcontrollers and mobile background daemons. Complex Transformers or deep recurrent nets incur significant memory footprints and inference energy that negate the very power savings gained by duty-cycling the gyroscope.
> Random Forests and linear classifiers represent the primary lightweight inference models actually deployed on wearable MCUs (such as Cortex-M4 or ESP32). Furthermore, our results show that even simple Linear SVM achieves 91.17% clean Macro-F1 on our 93 extracted kinematic features—matching deep learning performance on this dataset while executing inference in under 0.2 milliseconds per window."*

### Q3: "In your modality dropout experiment, why does the accelerometer loss cause a near-total collapse (F1 = 0.05), whereas gyroscope loss only drops F1 to 0.77?"
**Honest & Defensible Answer:**
> *"The underlying physics of inertial sensing explains this fundamental asymmetry. The accelerometer measures both dynamic linear acceleration and the static 1g gravitational acceleration vector. This DC component provides the absolute vertical reference frame necessary to separate static postural states—Sitting, Standing, and Laying—which represent 50% of all activity windows in the dataset.
> In contrast, a gyroscope measures only angular velocity (deg/s). When a person is sitting or standing still, the angular velocity on all three axes is virtually zero. When the accelerometer is removed, the model receives zero meaningful signals for half the dataset, destroying its classification boundaries.
> Conversely, when the gyroscope is lost, the accelerometer's gravitational and dynamic acceleration vectors still retain sufficient spatial information to recognize posture and basic cadence, though it struggles to differentiate flat walking from stair climbing."*

### Q4: "How does your Adaptive Fallback model work in practice? Doesn't running two models consume more memory?"
**Honest & Defensible Answer:**
> *"The Adaptive Fallback protocol is extremely lightweight. Rather than deploying two large networks, the system maintains a primary multimodal classifier (e.g. 93 features) and an auxiliary accelerometer-only classifier (62 features). Because both models share the exact same accelerometer feature extraction code, the memory overhead is just the weights of a small Random Forest or Logistic Regression table—less than 150 kilobytes on disk.
> At runtime, a hardware sensor health monitor (which already checks I2C/SPI bus ACKs or driver interrupts) detects if the gyroscope has been duty-cycled or disconnected. If flagged silent, the edge device routes the 62 accelerometer features directly to the fallback classifier rather than passing zeros to the multimodal classifier, recovering up to +9.2% Macro-F1 at zero energy cost."*

### Q5: "How does your Gaussian noise model reflect real-world sensor noise?"
**Honest & Defensible Answer:**
> *"In our noise sweeps, we tested additive zero-mean Gaussian noise scaled by standard deviation $\sigma \in [0.0, 2.0]$ relative to normalized signal amplitudes. In MEMS sensors, thermal noise (Johnson-Nyquist noise) and amplifier white noise are standard Gaussian processes characterized by noise spectral density (e.g., $150 \ \mu\text{g}/\sqrt{\text{Hz}}$ for accelerometers and $0.005 \ \text{dps}/\sqrt{\text{Hz}}$ for gyroscopes).
> Additionally, mobile phones carried in loose pockets experience high-frequency stochastic friction and micro-impacts that act as wideband noise. By sweeping across 9 distinct $\sigma$ levels, we established the exact breakdown slope across both benign and extreme noise regimes."*

### Q6: "Why did WALKING drop so much more severely (F1 dropped from 0.846 to 0.525) than other activities when the gyroscope was lost?"
**Honest & Defensible Answer:**
> *"In gait kinematics, differentiating level walking from stair ascent and descent depends on the angular pitch and roll velocities of the pelvis/waist as the leg swings through the sagittal plane.
> While total acceleration can detect the step impact frequency, the rotational tilt profile captured by the gyroscope provides the clean discriminative boundary between flat stride cycles and vertical stair climbing steps. Without the gyroscope, the classifier frequently mislabels walking as walking upstairs or downstairs, causing WALKING precision and recall to collapse."*

### Q7: "What are the primary limitations of this study that an attendee should keep in mind?"
**Honest & Defensible Answer:**
> *"We emphasize three explicit limitations:
> First, spatial placement: the data is collected exclusively from a waist-mounted smartphone. While waist-worn sensing is ideal for center-of-mass kinematics, it does not directly capture wrist-worn smartwatch dynamics or pocket orientation inversions.
> Second, participant demographics: the dataset consists of 30 healthy young adults (19–48 years). It does not evaluate elderly gait instability or pathological gait disorders (e.g., Parkinsonian tremors).
> Third, this is a work-in-progress contribution: our battery power claims are based on physical IMU datasheet specs; on-device micro-Joulemeter hardware measurements are currently being gathered for the next phase of this research."*

### Q8: "How does this contribution align with the mission of IEEE SENSORS@25 in Cairo?"
**Honest & Defensible Answer:**
> *"IEEE SENSORS@25 is a special single-track milestone event designed to stimulate deep dialogue, reflect on 25 years of progress, and explore pragmatic frontier challenges in sensing.
> For two decades, our community has optimized classification accuracy under pristine conditions. This work directly addresses the practical reality of pervasive sensing: edge devices operate under strict power budgets where sensors must go quiet. By formalizing sensor dropout resilience, we hope to spur collaborative standardization of fault-tolerant benchmarks across the IEEE Sensors Council community."*

### Q9: "How do you distinguish simulated signal/feature corruption from true physical IMU hardware failure in your single-axis analysis?"
**Honest & Defensible Answer:**
> *"This is a crucial scientific distinction. A physical smartphone IMU consists of two physical sensor chip triads: a 3-axis accelerometer and a 3-axis gyroscope. In the UCI HAR dataset, 'total acceleration' is the raw physical accelerometer signal containing gravity, while 'body acceleration' is a software-derived signal calculated via a 0.3 Hz low-pass filter.
> In true physical hardware, you cannot lose total acceleration on the X-axis while derived body acceleration continues working—if the physical X-axis sensor beam or ADC disconnects, both signals fail together. In our physical IMU axis analysis, we coupled total and body signals for each accelerometer axis: when physical Acc X drops, Random Forest F1 collapses to 0.0514 because the static 1g gravitational reference is destroyed.
> Simultaneously, our mathematical single-channel ablation isolates the model's reliance on specific filtered components. We explicitly document both to ensure complete scientific clarity."*
