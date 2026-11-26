"""
Comprehensive Experimental Pipeline for IEEE SENSORS@25 Contribution:
"When Sensors Go Quiet: A Controlled Study of Noise, Axis Degradation, and Modality Dropout in Smartphone Activity Recognition"

Dataset: UCI Human Activity Recognition Using Smartphones (Anguita et al.)
Split: Subject-independent designated train (70%, 21 subjects) and test (30%, 9 subjects)
Signal channels: 9 raw inertial signals (body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "UCI HAR Dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

# -------------------------------------------------------------
# 1. Dataset Loading
# -------------------------------------------------------------
def load_inertial_signals(subset):
    """
    Loads 9 raw inertial signal files for train or test subset.
    Returns: numpy array of shape (N, 128, 9)
    Channels:
      0, 1, 2: body_acc_x, body_acc_y, body_acc_z
      3, 4, 5: body_gyro_x, body_gyro_y, body_gyro_z
      6, 7, 8: total_acc_x, total_acc_y, total_acc_z
    """
    signal_names = [
        'body_acc_x_', 'body_acc_y_', 'body_acc_z_',
        'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
        'total_acc_x_', 'total_acc_y_', 'total_acc_z_'
    ]
    signals_data = []
    for sig in signal_names:
        filepath = os.path.join(DATASET_DIR, subset, 'Inertial Signals', f'{sig}{subset}.txt')
        data = np.loadtxt(filepath)
        signals_data.append(data)
    # Shape: (N, 128, 9)
    return np.transpose(signals_data, (1, 2, 0))

def load_data():
    print("Loading UCI HAR raw inertial signals and labels...")
    X_train_raw = load_inertial_signals('train')
    X_test_raw = load_inertial_signals('test')
    y_train = np.loadtxt(os.path.join(DATASET_DIR, 'train', 'y_train.txt'), dtype=int) - 1 # 0-indexed
    y_test = np.loadtxt(os.path.join(DATASET_DIR, 'test', 'y_test.txt'), dtype=int) - 1
    sub_train = np.loadtxt(os.path.join(DATASET_DIR, 'train', 'subject_train.txt'), dtype=int)
    sub_test = np.loadtxt(os.path.join(DATASET_DIR, 'test', 'subject_test.txt'), dtype=int)
    print(f"Train samples: {X_train_raw.shape[0]}, Test samples: {X_test_raw.shape[0]}")
    return X_train_raw, y_train, sub_train, X_test_raw, y_test, sub_test

# -------------------------------------------------------------
# 2. Defensible Feature Extraction Pipeline
# -------------------------------------------------------------
def extract_window_features(X_raw):
    """
    Extracts time- and frequency-domain features from raw inertial signals.
    X_raw: (N, 128, C) where C is number of channels (9, 6, or 3)
    """
    N, T, C = X_raw.shape
    
    # 1. Statistical moments per channel
    mean = np.mean(X_raw, axis=1) # (N, C)
    std = np.std(X_raw, axis=1)
    median = np.median(X_raw, axis=1)
    min_val = np.min(X_raw, axis=1)
    max_val = np.max(X_raw, axis=1)
    rms = np.sqrt(np.mean(X_raw**2, axis=1))
    iqr = stats.iqr(X_raw, axis=1)
    
    # 2. FFT frequency domain per channel
    fft_vals = np.abs(np.fft.rfft(X_raw, axis=1)) # (N, 65, C)
    energy = np.sum(fft_vals**2, axis=1) / float(T)
    dominant_freq = np.argmax(fft_vals[:, 1:, :], axis=1) + 1 # ignore DC component
    
    feats = [mean, std, median, min_val, max_val, rms, iqr, energy, dominant_freq]
    
    # 3. Multiaxial magnitude features if channels correspond to known triads
    # If 9 channels: body_acc (0:3), body_gyro (3:6), total_acc (6:9)
    if C == 9:
        body_acc_mag = np.sqrt(np.sum(X_raw[:, :, 0:3]**2, axis=2))
        body_gyro_mag = np.sqrt(np.sum(X_raw[:, :, 3:6]**2, axis=2))
        total_acc_mag = np.sqrt(np.sum(X_raw[:, :, 6:9]**2, axis=2))
        
        mag_feats = [
            np.mean(body_acc_mag, axis=1, keepdims=True),
            np.std(body_acc_mag, axis=1, keepdims=True),
            np.max(body_acc_mag, axis=1, keepdims=True),
            np.mean(body_gyro_mag, axis=1, keepdims=True),
            np.std(body_gyro_mag, axis=1, keepdims=True),
            np.max(body_gyro_mag, axis=1, keepdims=True),
            np.mean(total_acc_mag, axis=1, keepdims=True),
            np.std(total_acc_mag, axis=1, keepdims=True),
            np.max(total_acc_mag, axis=1, keepdims=True)
        ]
        
        sma_body_acc = np.mean(np.sum(np.abs(X_raw[:, :, 0:3]), axis=2), axis=1, keepdims=True)
        sma_body_gyro = np.mean(np.sum(np.abs(X_raw[:, :, 3:6]), axis=2), axis=1, keepdims=True)
        sma_total_acc = np.mean(np.sum(np.abs(X_raw[:, :, 6:9]), axis=2), axis=1, keepdims=True)
        
        feats.extend(mag_feats)
        feats.extend([sma_body_acc, sma_body_gyro, sma_total_acc])
        
    elif C == 6: # Acc only (body_acc + total_acc)
        body_acc_mag = np.sqrt(np.sum(X_raw[:, :, 0:3]**2, axis=2))
        total_acc_mag = np.sqrt(np.sum(X_raw[:, :, 3:6]**2, axis=2))
        mag_feats = [
            np.mean(body_acc_mag, axis=1, keepdims=True),
            np.std(body_acc_mag, axis=1, keepdims=True),
            np.max(body_acc_mag, axis=1, keepdims=True),
            np.mean(total_acc_mag, axis=1, keepdims=True),
            np.std(total_acc_mag, axis=1, keepdims=True),
            np.max(total_acc_mag, axis=1, keepdims=True)
        ]
        sma_body_acc = np.mean(np.sum(np.abs(X_raw[:, :, 0:3]), axis=2), axis=1, keepdims=True)
        sma_total_acc = np.mean(np.sum(np.abs(X_raw[:, :, 3:6]), axis=2), axis=1, keepdims=True)
        feats.extend(mag_feats)
        feats.extend([sma_body_acc, sma_total_acc])
        
    elif C == 3: # Gyro only
        body_gyro_mag = np.sqrt(np.sum(X_raw[:, :, 0:3]**2, axis=2))
        mag_feats = [
            np.mean(body_gyro_mag, axis=1, keepdims=True),
            np.std(body_gyro_mag, axis=1, keepdims=True),
            np.max(body_gyro_mag, axis=1, keepdims=True)
        ]
        sma_body_gyro = np.mean(np.sum(np.abs(X_raw[:, :, 0:3]), axis=2), axis=1, keepdims=True)
        feats.extend(mag_feats)
        feats.extend([sma_body_gyro])
        
    return np.hstack(feats)

# -------------------------------------------------------------
# 3. Model Training & Evaluation Helpers
# -------------------------------------------------------------
def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42)
    }

def fit_models(X_train_f, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    models = get_models()
    trained = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        print(f"  Trained {name} in {time.time()-t0:.2f}s")
        trained[name] = model
    return trained, scaler

def evaluate_models(trained_models, scaler, X_test_f, y_test):
    X_test_scaled = scaler.transform(X_test_f)
    results = {}
    for name, model in trained_models.items():
        preds = model.predict(X_test_scaled)
        macro_f1 = f1_score(y_test, preds, average='macro')
        bal_acc = balanced_accuracy_score(y_test, preds)
        per_class_f1 = f1_score(y_test, preds, average=None)
        cm = confusion_matrix(y_test, preds)
        results[name] = {
            "macro_f1": float(macro_f1),
            "balanced_acc": float(bal_acc),
            "per_class_f1": per_class_f1.tolist(),
            "confusion_matrix": cm.tolist(),
            "predictions": preds
        }
    return results

# -------------------------------------------------------------
# 4. Master Experiment Execution
# -------------------------------------------------------------
def run_all_experiments():
    X_train_raw, y_train, sub_train, X_test_raw, y_test, sub_test = load_data()
    
    # Channel slices
    # Multimodal: all 9 channels
    # Acc only: indices [0, 1, 2, 6, 7, 8] (body acc + total acc)
    # Gyro only: indices [3, 4, 5] (body gyro)
    acc_indices = [0, 1, 2, 6, 7, 8]
    gyro_indices = [3, 4, 5]
    
    print("\nExtracting clean features...")
    # Clean training features
    X_train_feat_multi = extract_window_features(X_train_raw)
    X_test_feat_multi = extract_window_features(X_test_raw)
    
    X_train_feat_acc = extract_window_features(X_train_raw[:, :, acc_indices])
    X_test_feat_acc = extract_window_features(X_test_raw[:, :, acc_indices])
    
    X_train_feat_gyro = extract_window_features(X_train_raw[:, :, gyro_indices])
    X_test_feat_gyro = extract_window_features(X_test_raw[:, :, gyro_indices])
    
    print(f"Multimodal feature dimension: {X_train_feat_multi.shape[1]}")
    print(f"Acc-only feature dimension:    {X_train_feat_acc.shape[1]}")
    print(f"Gyro-only feature dimension:   {X_train_feat_gyro.shape[1]}")
    
    # ---------------------------------------------------------
    # Train Baselines
    # ---------------------------------------------------------
    print("\n--- Training Multimodal Baseline Models ---")
    models_multi, scaler_multi = fit_models(X_train_feat_multi, y_train)
    clean_multi_res = evaluate_models(models_multi, scaler_multi, X_test_feat_multi, y_test)
    
    print("\n--- Training Accelerometer-Only Models ---")
    models_acc, scaler_acc = fit_models(X_train_feat_acc, y_train)
    clean_acc_res = evaluate_models(models_acc, scaler_acc, X_test_feat_acc, y_test)
    
    print("\n--- Training Gyroscope-Only Models ---")
    models_gyro, scaler_gyro = fit_models(X_train_feat_gyro, y_train)
    clean_gyro_res = evaluate_models(models_gyro, scaler_gyro, X_test_feat_gyro, y_test)
    
    # Store summary baseline metrics
    baseline_records = []
    for mod_name, res, config in [
        ("Multimodal (Acc+Gyro)", clean_multi_res, "Combined 9-channel"),
        ("Accelerometer-Only", clean_acc_res, "6-channel (Body+Total Acc)"),
        ("Gyroscope-Only", clean_gyro_res, "3-channel (Body Gyro)")
    ]:
        for model_name, m_data in res.items():
            baseline_records.append({
                "Configuration": mod_name,
                "Model": model_name,
                "Macro_F1": m_data["macro_f1"],
                "Balanced_Accuracy": m_data["balanced_acc"],
                "Channels": config
            })
    df_baselines = pd.DataFrame(baseline_records)
    df_baselines.to_csv(os.path.join(RESULTS_DIR, "clean_baselines.csv"), index=False)
    print("\nClean Baselines:")
    print(df_baselines)
    
    # ---------------------------------------------------------
    # Experiment 2: Modality Dropout under Deployment
    # ---------------------------------------------------------
    # Scenario A: Multimodal model encounters Silent Gyroscope (Gyro channels = 0)
    # Scenario B: Multimodal model encounters Silent Accelerometer (Acc channels = 0)
    # Scenario C: Adaptive Fallback to dedicated Acc-only model when Gyro is silent
    print("\n--- Experiment 2: Modality Dropout ---")
    X_test_silent_gyro = X_test_raw.copy()
    X_test_silent_gyro[:, :, gyro_indices] = 0.0 # Physical Gyroscope silent
    X_test_feat_silent_gyro = extract_window_features(X_test_silent_gyro)
    res_silent_gyro = evaluate_models(models_multi, scaler_multi, X_test_feat_silent_gyro, y_test)
    
    X_test_silent_acc = X_test_raw.copy()
    X_test_silent_acc[:, :, acc_indices] = 0.0 # Physical Accelerometer silent
    X_test_feat_silent_acc = extract_window_features(X_test_silent_acc)
    res_silent_acc = evaluate_models(models_multi, scaler_multi, X_test_feat_silent_acc, y_test)
    
    dropout_records = []
    for model_name in models_multi.keys():
        clean_f1 = clean_multi_res[model_name]["macro_f1"]
        
        # Multimodal with Gyro dropped
        gyro_drop_f1 = res_silent_gyro[model_name]["macro_f1"]
        gyro_drop_bal = res_silent_gyro[model_name]["balanced_acc"]
        
        # Multimodal with Acc dropped
        acc_drop_f1 = res_silent_acc[model_name]["macro_f1"]
        acc_drop_bal = res_silent_acc[model_name]["balanced_acc"]
        
        # Dedicated Acc fallback
        acc_fallback_f1 = clean_acc_res[model_name]["macro_f1"]
        acc_fallback_bal = clean_acc_res[model_name]["balanced_acc"]
        
        dropout_records.append({
            "Model": model_name,
            "Clean_Multimodal_F1": clean_f1,
            "Gyro_Silent_F1": gyro_drop_f1,
            "Gyro_Silent_F1_Retention": gyro_drop_f1 / clean_f1,
            "Acc_Silent_F1": acc_drop_f1,
            "Acc_Silent_F1_Retention": acc_drop_f1 / clean_f1,
            "Adaptive_Acc_Fallback_F1": acc_fallback_f1,
            "Fallback_Advantage_Over_Silent": acc_fallback_f1 - gyro_drop_f1
        })
    df_dropout = pd.DataFrame(dropout_records)
    df_dropout.to_csv(os.path.join(RESULTS_DIR, "modality_dropout_comparison.csv"), index=False)
    print("\nModality Dropout Comparison:")
    print(df_dropout)
    
    # ---------------------------------------------------------
    # Experiment 3: Physical IMU Axis Failure vs. Simulated Channel Ablation
    # ---------------------------------------------------------
    print("\n--- Experiment 3: Physical Axis Failure & Channel Ablation ---")
    
    # 3A: True Physical IMU Axis Failures
    # Physical Accelerometer Axis Failure drops BOTH total_acc and body_acc for that axis (since body_acc is derived from total_acc in software).
    # Physical Gyroscope Axis Failure drops body_gyro for that axis.
    physical_axes = [
        ("Physical Acc X", [0, 6]),
        ("Physical Acc Y", [1, 7]),
        ("Physical Acc Z", [2, 8]),
        ("Physical Gyro X", [3]),
        ("Physical Gyro Y", [4]),
        ("Physical Gyro Z", [5])
    ]
    phys_records = []
    for axis_label, ch_indices in physical_axes:
        X_test_phys_drop = X_test_raw.copy()
        X_test_phys_drop[:, :, ch_indices] = 0.0 # Physical axis hardware failure
        X_test_feat_phys = extract_window_features(X_test_phys_drop)
        phys_res = evaluate_models(models_multi, scaler_multi, X_test_feat_phys, y_test)
        
        for model_name, m_data in phys_res.items():
            clean_f1 = clean_multi_res[model_name]["macro_f1"]
            phys_records.append({
                "Physical_Axis": axis_label,
                "Coupled_Channels": str(ch_indices),
                "Model": model_name,
                "Macro_F1": m_data["macro_f1"],
                "Balanced_Acc": m_data["balanced_acc"],
                "F1_Retention": m_data["macro_f1"] / clean_f1,
                "Absolute_Drop": clean_f1 - m_data["macro_f1"]
            })
    df_phys_axis = pd.DataFrame(phys_records)
    df_phys_axis.to_csv(os.path.join(RESULTS_DIR, "physical_axis_dropout.csv"), index=False)
    print("\nPhysical Axis Degradation (Random Forest):")
    print(df_phys_axis[df_phys_axis["Model"] == "Random Forest"].sort_values(by="Absolute_Drop", ascending=False))

    # 3B: Simulated Single-Channel Mathematical Ablation
    channel_names = [
        (0, "Body Acc X"), (1, "Body Acc Y"), (2, "Body Acc Z"),
        (3, "Body Gyro X"), (4, "Body Gyro Y"), (5, "Body Gyro Z"),
        (6, "Total Acc X"), (7, "Total Acc Y"), (8, "Total Acc Z")
    ]
    chan_records = []
    for chan_idx, chan_label in channel_names:
        X_test_chan_drop = X_test_raw.copy()
        X_test_chan_drop[:, :, chan_idx] = 0.0 # Flatline isolated single signal channel
        X_test_feat_chan_drop = extract_window_features(X_test_chan_drop)
        chan_res = evaluate_models(models_multi, scaler_multi, X_test_feat_chan_drop, y_test)
        
        for model_name, m_data in chan_res.items():
            clean_f1 = clean_multi_res[model_name]["macro_f1"]
            chan_records.append({
                "Dropped_Channel": chan_label,
                "Channel_Index": chan_idx,
                "Model": model_name,
                "Macro_F1": m_data["macro_f1"],
                "Balanced_Acc": m_data["balanced_acc"],
                "F1_Retention": m_data["macro_f1"] / clean_f1,
                "Absolute_Drop": clean_f1 - m_data["macro_f1"]
            })
    df_chan = pd.DataFrame(chan_records)
    df_chan.to_csv(os.path.join(RESULTS_DIR, "simulated_channel_ablation.csv"), index=False)
    # Also write single_axis_dropout.csv for backward compatibility
    df_chan.to_csv(os.path.join(RESULTS_DIR, "single_axis_dropout.csv"), index=False)
    print("\nSimulated Single-Channel Ablation (RF Top Impact):")
    print(df_chan[df_chan["Model"] == "Random Forest"].sort_values(by="Absolute_Drop", ascending=False))
    
    # ---------------------------------------------------------
    # Experiment 4: Controlled Additive Gaussian Noise Sweeps
    # ---------------------------------------------------------
    print("\n--- Experiment 4: Controlled Additive Noise Sweeps ---")
    # Severity levels: sigma relative to normalized signal scale
    noise_sigmas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    noise_records = []
    
    # We test noise added to:
    # 1. Multimodal inputs
    # 2. Gyroscope channels only
    # 3. Accelerometer channels only
    for sigma in noise_sigmas:
        # Multimodal noise
        if sigma == 0.0:
            X_test_noisy_all = X_test_raw
            X_test_noisy_gyro = X_test_raw
            X_test_noisy_acc = X_test_raw
        else:
            noise = np.random.normal(0, sigma, X_test_raw.shape)
            X_test_noisy_all = X_test_raw + noise
            
            # Gyro only noise
            X_test_noisy_gyro = X_test_raw.copy()
            X_test_noisy_gyro[:, :, gyro_indices] += np.random.normal(0, sigma, X_test_raw[:, :, gyro_indices].shape)
            
            # Acc only noise
            X_test_noisy_acc = X_test_raw.copy()
            X_test_noisy_acc[:, :, acc_indices] += np.random.normal(0, sigma, X_test_raw[:, :, acc_indices].shape)
            
        feats_noisy_all = extract_window_features(X_test_noisy_all)
        feats_noisy_gyro = extract_window_features(X_test_noisy_gyro)
        feats_noisy_acc = extract_window_features(X_test_noisy_acc)
        
        res_noisy_all = evaluate_models(models_multi, scaler_multi, feats_noisy_all, y_test)
        res_noisy_gyro = evaluate_models(models_multi, scaler_multi, feats_noisy_gyro, y_test)
        res_noisy_acc = evaluate_models(models_multi, scaler_multi, feats_noisy_acc, y_test)
        
        for model_name in models_multi.keys():
            noise_records.append({
                "Noise_Sigma": sigma,
                "Model": model_name,
                "Corrupted_Modality": "All Channels",
                "Macro_F1": res_noisy_all[model_name]["macro_f1"],
                "Balanced_Acc": res_noisy_all[model_name]["balanced_acc"]
            })
            noise_records.append({
                "Noise_Sigma": sigma,
                "Model": model_name,
                "Corrupted_Modality": "Gyroscope Only",
                "Macro_F1": res_noisy_gyro[model_name]["macro_f1"],
                "Balanced_Acc": res_noisy_gyro[model_name]["balanced_acc"]
            })
            noise_records.append({
                "Noise_Sigma": sigma,
                "Model": model_name,
                "Corrupted_Modality": "Accelerometer Only",
                "Macro_F1": res_noisy_acc[model_name]["macro_f1"],
                "Balanced_Acc": res_noisy_acc[model_name]["balanced_acc"]
            })
    df_noise = pd.DataFrame(noise_records)
    df_noise.to_csv(os.path.join(RESULTS_DIR, "noise_robustness_sweeps.csv"), index=False)
    print("Completed noise sweeps.")
    
    # ---------------------------------------------------------
    # Experiment 5: Per-Activity Vulnerability Breakdown
    # ---------------------------------------------------------
    print("\n--- Experiment 5: Per-Activity Analysis ---")
    activity_records = []
    
    # Compare per-class F1 for: Clean, Gyro Silent, Acc Silent, High Noise (sigma=0.5)
    high_noise_feats = extract_window_features(X_test_raw + np.random.normal(0, 0.5, X_test_raw.shape))
    res_high_noise = evaluate_models(models_multi, scaler_multi, high_noise_feats, y_test)
    
    for model_name in ["Random Forest", "Logistic Regression"]:
        clean_class_f1 = clean_multi_res[model_name]["per_class_f1"]
        gyro_silent_class_f1 = res_silent_gyro[model_name]["per_class_f1"]
        acc_silent_class_f1 = res_silent_acc[model_name]["per_class_f1"]
        noise_class_f1 = res_high_noise[model_name]["per_class_f1"]
        
        for i, act_name in enumerate(ACTIVITY_NAMES):
            activity_records.append({
                "Model": model_name,
                "Activity": act_name,
                "Clean_F1": clean_class_f1[i],
                "Gyro_Silent_F1": gyro_silent_class_f1[i],
                "Acc_Silent_F1": acc_silent_class_f1[i],
                "Noise_Sigma_0_5_F1": noise_class_f1[i],
                "Gyro_Silent_Drop": clean_class_f1[i] - gyro_silent_class_f1[i],
                "Acc_Silent_Drop": clean_class_f1[i] - acc_silent_class_f1[i]
            })
    df_activities = pd.DataFrame(activity_records)
    df_activities.to_csv(os.path.join(RESULTS_DIR, "per_activity_vulnerability.csv"), index=False)
    print(df_activities[df_activities["Model"] == "Random Forest"])
    
    # Save raw confusion matrices
    np.savez(
        os.path.join(RESULTS_DIR, "confusion_matrices.npz"),
        rf_clean=np.array(clean_multi_res["Random Forest"]["confusion_matrix"]),
        rf_gyro_silent=np.array(res_silent_gyro["Random Forest"]["confusion_matrix"]),
        rf_acc_silent=np.array(res_silent_acc["Random Forest"]["confusion_matrix"]),
        rf_acc_fallback=np.array(clean_acc_res["Random Forest"]["confusion_matrix"]),
        lr_clean=np.array(clean_multi_res["Logistic Regression"]["confusion_matrix"]),
        lr_gyro_silent=np.array(res_silent_gyro["Logistic Regression"]["confusion_matrix"]),
        activities=ACTIVITY_NAMES
    )
    
    # ---------------------------------------------------------
    # 5. Generate Publication-Quality Figures
    # ---------------------------------------------------------
    print("\nGenerating publication-quality figures...")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")
    
    # FIGURE 1: Clean Baseline Model Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_baselines,
        x="Configuration",
        y="Macro_F1",
        hue="Model",
        palette="Blues_d"
    )
    plt.title("Clean Baseline Performance Across Sensor Modalities", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Macro-F1 Score", fontsize=12)
    plt.xlabel("Sensor Modality Configuration", fontsize=12)
    plt.ylim(0.5, 1.0)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_clean_baselines.png"), dpi=300)
    plt.close()
    
    # FIGURE 2: Noise Robustness Curves
    plt.figure(figsize=(9, 5.5))
    df_rf_noise = df_noise[df_noise["Model"] == "Random Forest"]
    sns.lineplot(
        data=df_rf_noise,
        x="Noise_Sigma",
        y="Macro_F1",
        hue="Corrupted_Modality",
        marker="o",
        linewidth=2.5,
        palette=["#e74c3c", "#3498db", "#2ecc71"]
    )
    plt.title("Random Forest Robustness Under Additive Gaussian Noise", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(r"Noise Severity ($\sigma$ / Signal Normal Scale)", fontsize=12)
    plt.ylabel("Macro-F1 Score", fontsize=12)
    plt.ylim(0.2, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Corrupted Modality", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_noise_robustness.png"), dpi=300)
    plt.close()
    
    # FIGURE 3: Modality Dropout & Adaptive Fallback
    plt.figure(figsize=(8.5, 5))
    dropout_plot_data = pd.DataFrame([
        {"Condition": "Clean Multimodal", "Macro_F1": clean_multi_res["Random Forest"]["macro_f1"], "Type": "Multimodal Model"},
        {"Condition": "Silent Gyroscope (0-input)", "Macro_F1": res_silent_gyro["Random Forest"]["macro_f1"], "Type": "Multimodal Model"},
        {"Condition": "Adaptive Acc Fallback", "Macro_F1": clean_acc_res["Random Forest"]["macro_f1"], "Type": "Fallback Model"},
        {"Condition": "Silent Accelerometer (0-input)", "Macro_F1": res_silent_acc["Random Forest"]["macro_f1"], "Type": "Multimodal Model"}
    ])
    ax = sns.barplot(
        data=dropout_plot_data,
        x="Condition",
        y="Macro_F1",
        hue="Type",
        dodge=False,
        palette=["#2b5c8f", "#e67e22"]
    )
    plt.title("Sensor Modality Failure & Fallback Resilience (Random Forest)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Macro-F1 Score", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=10, fontweight="bold",
                        xytext=(0, 4), textcoords='offset points')
    plt.legend(title="Strategy", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_dropout_and_fallback.png"), dpi=300)
    plt.close()
    
    # FIGURE 4: Physical IMU Axis Degradation vs. Channel Ablation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: True Physical IMU Axis Failure
    df_rf_phys = df_phys_axis[df_phys_axis["Model"] == "Random Forest"].sort_values(by="Absolute_Drop", ascending=False)
    ax0 = sns.barplot(
        data=df_rf_phys,
        x="Physical_Axis",
        y="Absolute_Drop",
        hue="Physical_Axis",
        palette="Reds_r",
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title("(A) Physical IMU Hardware Axis Failure", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Physical Sensor Axis (Coupled Triad)", fontsize=11)
    axes[0].set_ylabel("Absolute Macro-F1 Drop", fontsize=11)
    axes[0].tick_params(axis='x', rotation=25)
    for p in ax0.patches:
        val = p.get_height()
        if val > 0:
            ax0.annotate(f"-{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=9, fontweight="bold",
                         xytext=(0, 2), textcoords='offset points')
                         
    # Panel B: Simulated Single-Channel Ablation
    df_rf_chan = df_chan[df_chan["Model"] == "Random Forest"].sort_values(by="Absolute_Drop", ascending=False)
    ax1 = sns.barplot(
        data=df_rf_chan,
        x="Dropped_Channel",
        y="Absolute_Drop",
        hue="Dropped_Channel",
        palette="Blues_r",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("(B) Mathematical Single-Channel Ablation", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Isolated Signal Channel", fontsize=11)
    axes[1].set_ylabel("Absolute Macro-F1 Drop", fontsize=11)
    axes[1].tick_params(axis='x', rotation=35)
    for p in ax1.patches:
        val = p.get_height()
        if val > 0:
            ax1.annotate(f"-{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=8,
                         xytext=(0, 2), textcoords='offset points')
                         
    plt.suptitle("Sensor Failure Characterization: Physical Hardware vs. Simulated Signal Ablation (RF)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_single_axis_dropout.png"), dpi=300)
    plt.close()
    
    # FIGURE 5: Per-Activity Vulnerability
    plt.figure(figsize=(10, 5.5))
    df_rf_act = df_activities[df_activities["Model"] == "Random Forest"]
    df_melt = pd.melt(
        df_rf_act,
        id_vars=["Activity"],
        value_vars=["Clean_F1", "Gyro_Silent_F1", "Acc_Silent_F1", "Noise_Sigma_0_5_F1"],
        var_name="Condition",
        value_name="F1_Score"
    )
    condition_labels = {
        "Clean_F1": "Clean Baseline",
        "Gyro_Silent_F1": "Silent Gyroscope",
        "Acc_Silent_F1": "Silent Accelerometer",
        "Noise_Sigma_0_5_F1": "Gaussian Noise (σ=0.5)"
    }
    df_melt["Condition"] = df_melt["Condition"].map(condition_labels)
    ax = sns.barplot(
        data=df_melt,
        x="Activity",
        y="F1_Score",
        hue="Condition",
        palette=["#27ae60", "#f39c12", "#c0392b", "#8e44ad"]
    )
    plt.title("Per-Activity F1-Score Vulnerability Under Sensor Corruption (RF)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("F1-Score", fontsize=12)
    plt.xlabel("Human Activity Class", fontsize=12)
    plt.ylim(0.0, 1.1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Operating Condition", frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_per_activity_vulnerability.png"), dpi=300)
    plt.close()
    
    # FIGURE 6: Confusion Matrix Comparison (Clean vs Silent Gyro vs Adaptive Fallback)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cm_clean = np.array(clean_multi_res["Random Forest"]["confusion_matrix"])
    cm_gyro_silent = np.array(res_silent_gyro["Random Forest"]["confusion_matrix"])
    cm_acc_fallback = np.array(clean_acc_res["Random Forest"]["confusion_matrix"])
    
    # Short labels for plot
    short_labels = ["W", "WU", "WD", "SIT", "STD", "LAY"]
    
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=short_labels, yticklabels=short_labels)
    axes[0].set_title(f"A. Clean Multimodal (F1: {clean_multi_res['Random Forest']['macro_f1']:.3f})", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Activity", fontsize=11)
    axes[0].set_xlabel("Predicted Activity", fontsize=11)
    
    sns.heatmap(cm_gyro_silent, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=False,
                xticklabels=short_labels, yticklabels=short_labels)
    axes[1].set_title(f"B. Silent Gyroscope [No Fallback] (F1: {res_silent_gyro['Random Forest']['macro_f1']:.3f})", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Activity", fontsize=11)
    axes[1].set_xlabel("Predicted Activity", fontsize=11)
    
    sns.heatmap(cm_acc_fallback, annot=True, fmt='d', cmap='Greens', ax=axes[2], cbar=False,
                xticklabels=short_labels, yticklabels=short_labels)
    axes[2].set_title(f"C. Adaptive Acc Fallback (F1: {clean_acc_res['Random Forest']['macro_f1']:.3f})", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("True Activity", fontsize=11)
    axes[2].set_xlabel("Predicted Activity", fontsize=11)
    
    plt.suptitle("Confusion Matrix Dynamics Under Sensor Failure (W: Walk, WU: Upstairs, WD: Downstairs, SIT: Sit, STD: Stand, LAY: Lay)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_confusion_matrix_comparison.png"), dpi=300)
    plt.close()
    
    print("\nAll experiments and figures completed successfully!")
    return {
        "baselines": df_baselines.to_dict(orient="records"),
        "dropout": df_dropout.to_dict(orient="records"),
        "physical_axis": df_phys_axis.to_dict(orient="records"),
        "channel_ablation": df_chan.to_dict(orient="records")
    }

if __name__ == "__main__":
    t_start = time.time()
    results = run_all_experiments()
    print(f"\nTotal execution time: {time.time()-t_start:.2f} seconds.")
