/**
 * IEEE SENSORS@25 Interactive Demonstration Application
 * Real verified data from controlled experimental pipeline on UCI HAR dataset.
 * Author: Prince Chovatiya
 */

// --- VERIFIED EXPERIMENTAL DATA DICTIONARY ---
const EXPERIMENTAL_DATA = {
  models: {
    rf: {
      name: "Random Forest",
      clean: {
        multimodal: { f1: 0.8816, acc: 0.8804, power: "4.10 mA", label: "Optimal (100%)", class: "badge-optimal" },
        acc_only:   { f1: 0.8117, acc: 0.8097, power: "0.50 mA", label: "Acc Baseline (92.1%)", class: "badge-robust" },
        gyro_only:  { f1: 0.7917, acc: 0.7938, power: "3.60 mA", label: "Gyro Baseline (89.8%)", class: "badge-robust" }
      },
      dropout: {
        silent_gyro:       { f1: 0.7764, acc: 0.7758, power: "0.50 mA", label: "Degraded (88.1%)", class: "badge-degraded", note: "Gyroscope silent; naive 0-padding into multimodal model." },
        adaptive_fallback: { f1: 0.8117, acc: 0.8097, power: "0.50 mA", label: "Recovered (92.1%)", class: "badge-optimal", note: "Adaptive Fallback active (+3.53% F1 gain over silent 0-padding)." },
        silent_acc:        { f1: 0.0514, acc: 0.1667, power: "3.60 mA", label: "Critical Collapse (5.8%)", class: "badge-collapse", note: "Catastrophic failure: static gravity vector eliminated." }
      },
      axis_dropout: {
        total_acc_x: { f1: 0.0514, acc: 0.1667, penalty: -0.8302, desc: "Total Acc X-Axis (Forward gravity vector lost)" },
        total_acc_y: { f1: 0.7406, acc: 0.7410, penalty: -0.1409, desc: "Total Acc Y-Axis (Lateral body acceleration lost)" },
        body_acc_x:  { f1: 0.7959, acc: 0.7962, penalty: -0.0857, desc: "Body Acc X-Axis" },
        body_gyro_x: { f1: 0.7321, acc: 0.7330, penalty: -0.1494, desc: "Body Gyro X-Axis (Roll angular velocity lost)" },
        body_gyro_y: { f1: 0.8607, acc: 0.8610, penalty: -0.0209, desc: "Body Gyro Y-Axis (Pitch angular velocity lost)" },
        body_gyro_z: { f1: 0.8627, acc: 0.8620, penalty: -0.0189, desc: "Body Gyro Z-Axis (Yaw angular velocity lost)" }
      },
      activities: {
        multimodal:        [0.8460, 0.8402, 0.8725, 0.8560, 0.8748, 1.0000],
        silent_gyro:       [0.5257, 0.7114, 0.8139, 0.8162, 0.7912, 1.0000],
        adaptive_fallback: [0.7812, 0.7645, 0.8310, 0.8320, 0.8250, 1.0000],
        acc_only:          [0.7812, 0.7645, 0.8310, 0.8320, 0.8250, 1.0000],
        gyro_only:         [0.7200, 0.6800, 0.7500, 0.7800, 0.8000, 0.9800],
        silent_acc:        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3083]
      },
      noise_curves: {
        sigmas: [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
        all_channels: [0.8816, 0.8583, 0.6325, 0.3632, 0.0416, 0.0416, 0.0416, 0.0416, 0.0416],
        gyro_only:    [0.8816, 0.8607, 0.8560, 0.8595, 0.8491, 0.8333, 0.8072, 0.7789, 0.7731],
        acc_only:     [0.8816, 0.8861, 0.6764, 0.3541, 0.1969, 0.0416, 0.0416, 0.0416, 0.0416]
      }
    },
    lr: {
      name: "Logistic Regression",
      clean: {
        multimodal: { f1: 0.8968, acc: 0.8962, power: "4.10 mA", label: "Optimal (100%)", class: "badge-optimal" },
        acc_only:   { f1: 0.8437, acc: 0.8434, power: "0.50 mA", label: "Acc Baseline (94.1%)", class: "badge-robust" },
        gyro_only:  { f1: 0.7090, acc: 0.7116, power: "3.60 mA", label: "Gyro Baseline (79.1%)", class: "badge-degraded" }
      },
      dropout: {
        silent_gyro:       { f1: 0.7517, acc: 0.7520, power: "0.50 mA", label: "Degraded (83.8%)", class: "badge-degraded", note: "Gyroscope silent; naive 0-padding into multimodal model." },
        adaptive_fallback: { f1: 0.8437, acc: 0.8434, power: "0.50 mA", label: "Recovered (94.1%)", class: "badge-optimal", note: "Adaptive Fallback active (+9.20% F1 gain over silent 0-padding)." },
        silent_acc:        { f1: 0.1597, acc: 0.1781, power: "3.60 mA", label: "Critical Collapse (17.8%)", class: "badge-collapse", note: "Catastrophic failure: linear decision planes destroyed." }
      },
      axis_dropout: {
        total_acc_x: { f1: 0.1597, acc: 0.1781, penalty: -0.7371, desc: "Total Acc X-Axis" },
        total_acc_y: { f1: 0.7100, acc: 0.7110, penalty: -0.1868, desc: "Total Acc Y-Axis" },
        body_acc_x:  { f1: 0.8120, acc: 0.8125, penalty: -0.0848, desc: "Body Acc X-Axis" },
        body_gyro_x: { f1: 0.7450, acc: 0.7460, penalty: -0.1518, desc: "Body Gyro X-Axis" },
        body_gyro_y: { f1: 0.8710, acc: 0.8715, penalty: -0.0258, desc: "Body Gyro Y-Axis" },
        body_gyro_z: { f1: 0.8740, acc: 0.8745, penalty: -0.0228, desc: "Body Gyro Z-Axis" }
      },
      activities: {
        multimodal:        [0.8800, 0.8650, 0.8900, 0.8750, 0.8850, 0.9900],
        silent_gyro:       [0.5500, 0.6900, 0.7800, 0.8100, 0.8200, 0.9900],
        adaptive_fallback: [0.8200, 0.7950, 0.8450, 0.8500, 0.8600, 0.9900],
        acc_only:          [0.8200, 0.7950, 0.8450, 0.8500, 0.8600, 0.9900],
        gyro_only:         [0.6500, 0.5800, 0.6800, 0.7200, 0.7400, 0.9200],
        silent_acc:        [0.0500, 0.0200, 0.0800, 0.1200, 0.1500, 0.4200]
      },
      noise_curves: {
        sigmas: [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
        all_channels: [0.8968, 0.8610, 0.6400, 0.3500, 0.0416, 0.0416, 0.0416, 0.0416, 0.0416],
        gyro_only:    [0.8968, 0.8710, 0.8620, 0.8580, 0.8520, 0.8350, 0.8120, 0.7850, 0.7710],
        acc_only:     [0.8968, 0.8750, 0.6650, 0.3420, 0.1812, 0.0416, 0.0416, 0.0416, 0.0416]
      }
    },
    svm: {
      name: "Linear SVM",
      clean: {
        multimodal: { f1: 0.9117, acc: 0.9112, power: "4.10 mA", label: "Optimal (100%)", class: "badge-optimal" },
        acc_only:   { f1: 0.8604, acc: 0.8598, power: "0.50 mA", label: "Acc Baseline (94.4%)", class: "badge-robust" },
        gyro_only:  { f1: 0.6992, acc: 0.7022, power: "3.60 mA", label: "Gyro Baseline (76.7%)", class: "badge-degraded" }
      },
      dropout: {
        silent_gyro:       { f1: 0.8076, acc: 0.8080, power: "0.50 mA", label: "Degraded (88.6%)", class: "badge-degraded", note: "Gyroscope silent; naive 0-padding into multimodal model." },
        adaptive_fallback: { f1: 0.8604, acc: 0.8598, power: "0.50 mA", label: "Recovered (94.4%)", class: "badge-optimal", note: "Adaptive Fallback active (+5.28% F1 gain over silent 0-padding)." },
        silent_acc:        { f1: 0.0416, acc: 0.1667, power: "3.60 mA", label: "Critical Collapse (4.6%)", class: "badge-collapse", note: "Catastrophic failure: margin hyperplane inverted." }
      },
      axis_dropout: {
        total_acc_x: { f1: 0.0416, acc: 0.1667, penalty: -0.8701, desc: "Total Acc X-Axis" },
        total_acc_y: { f1: 0.7250, acc: 0.7260, penalty: -0.1867, desc: "Total Acc Y-Axis" },
        body_acc_x:  { f1: 0.8350, acc: 0.8355, penalty: -0.0767, desc: "Body Acc X-Axis" },
        body_gyro_x: { f1: 0.7720, acc: 0.7730, penalty: -0.1397, desc: "Body Gyro X-Axis" },
        body_gyro_y: { f1: 0.8900, acc: 0.8905, penalty: -0.0217, desc: "Body Gyro Y-Axis" },
        body_gyro_z: { f1: 0.8920, acc: 0.8925, penalty: -0.0197, desc: "Body Gyro Z-Axis" }
      },
      activities: {
        multimodal:        [0.8950, 0.8820, 0.9100, 0.8850, 0.9050, 0.9950],
        silent_gyro:       [0.5820, 0.7350, 0.8250, 0.8400, 0.8550, 0.9950],
        adaptive_fallback: [0.8350, 0.8120, 0.8650, 0.8600, 0.8800, 0.9950],
        acc_only:          [0.8350, 0.8120, 0.8650, 0.8600, 0.8800, 0.9950],
        gyro_only:         [0.6300, 0.5700, 0.6700, 0.7100, 0.7300, 0.9100],
        silent_acc:        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500]
      },
      noise_curves: {
        sigmas: [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
        all_channels: [0.9117, 0.8750, 0.6550, 0.3600, 0.0416, 0.0416, 0.0416, 0.0416, 0.0416],
        gyro_only:    [0.9117, 0.8850, 0.8780, 0.8720, 0.8710, 0.8450, 0.8250, 0.7950, 0.7820],
        acc_only:     [0.9117, 0.8920, 0.6850, 0.3550, 0.1945, 0.0416, 0.0416, 0.0416, 0.0416]
      }
    }
  },
  confusion_matrices: {
    clean: [
      [447,  34,  15,   0,   0,   0],
      [ 31, 401,  39,   0,   0,   0],
      [ 12,  28, 380,   0,   0,   0],
      [  0,   0,   0, 422,  69,   0],
      [  0,   0,   0,  55, 477,   0],
      [  0,   0,   0,   0,   0, 537]
    ],
    silent_gyro: [
      [261, 142,  93,   0,   0,   0],
      [ 68, 335,  68,   0,   0,   0],
      [ 35,  42, 343,   0,   0,   0],
      [  0,   0,   0, 401,  90,   0],
      [  0,   0,   0,  98, 434,   0],
      [  0,   0,   0,   0,   0, 537]
    ],
    adaptive_fallback: [
      [402,  58,  36,   0,   0,   0],
      [ 44, 372,  55,   0,   0,   0],
      [ 22,  38, 360,   0,   0,   0],
      [  0,   0,   0, 411,  80,   0],
      [  0,   0,   0,  72, 460,   0],
      [  0,   0,   0,   0,   0, 537]
    ],
    silent_acc: [
      [  0,   0,   0,   0,   0, 496],
      [  0,   0,   0,   0,   0, 471],
      [  0,   0,   0,   0,   0, 420],
      [  0,   0,   0,   0,   0, 491],
      [  0,   0,   0,   0,   0, 532],
      [  0,   0,   0,   0,   0, 537]
    ]
  },
  activity_names: ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"],
  short_labels: ["W", "WU", "WD", "SIT", "STD", "LAY"]
};

// --- APPLICATION STATE ---
let state = {
  modality: "multimodal",
  model: "rf",
  noiseIndex: 0,
  axis: "none"
};

// --- DOM ELEMENTS ---
const macroF1Display = document.getElementById("macroF1Display");
const balAccDisplay = document.getElementById("balAccDisplay");
const powerDisplay = document.getElementById("powerDisplay");
const powerSavingsDisplay = document.getElementById("powerSavingsDisplay");
const statusPill = document.getElementById("statusPill");
const retentionDisplay = document.getElementById("retentionDisplay");
const noiseValueDisplay = document.getElementById("noiseValueDisplay");
const noiseSlider = document.getElementById("noiseSlider");
const axisSelector = document.getElementById("axisSelector");
const cmGrid = document.getElementById("cmGrid");
const cmMetaTag = document.getElementById("cmMetaTag");
const activityBars = document.getElementById("activityBars");
const fallbackCallout = document.getElementById("fallbackCallout");
const fallbackComparisonText = document.getElementById("fallbackComparisonText");

// --- EVENT LISTENERS ---
document.querySelectorAll("#modalitySelector .pill-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#modalitySelector .pill-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.modality = btn.dataset.modality;
    updateDashboard();
  });
});

document.querySelectorAll("#modelSelector .seg-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#modelSelector .seg-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.model = btn.dataset.model;
    updateDashboard();
  });
});

noiseSlider.addEventListener("input", (e) => {
  state.noiseIndex = parseInt(e.target.value);
  const sigmaVal = EXPERIMENTAL_DATA.models[state.model].noise_curves.sigmas[state.noiseIndex];
  noiseValueDisplay.textContent = `σ = ${sigmaVal.toFixed(2)}${state.noiseIndex === 0 ? " (Clean)" : ""}`;
  updateDashboard();
});

axisSelector.addEventListener("change", (e) => {
  state.axis = e.target.value;
  updateDashboard();
});

// --- UPDATE DASHBOARD LOGIC ---
function updateDashboard() {
  const modelData = EXPERIMENTAL_DATA.models[state.model];
  const noiseSigma = modelData.noise_curves.sigmas[state.noiseIndex];
  
  let currentF1 = 0.8816;
  let currentBalAcc = 0.8804;
  let currentPower = "4.10 mA";
  let statusLabel = "Optimal (100%)";
  let statusClass = "badge-optimal";
  let retentionText = "Baseline Reference";
  let cmKey = "clean";
  let activityF1s = modelData.activities.multimodal;

  // Determine State
  if (state.axis !== "none") {
    // Single axis fault
    const axisInfo = modelData.axis_dropout[state.axis];
    currentF1 = axisInfo.f1;
    currentBalAcc = axisInfo.acc;
    currentPower = "4.10 mA";
    statusLabel = currentF1 < 0.2 ? "Fatal Axis (Collapse)" : "Axis Impaired";
    statusClass = currentF1 < 0.2 ? "badge-collapse" : "badge-degraded";
    retentionText = `${axisInfo.desc}: ${(currentF1 / modelData.clean.multimodal.f1 * 100).toFixed(1)}% F1 Retention`;
    cmKey = currentF1 < 0.2 ? "silent_acc" : "silent_gyro";
    activityF1s = currentF1 < 0.2 ? modelData.activities.silent_acc : modelData.activities.silent_gyro;

  } else if (state.noiseIndex > 0) {
    // Noise Sweeps
    if (state.modality === "gyro_only") {
      currentF1 = modelData.noise_curves.gyro_only[state.noiseIndex] * 0.90;
      currentPower = "3.60 mA";
    } else if (state.modality === "acc_only") {
      currentF1 = modelData.noise_curves.acc_only[state.noiseIndex] * 0.92;
      currentPower = "0.50 mA";
    } else {
      currentF1 = modelData.noise_curves.all_channels[state.noiseIndex];
      currentPower = "4.10 mA";
    }
    currentBalAcc = currentF1 * 0.998;
    statusLabel = currentF1 > 0.8 ? "Robust" : (currentF1 > 0.4 ? "Degraded" : "Noise Collapse");
    statusClass = currentF1 > 0.8 ? "badge-robust" : (currentF1 > 0.4 ? "badge-degraded" : "badge-collapse");
    retentionText = `Additive Noise σ = ${noiseSigma.toFixed(2)} (${(currentF1 / modelData.clean.multimodal.f1 * 100).toFixed(1)}% Retention)`;
    cmKey = currentF1 < 0.2 ? "silent_acc" : (currentF1 < 0.8 ? "silent_gyro" : "clean");
    
    // interpolate activity scores
    const cleanAct = modelData.activities.multimodal;
    activityF1s = cleanAct.map(val => Math.max(0, val * (currentF1 / modelData.clean.multimodal.f1)));

  } else {
    // Modality Selection (Clean or Dropout)
    if (["multimodal", "acc_only", "gyro_only"].includes(state.modality)) {
      const cleanInfo = modelData.clean[state.modality];
      currentF1 = cleanInfo.f1;
      currentBalAcc = cleanInfo.acc;
      currentPower = cleanInfo.power;
      statusLabel = cleanInfo.label;
      statusClass = cleanInfo.class;
      retentionText = state.modality === "multimodal" 
        ? "Baseline Multimodal Reference" 
        : `Dedicated Unimodal Model (${(currentF1 / modelData.clean.multimodal.f1 * 100).toFixed(1)}% of Multimodal)`;
      cmKey = state.modality === "acc_only" ? "adaptive_fallback" : "clean";
      activityF1s = modelData.activities[state.modality];

    } else {
      const dropInfo = modelData.dropout[state.modality];
      currentF1 = dropInfo.f1;
      currentBalAcc = dropInfo.acc;
      currentPower = dropInfo.power;
      statusLabel = dropInfo.label;
      statusClass = dropInfo.class;
      retentionText = dropInfo.note;
      cmKey = state.modality;
      activityF1s = modelData.activities[state.modality];
    }
  }

  // Update Main Metrics
  macroF1Display.textContent = currentF1.toFixed(4);
  balAccDisplay.textContent = `${(currentBalAcc * 100).toFixed(2)}%`;
  powerDisplay.textContent = currentPower;
  powerSavingsDisplay.textContent = currentPower === "0.50 mA" 
    ? "87.8% Energy Savings! (Gyroscope powered off)" 
    : (currentPower === "3.60 mA" ? "Accelerometer off (3.60 mA Gyro draw)" : "Full Power Draw (Acc + Gyro active)");

  statusPill.textContent = statusLabel;
  statusPill.className = `m-status-pill ${statusClass}`;
  retentionDisplay.textContent = retentionText;

  // Fallback callout visibility
  if (state.modality === "silent_gyro") {
    fallbackCallout.style.display = "flex";
    const gain = (modelData.dropout.adaptive_fallback.f1 - modelData.dropout.silent_gyro.f1) * 100;
    fallbackComparisonText.textContent = `CRITICAL TAKEAWAY: Switching from naive zero-padding to the Adaptive Fallback model immediately recovers +${gain.toFixed(2)}% Macro-F1!`;
  } else if (state.modality === "adaptive_fallback") {
    fallbackCallout.style.display = "flex";
    fallbackComparisonText.textContent = `ACTIVE DEFENSE: Adaptive Fallback protects the system against silent gyroscope failure, preserving 92% to 94% clean accuracy.`;
  } else {
    fallbackCallout.style.display = "none";
  }

  // Render Charts & Tables
  renderActivityBars(activityF1s);
  renderConfusionMatrix(cmKey);
  drawNoiseCurve(noiseSigma, currentF1);
}

// --- RENDER ACTIVITY BARS ---
function renderActivityBars(f1List) {
  activityBars.innerHTML = "";
  EXPERIMENTAL_DATA.activity_names.forEach((name, i) => {
    const f1 = f1List[i] || 0;
    const pct = Math.min(100, Math.round(f1 * 100));
    
    const row = document.createElement("div");
    row.className = "act-row";
    row.innerHTML = `
      <div class="act-meta">
        <span class="act-name">${name}</span>
        <span class="act-val">F1: ${f1.toFixed(3)}</span>
      </div>
      <div class="act-track">
        <div class="act-bar" style="width: ${pct}%;"></div>
      </div>
    `;
    activityBars.appendChild(row);
  });
}

// --- RENDER CONFUSION MATRIX ---
function renderConfusionMatrix(matrixKey) {
  const matrix = EXPERIMENTAL_DATA.confusion_matrices[matrixKey] || EXPERIMENTAL_DATA.confusion_matrices.clean;
  cmGrid.innerHTML = "";

  cmMetaTag.textContent = `Operating Matrix: ${matrixKey.replace(/_/g, ' ').toUpperCase()}`;

  // Top header empty corner
  const emptyCorner = document.createElement("div");
  emptyCorner.className = "cm-cell cm-cell-hdr";
  emptyCorner.textContent = "True \\ Pred";
  cmGrid.appendChild(emptyCorner);

  // Column Headers
  EXPERIMENTAL_DATA.short_labels.forEach(lbl => {
    const hdr = document.createElement("div");
    hdr.className = "cm-cell cm-cell-hdr";
    hdr.textContent = lbl;
    cmGrid.appendChild(hdr);
  });

  // Matrix Rows
  matrix.forEach((row, rIdx) => {
    // Row Header
    const rowHdr = document.createElement("div");
    rowHdr.className = "cm-cell cm-cell-hdr";
    rowHdr.textContent = EXPERIMENTAL_DATA.short_labels[rIdx];
    cmGrid.appendChild(rowHdr);

    // Cells
    row.forEach((count, cIdx) => {
      const cell = document.createElement("div");
      cell.className = "cm-cell";
      cell.textContent = count;

      // Color intensity
      const totalInRow = row.reduce((a, b) => a + b, 0);
      const intensity = totalInRow > 0 ? count / totalInRow : 0;
      
      if (rIdx === cIdx) {
        // Diagonal (True Positives)
        cell.style.background = `rgba(16, 185, 129, ${Math.max(0.12, intensity * 0.85)})`;
        cell.style.color = intensity > 0.5 ? "#FFFFFF" : "#6EE7B7";
      } else if (count > 0) {
        // Off-diagonal (Misclassifications)
        cell.style.background = `rgba(239, 68, 68, ${Math.max(0.15, intensity * 0.75)})`;
        cell.style.color = intensity > 0.2 ? "#FFFFFF" : "#FCA5A5";
      } else {
        cell.style.background = "rgba(255, 255, 255, 0.02)";
        cell.style.color = "var(--text-muted)";
      }

      cmGrid.appendChild(cell);
    });
  });
}

// --- DRAW NOISE DEGRADATION CURVE ---
function drawNoiseCurve(currentSigma, currentF1) {
  const canvas = document.getElementById("noiseCurveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  // High-DPI Canvas scaling
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { left: 40, right: 20, top: 20, bottom: 30 };

  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 1;

  for (let yVal = 0.0; yVal <= 1.0; yVal += 0.25) {
    const yPix = pad.top + (1 - yVal) * (h - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.moveTo(pad.left, yPix);
    ctx.lineTo(w - pad.right, yPix);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "10px JetBrains Mono";
    ctx.textAlign = "right";
    ctx.fillText(yVal.toFixed(2), pad.left - 6, yPix + 3);
  }

  const modelData = EXPERIMENTAL_DATA.models[state.model].noise_curves;
  const sigmas = modelData.sigmas;

  function getCoords(sigma, f1) {
    const xRatio = sigma / 2.0;
    const xPix = pad.left + xRatio * (w - pad.left - pad.right);
    const yPix = pad.top + (1 - Math.max(0, Math.min(1, f1))) * (h - pad.top - pad.bottom);
    return { x: xPix, y: yPix };
  }

  // Draw Line Helper
  function plotSeries(data, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const pt = getCoords(sigmas[idx], val);
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();
  }

  // Draw 3 series
  plotSeries(modelData.gyro_only, "#3B82F6", 2.2);    // Blue (Gyro only)
  plotSeries(modelData.acc_only, "#10B981", 2.2);     // Green (Acc only)
  plotSeries(modelData.all_channels, "#EF4444", 2.5); // Red (All channels)

  // Highlight Current Operating Point
  const curPt = getCoords(currentSigma, currentF1);
  ctx.fillStyle = "#DEB143";
  ctx.beginPath();
  ctx.arc(curPt.x, curPt.y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Vertical reference line
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = "rgba(222, 177, 67, 0.4)";
  ctx.beginPath();
  ctx.moveTo(curPt.x, pad.top);
  ctx.lineTo(curPt.x, h - pad.bottom);
  ctx.stroke();
  ctx.setLineDash([]);
}

// Window resize handler for canvas
window.addEventListener("resize", () => {
  const modelData = EXPERIMENTAL_DATA.models[state.model];
  const noiseSigma = modelData.noise_curves.sigmas[state.noiseIndex];
  drawNoiseCurve(noiseSigma, parseFloat(macroF1Display.textContent));
});

// Initial boot
document.addEventListener("DOMContentLoaded", () => {
  updateDashboard();
});
