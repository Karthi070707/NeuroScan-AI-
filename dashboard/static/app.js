/* ═══════════════════════════════════════════════════════════════════════════
   NeuroScan AI — Dashboard Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API = "http://localhost:8000";

const CLASS_COLORS = {
  glioma:      "#ef4444",
  meningioma:  "#f97316",
  notumor:     "#22c55e",
  pituitary:   "#3b82f6",
};
const CLASS_LABELS = {
  glioma:      "Glioma",
  meningioma:  "Meningioma",
  notumor:     "No Tumor",
  pituitary:   "Pituitary Tumor",
};

// ── State ─────────────────────────────────────────────────────────────────
let currentFile   = null;
let currentResult = null;
let probChart     = null;
let history       = [];
let currentMode   = "single";

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setInterval(checkHealth, 15000);
});

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    setStatus(d.model_ready ? "ok" : "warn",
              d.model_ready ? "Model Ready" : "Checkpoint Missing",
              d.device?.toUpperCase() || "CPU");
  } catch {
    setStatus("err", "Server Offline", "—");
  }
}

function setStatus(state, text, device) {
  const dot  = document.getElementById("statusDot");
  const txt  = document.getElementById("statusText");
  const dev  = document.getElementById("deviceBadge");
  dot.className  = "status-dot" + (state === "ok" ? " ok" : "");
  dot.style.background = state === "ok" ? "#22c55e" : state === "warn" ? "#f59e0b" : "#ef4444";
  dot.style.boxShadow  = `0 0 8px ${dot.style.background}`;
  txt.textContent = text;
  dev.textContent = device;
}

// ── Mode ──────────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById("btnSingle").classList.toggle("active", mode === "single");
  document.getElementById("btnBatch").classList.toggle("active", mode === "batch");
  if (mode === "batch") {
    document.getElementById("batchInput").click();
  } else {
    resetUpload();
  }
}

// ── Drag & Drop ───────────────────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById("uploadZone").classList.add("dragover");
}
function handleDragLeave(e) {
  document.getElementById("uploadZone").classList.remove("dragover");
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("uploadZone").classList.remove("dragover");
  const files = e.dataTransfer.files;
  if (files.length === 1) loadSingleFile(files[0]);
  else if (files.length > 1) handleBatchFiles(Array.from(files));
}

// ── File selection ────────────────────────────────────────────────────────
function handleFileSelect(e) {
  if (e.target.files.length > 0) loadSingleFile(e.target.files[0]);
}
function handleBatchSelect(e) {
  handleBatchFiles(Array.from(e.target.files));
}

function loadSingleFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  document.getElementById("previewImg").src = url;
  document.getElementById("infoName").textContent = file.name;
  document.getElementById("infoSize").textContent = formatBytes(file.size);

  const img = new Image();
  img.onload = () => {
    document.getElementById("infoDims").textContent = `${img.width}×${img.height}`;
  };
  img.src = url;

  document.getElementById("previewSection").style.display = "flex";
  document.getElementById("previewSection").style.flexDirection = "column";
  document.getElementById("resultCard").style.display  = "none";
  document.getElementById("gradcamCard").style.display = "none";
  document.getElementById("summaryCard").style.display = "none";
}

function resetUpload() {
  currentFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("previewSection").style.display = "none";
  document.getElementById("resultCard").style.display     = "none";
  document.getElementById("gradcamCard").style.display    = "none";
  document.getElementById("summaryCard").style.display    = "none";
}

// ── Analyze ───────────────────────────────────────────────────────────────
async function analyzeImage() {
  if (!currentFile) return;
  showLoading("Analyzing MRI scan…");
  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    const r = await fetch(`${API}/predict`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    currentResult = data;
    showResult(data);
    addToHistory(currentFile, data);
    showToast("Analysis complete ✓", "success");
  } catch (e) {
    showToast(`Error: ${e.message}`, "error");
    console.error(e);
  } finally {
    hideLoading();
  }
}

// ── Render result ─────────────────────────────────────────────────────────
function showResult(data) {
  const { predicted_class, predicted_label, confidence, probabilities, is_confident, color } = data;

  document.getElementById("resultCard").style.display = "block";
  document.getElementById("resultLabel").textContent   = predicted_label;
  document.getElementById("resultSublabel").textContent = `EfficientNet-B4 Classification`;

  const pill = document.getElementById("confPill");
  pill.textContent   = `${(confidence * 100).toFixed(1)}% confidence`;
  pill.style.color   = color;
  pill.style.background = `${color}18`;
  pill.style.borderColor = `${color}40`;

  document.getElementById("resultWarning").style.display = is_confident ? "none" : "block";

  // Donut chart
  const probs = Object.values(probabilities);
  const colors = Object.keys(probabilities).map(k => CLASS_COLORS[k]);
  renderProbChart(probs, colors);
  document.getElementById("chartCenter").textContent = `${(confidence * 100).toFixed(0)}%`;

  // Probability bars
  const barsDiv = document.getElementById("probBars");
  barsDiv.innerHTML = "";
  Object.entries(probabilities).forEach(([cls, prob]) => {
    const row  = document.createElement("div");
    row.className = "prob-row";
    const pct  = (prob * 100).toFixed(1);
    const clr  = CLASS_COLORS[cls];
    row.innerHTML = `
      <span class="prob-name">${CLASS_LABELS[cls]}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:${pct}%;background:${clr};"></div>
      </div>
      <span class="prob-val" style="color:${clr}">${pct}%</span>
    `;
    barsDiv.appendChild(row);
  });

  // Summary
  renderSummary(data);
}

function renderProbChart(probs, colors) {
  const ctx = document.getElementById("probChart").getContext("2d");
  if (probChart) probChart.destroy();
  probChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: probs,
        backgroundColor: colors.map(c => c + "cc"),
        borderColor    : colors,
        borderWidth    : 2,
        hoverOffset    : 6,
      }]
    },
    options: {
      cutout: "68%",
      animation: { animateRotate: true, duration: 800 },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    }
  });
}

function renderSummary(data) {
  const { predicted_label, confidence, is_confident, predicted_class } = data;
  const grid = document.getElementById("summaryGrid");
  const risk = predicted_class === "notumor" ? "None detected" :
               confidence > 0.9 ? "High" : confidence > 0.7 ? "Moderate" : "Low";
  const riskColor = predicted_class === "notumor" ? "#22c55e" :
                    confidence > 0.9 ? "#ef4444" : confidence > 0.7 ? "#f97316" : "#eab308";

  grid.innerHTML = `
    <div class="summary-item">
      <div class="summary-item-label">Classification</div>
      <div class="summary-item-val" style="color:${CLASS_COLORS[predicted_class]}">${predicted_label}</div>
      <div class="summary-item-sub">EfficientNet-B4</div>
    </div>
    <div class="summary-item">
      <div class="summary-item-label">Confidence</div>
      <div class="summary-item-val">${(confidence * 100).toFixed(1)}%</div>
      <div class="summary-item-sub">${is_confident ? "High confidence" : "⚠️ Low confidence"}</div>
    </div>
    <div class="summary-item">
      <div class="summary-item-label">Risk Indicator</div>
      <div class="summary-item-val" style="color:${riskColor}">${risk}</div>
      <div class="summary-item-sub">Based on classification</div>
    </div>
    <div class="summary-item">
      <div class="summary-item-label">XAI Method</div>
      <div class="summary-item-val" style="font-size:0.85rem">Grad-CAM</div>
      <div class="summary-item-sub">backbone.blocks[-1]</div>
    </div>
  `;
  document.getElementById("summaryCard").style.display = "block";
}

// ── Grad-CAM ──────────────────────────────────────────────────────────────
async function loadGradCAM() {
  if (!currentFile || !currentResult) return;
  document.getElementById("gradcamCard").style.display = "block";
  showCamLoading(true);
  showView("single");

  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    const r = await fetch(`${API}/gradcam`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();

    document.getElementById("camOriginal").src = `data:image/png;base64,${d.original_b64}`;
    document.getElementById("camOverlay").src  = `data:image/png;base64,${d.overlay_b64}`;
    showCamLoading(false);
    document.getElementById("singleView").style.display = "block";
    showToast("Grad-CAM generated ✓", "success");
  } catch (e) {
    showCamLoading(false);
    showToast(`Grad-CAM error: ${e.message}`, "error");
  }
}

async function loadAllClasses() {
  if (!currentFile || !currentResult) return;
  document.getElementById("gradcamCard").style.display = "block";
  showCamLoading(true);
  showView("all");

  try {
    const fd = new FormData();
    fd.append("file", currentFile);
    const r = await fetch(`${API}/all-classes`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();

    const grid = document.getElementById("allCamGrid");
    grid.innerHTML = "";
    d.class_overlays.forEach(item => {
      const div = document.createElement("div");
      div.className = "all-cam-item" + (item.is_predicted ? " predicted" : "");
      if (item.is_predicted) div.style.borderColor = item.color;
      div.innerHTML = `
        <img src="data:image/png;base64,${item.overlay_b64}" alt="${item.class_label}" />
        <div class="all-cam-footer">
          <span class="all-cam-name" style="color:${item.color}">${item.class_label}</span>
          <span class="all-cam-prob">${(item.probability * 100).toFixed(1)}%</span>
        </div>
      `;
      grid.appendChild(div);
    });

    showCamLoading(false);
    document.getElementById("allView").style.display  = "grid";
    showToast("All-class CAM complete ✓", "success");
  } catch (e) {
    showCamLoading(false);
    showToast(`Error: ${e.message}`, "error");
  }
}

function showView(which) {
  document.getElementById("singleView").style.display = which === "single" ? "block" : "none";
  document.getElementById("allView").style.display    = which === "all"    ? "grid"  : "none";
  document.querySelectorAll(".cam-tab").forEach((btn, i) => {
    btn.classList.toggle("active", (i === 0 && which === "single") || (i === 1 && which === "all"));
  });
}

function showCamLoading(show) {
  document.getElementById("camLoading").style.display = show ? "flex" : "none";
  if (show) {
    document.getElementById("singleView").style.display = "none";
    document.getElementById("allView").style.display    = "none";
  }
}

// ── Batch ─────────────────────────────────────────────────────────────────
async function handleBatchFiles(files) {
  showLoading(`Batch analyzing ${files.length} images…`);
  try {
    const fd = new FormData();
    files.forEach(f => fd.append("files", f));
    const r = await fetch(`${API}/batch`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(r.statusText);
    const d = await r.json();
    renderBatchResults(d.results);
    showToast(`Batch complete: ${d.total} images`, "success");
  } catch (e) {
    showToast(`Batch error: ${e.message}`, "error");
  } finally {
    hideLoading();
    setMode("single");
  }
}

function renderBatchResults(results) {
  // Add each result to history
  results.forEach(r => {
    if (!r.error) addToHistoryRaw(r.filename, r.predicted_label, r.predicted_class,
                                   r.confidence, r.color);
  });
  showToast(`${results.filter(r => !r.error).length} classified successfully`, "success");
}

// ── History ───────────────────────────────────────────────────────────────
function addToHistory(file, data) {
  const url   = URL.createObjectURL(file);
  const entry = {
    thumb    : url,
    name     : file.name,
    label    : data.predicted_label,
    cls      : data.predicted_class,
    conf     : data.confidence,
    color    : data.color,
    time     : new Date(),
    file,
    data,
  };
  history.unshift(entry);
  renderHistory();
}

function addToHistoryRaw(name, label, cls, conf, color) {
  history.unshift({ thumb: null, name, label, cls, conf, color, time: new Date() });
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById("historyList");
  if (history.length === 0) {
    list.innerHTML = '<div class="history-empty">No analyses yet. Upload an MRI to begin.</div>';
    return;
  }
  list.innerHTML = "";
  history.slice(0, 20).forEach((entry, i) => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.onclick = () => entry.data && restoreResult(entry);
    div.innerHTML = `
      ${entry.thumb
        ? `<img class="history-thumb" src="${entry.thumb}" alt="" />`
        : `<div class="history-thumb" style="background:rgba(255,255,255,0.05);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🧠</div>`}
      <div class="history-info">
        <div class="history-name">${entry.name}</div>
        <div class="history-class" style="color:${entry.color}">${entry.label}</div>
        <div class="history-conf">${(entry.conf * 100).toFixed(1)}% confidence</div>
      </div>
      <div class="history-time">${formatTime(entry.time)}</div>
    `;
    list.appendChild(div);
  });
}

function restoreResult(entry) {
  if (!entry.file) return;
  loadSingleFile(entry.file);
  currentResult = entry.data;
  showResult(entry.data);
}

function clearHistory() {
  history = [];
  renderHistory();
}

// ── Loading overlay ───────────────────────────────────────────────────────
function showLoading(text = "Analyzing…") {
  document.getElementById("loadingText").textContent = text;
  document.getElementById("loadingOverlay").style.display = "flex";
}
function hideLoading() {
  document.getElementById("loadingOverlay").style.display = "none";
}

// ── Toast ─────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent  = msg;
  el.className    = `toast ${type} show`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

// ── Utilities ─────────────────────────────────────────────────────────────
function formatBytes(b) {
  if (b < 1024)       return `${b} B`;
  if (b < 1048576)    return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1048576).toFixed(1)} MB`;
}
function formatTime(d) {
  const now = new Date();
  const diff = Math.floor((now - d) / 1000);
  if (diff < 60)   return "just now";
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
