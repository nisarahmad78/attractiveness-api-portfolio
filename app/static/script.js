// =======================================================
// 🌟 Multi-face Attractiveness UI (CSS Synced Version)
// ✅ Sequential Upload + Dropdown Overlay + Compact No-Face Card
// ✅ Added: Click-to-view full image with transparent overlay + close button
// =======================================================

console.log("✅ Multi-face Attractiveness UI loaded (CSS synced)");

const form = document.getElementById("uploadForm");
const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");
const modelNameSpan = document.getElementById("model-name");
modelNameSpan.textContent = "EfficientNet-B3";

// ------------------------------------------------
// ⚙️ Status Message Helper
// ------------------------------------------------
function setStatus(msg, cls = "note", loading = false) {
  statusDiv.className = cls;
  statusDiv.textContent = "";
  if (loading) {
    const spin = document.createElement("span");
    spin.className = "spinner";
    statusDiv.appendChild(spin);
  }
  statusDiv.appendChild(document.createTextNode(msg));
}

// ------------------------------------------------
// 🧮 Score Extractor
// ------------------------------------------------
function getScoreValue(face) {
  if (!face) return null;
  const val = face.observed ?? face.fitted_constrained ?? face.fitted ?? null;
  if (val === null || val === undefined) return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

// ------------------------------------------------
// ✨ Format Score (0–10 range)
// ------------------------------------------------
function formatScore(face) {
  const n = getScoreValue(face);
  if (n === null) return "—";
  if (n < 0) return "0.00";
  if (n > 10) return "10.00";
  return n.toFixed(2);
}

// ------------------------------------------------
// 🎯 Dropdown Builder (auto-hide for single face)
// ------------------------------------------------
function createFaceSelector(faces, container, filename, previewUrl) {
  if (!faces || faces.length <= 1) return null;

  const selector = document.createElement("select");
  selector.className = "face-selector";

  faces.forEach((f, i) => {
    const gender = f.gender || "Unknown";
    const score = getScoreValue(f);
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `Face #${i + 1} — ${gender} — ${score ? score.toFixed(2) : "—"}`;
    selector.appendChild(opt);
  });

  selector.addEventListener("change", () => {
    const idx = parseInt(selector.value, 10);
    renderFaceDetails(faces[idx], container, filename, previewUrl);
  });

  return selector;
}

// ------------------------------------------------
// 🧍‍♂️ Single Face Card Renderer
// ------------------------------------------------
function renderFaceDetails(face, container, filename, previewUrl) {
  const scoreText = formatScore(face);
  const gender = face?.gender || "Unknown";

  container.innerHTML = `
    <div class="card">
      ${previewUrl ? `<img class="thumb" src="${previewUrl}" alt="${filename}" />` : ""}
      <div>
        <div class="fn">${filename}</div>
        <div class="${scoreText !== "—" ? "ok" : "muted"}">🌟 Score: ${scoreText}</div>
        <div class="muted">👤 Gender: ${gender}</div>
      </div>
    </div>
  `;

  // 🖱️ Click to enlarge image
  const img = container.querySelector(".thumb");
  if (img) img.addEventListener("click", () => openImageOverlay(previewUrl));
}

// ------------------------------------------------
// 🖼️ Render Image Result Block
// ------------------------------------------------
function renderImageResult(item) {
  const wrapper = document.createElement("div");
  wrapper.className = "result-block";

  // No face detected
  if (!item.faces?.length) {
    wrapper.innerHTML = `
      <div class="card missing">
        ${item.previewUrl ? `<img class="thumb" src="${item.previewUrl}" alt="${item.filename}" />` : ""}
        <div class="fn">${item.filename}</div>
        <div class="face-missing">⚠️ No faces detected</div>
      </div>
    `;

    // click enlarge even when no face detected
    const thumb = wrapper.querySelector(".thumb");
    if (thumb) thumb.addEventListener("click", () => openImageOverlay(item.previewUrl));
    return wrapper;
  }

  const faceContainer = document.createElement("div");
  const selector = createFaceSelector(item.faces, faceContainer, item.filename, item.previewUrl);

  if (selector) wrapper.appendChild(selector);
  wrapper.appendChild(faceContainer);

  renderFaceDetails(item.faces[0], faceContainer, item.filename, item.previewUrl);
  return wrapper;
}

// ------------------------------------------------
// 🔍 Match Result by Filename
// ------------------------------------------------
function findResultEntry(resultsArray, filename) {
  if (!Array.isArray(resultsArray)) return null;
  let entry = resultsArray.find((r) => r.filename === filename);
  if (entry) return entry;
  entry = resultsArray.find(
    (r) => r.filename && r.filename.toLowerCase() === filename.toLowerCase()
  );
  if (entry) return entry;
  entry = resultsArray.find(
    (r) => filename.endsWith(r.filename) || r.filename.endsWith(filename)
  );
  return entry ?? null;
}

// ------------------------------------------------
// 🚀 Sequential Upload (Real-time Update)
// ------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const input = document.getElementById("fileInput");
  let files = Array.from(input.files || []);
  if (!files.length) return alert("Please select image files.");

  if (files.length > 100) {
    const skipped = files.length - 100;
    files = files.slice(0, 100);
    setStatus(`⚠️ Only first 100 images uploaded — ${skipped} skipped.`, "error");
  }

  const submitBtn = form.querySelector("button[type='submit']");
  submitBtn.disabled = true;
  resultsDiv.classList.add("grid");
  resultsDiv.innerHTML = "";
  setStatus("🔄 Processing images one by one...", "note", true);

  let processedCount = 0;

  for (const f of files) {
    const previewUrl = URL.createObjectURL(f);
    const placeholder = document.createElement("div");
    placeholder.className = "card";
    placeholder.innerHTML = `
      <div class="fn">${f.name}</div>
      <div class="muted">⏳ Processing…</div>
    `;
    resultsDiv.appendChild(placeholder);

    try {
      const formData = new FormData();
      formData.append("files", f);

      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Server error");

      const serverEntry = findResultEntry(data.results, f.name) || {};
      const faces = Array.isArray(serverEntry.faces) ? serverEntry.faces : [];

      const resultBlock = renderImageResult({
        filename: f.name,
        previewUrl,
        faces,
      });

      resultsDiv.replaceChild(resultBlock, placeholder);
      processedCount++;
      setStatus(`✅ Processed ${processedCount}/${files.length}`, "success");
    } catch (err) {
      console.error("❌ Error:", err);
      placeholder.innerHTML = `
        <div class="card error">
          <div class="fn">${f.name}</div>
          <div class="face-missing">❌ Error processing file</div>
        </div>
      `;
      setStatus("❌ Some errors occurred during processing", "error");
    }
  }

  setStatus(`🎉 Done — Processed ${processedCount}/${files.length} image(s).`, "success");
  submitBtn.disabled = false;
});

// ------------------------------------------------
// 🪟 Fullscreen Image Overlay (Transparent + Close)
// ------------------------------------------------
function openImageOverlay(src) {
  const overlay = document.createElement("div");
  overlay.className = "image-overlay";

  const img = document.createElement("img");
  img.src = src;
  img.className = "overlay-img";

  const closeBtn = document.createElement("span");
  closeBtn.className = "close-btn";
  closeBtn.innerHTML = "✖";

  overlay.appendChild(img);
  overlay.appendChild(closeBtn);
  document.body.appendChild(overlay);

  const closeOverlay = () => {
    overlay.classList.add("fade-out");
    setTimeout(() => overlay.remove(), 300);
  };

  closeBtn.addEventListener("click", closeOverlay);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeOverlay();
  });

  document.addEventListener("keydown", function escHandler(e) {
    if (e.key === "Escape") {
      closeOverlay();
      document.removeEventListener("keydown", escHandler);
    }
  });
}
