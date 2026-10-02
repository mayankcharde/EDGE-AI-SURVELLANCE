// ============================================================
// Surveillance Dashboard - Frontend Logic
// ============================================================

const tbody = document.getElementById("events-body");
const statsDiv = document.getElementById("stats");
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modal-title");
const modalBody = document.getElementById("modal-body");
const closeBtn = document.getElementById("modal-close");

let lastUnknownEventId = 0;

function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();
    osc.connect(gainNode);
    gainNode.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
    gainNode.gain.setValueAtTime(0.2, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) {
    console.error("Audio playback failed", e);
  }
}

async function fetchEvents() {
  const event = document.getElementById("filter-event").value;
  const severity = document.getElementById("filter-severity").value;
  const params = new URLSearchParams({ limit: 100 });
  if (event) params.set("event", event);
  if (severity) params.set("severity", severity);

  const res = await fetch("/api/events?" + params.toString());
  const data = await res.json();
  
  // Check for new UNKNOWN_PERSON events
  if (data && data.length > 0) {
    const newUnknowns = data.filter(e => e.event === "UNKNOWN_PERSON" && e.id > lastUnknownEventId);
    if (newUnknowns.length > 0) {
      lastUnknownEventId = Math.max(...newUnknowns.map(e => e.id));
      playAlertSound();
    }
  }

  renderEvents(data);
}

async function fetchStats() {
  const res = await fetch("/api/stats");
  const s = await res.json();
  const byType =
    Object.entries(s.by_type || {})
      .map(([k, v]) => `${k}: ${v}`)
      .join(" | ") || "no events yet";
  statsDiv.innerHTML = `<b>Total: ${s.total}</b> &nbsp; <span>${byType}</span>`;
}

function renderEvents(events) {
  tbody.innerHTML = "";
  if (!events.length) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:20px;">No events</td></tr>`;
    return;
  }
  for (const e of events) {
    const tr = document.createElement("tr");
    const snap = e.snapshot_path
      ? `<img src="/snapshots/${basename(e.snapshot_path)}" onclick="showDetail(${e.id})">`
      : "-";
    tr.innerHTML = `
      <td>${e.id}</td>
      <td>${new Date(e.timestamp).toLocaleString()}</td>
      <td>${e.camera || "-"}</td>
      <td>${e.person || "-"}</td>
      <td class="ev-${e.event}">${e.event}</td>
      <td><span class="sev-${e.severity}">${e.severity || "-"}</span></td>
      <td>${e.confidence?.toFixed(2) ?? "-"}</td>
      <td style="max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${e.explanation || "-"}</td>
      <td class="snap">${snap}</td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", (ev) => {
      if (ev.target.tagName !== "IMG") showDetail(e.id);
    });
    tbody.appendChild(tr);
  }
}

function basename(p) {
  return p.split(/[\\/]/).pop();
}

// ---------- Event Detail Modal ----------
async function showDetail(id) {
  const res = await fetch(`/api/events/${id}`);
  const e = await res.json();
  modalTitle.textContent = `Event #${e.id} — ${e.event}`;

  let xaiBlock = "";
  try {
    const md = JSON.parse(e.metadata || "{}");
    const xai = md.xai;
    if (xai) {
      xaiBlock = `
        <div class="xai-panel">
          <h3>🧠 AI Explanation (XAI)</h3>
          <p><b>Reasoning:</b> ${xai.text}</p>
          <p><b>Confidence:</b> ${xai.confidence}</p>
          <p><b>Feature Weights:</b></p>
          <pre>${JSON.stringify(xai.features, null, 2)}</pre>
        </div>
      `;
    }
  } catch (_) {}

  const snapHtml = e.snapshot_path
    ? `<img src="/snapshots/${basename(e.snapshot_path)}" style="max-width:100%;border-radius:8px;margin-top:16px;box-shadow:0 4px 12px rgba(0,0,0,0.3);">`
    : "";

  modalBody.innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom:16px;">
      <p><b>Time:</b><br/> ${new Date(e.timestamp).toLocaleString()}</p>
      <p><b>Camera:</b><br/> ${e.camera || "-"}</p>
      <p><b>Subject:</b><br/> ${e.person}</p>
      <p><b>Severity:</b><br/> <span class="sev-${e.severity}">${e.severity}</span></p>
    </div>
    <p><b>Event:</b> <span class="ev-${e.event}">${e.event}</span></p>
    <p><b>Confidence:</b> ${e.confidence}</p>
    <p><b>Overview:</b> ${e.explanation}</p>
    ${xaiBlock}
    ${snapHtml}
  `;
  modal.classList.add("active");
}

closeBtn.onclick = () => modal.classList.remove("active");
modal.onclick = (e) => {
  if (e.target === modal) modal.classList.remove("active");
};

// ---------- Filter / Refresh / Auto-refresh ----------
document.getElementById("refresh").onclick = () => {
  fetchEvents();
  fetchStats();
};
document.getElementById("filter-event").onchange = fetchEvents;
document.getElementById("filter-severity").onchange = fetchEvents;

let timer = null;
function setupAuto() {
  if (timer) clearInterval(timer);
  if (document.getElementById("auto").checked) {
    timer = setInterval(() => {
      fetchEvents();
      fetchStats();
    }, 5000);
  }
}
document.getElementById("auto").onchange = setupAuto;

// ---------- Source Control ----------
const sourceEl = document.getElementById("current-source");
const progressEl = document.getElementById("video-progress");
const uploadStatus = document.getElementById("upload-status");

async function refreshSource() {
  try {
    const r = await fetch("/api/source");
    const s = await r.json();
    sourceEl.textContent = s.current || "—";
    if (s.progress) {
      progressEl.textContent = `▶ ${s.progress.current}/${s.progress.total} (${s.progress.percent}%)`;
    } else {
      progressEl.textContent = "";
    }
  } catch (_) {}
}

// Webcam
document.getElementById("btn-webcam").onclick = async () => {
  const r = await fetch("/api/source/webcam", {
    method: "POST",
    body: new URLSearchParams({ index: "0" }),
  });
  const data = await r.json();
  alert("Switched to webcam: " + data.source);
  refreshSource();
};

// RTSP
document.getElementById("btn-rtsp").onclick = async () => {
  const url = document.getElementById("rtsp-input").value.trim();
  if (!url) return alert("Enter RTSP URL");
  const r = await fetch("/api/source/rtsp", {
    method: "POST",
    body: new URLSearchParams({ url }),
  });
  await r.json();
  alert("Switched to RTSP");
  refreshSource();
};

// Video upload
document.getElementById("video-file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  uploadStatus.textContent = "⏳ Uploading...";
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/source/video", { method: "POST", body: fd });
    const data = await r.json();
    uploadStatus.textContent = `✅ Loaded: ${data.source}`;
    refreshSource();
  } catch (err) {
    uploadStatus.textContent = "❌ Upload failed";
    console.error(err);
  }
};

// ---------- Initial load ----------
fetchEvents();
fetchStats();
setupAuto();
refreshSource();
setInterval(refreshSource, 3000);
