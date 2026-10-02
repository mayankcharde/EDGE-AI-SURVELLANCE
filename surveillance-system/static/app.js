// ============================================================
// SENTINEL — Edge AI Surveillance Dashboard
// Premium Frontend Logic with animations & toasts
// ============================================================

(() => {
  "use strict";

  // ── DOM References ──
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const tbody = $("#events-body");
  const modal = $("#modal");
  const modalTitle = $("#modal-title");
  const modalBody = $("#modal-body");
  const closeBtn = $("#modal-close");
  const clockEl = $("#clock");
  const toastContainer = $("#toast-container");
  const eventBadge = $("#event-count-badge");
  const eventCounter = $("#event-counter");
  const sourceEl = $("#current-source");
  const progressEl = $("#video-progress");
  const uploadStatus = $("#upload-status");
  const stopCameraBtn = $("#btn-stop-camera");
  const videoPanel = $("#video-panel");
  const sidebar = $("#sidebar");

  // ── State ──
  let lastUnknownEventId = 0;
  let prevStats = null;

  // =========================================================================
  //  CLOCK
  // =========================================================================
  function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    if (clockEl) clockEl.textContent = `${h}:${m}:${s}`;
  }
  setInterval(updateClock, 1000);
  updateClock();

  // =========================================================================
  //  TOAST NOTIFICATIONS
  // =========================================================================
  function showToast(message, type = "danger", duration = 5000) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <span>${type === "danger" ? "🚨" : type === "success" ? "✅" : "⚠️"}</span>
      <span>${message}</span>
    `;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("toast-out");
      toast.addEventListener("animationend", () => toast.remove());
    }, duration);
  }

  // =========================================================================
  //  ALERT SOUND
  // =========================================================================
  function playAlertSound() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.start();
      osc.stop(ctx.currentTime + 0.3);
    } catch (e) {
      console.error("Audio playback failed", e);
    }
  }

  // =========================================================================
  //  ANIMATED COUNTER
  // =========================================================================
  function animateNumber(el, target, duration = 600) {
    if (!el) return;
    const start = parseInt(el.textContent) || 0;
    if (start === target) return;
    const diff = target - start;
    const startTime = performance.now();

    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(start + diff * eased);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // =========================================================================
  //  FETCH EVENTS
  // =========================================================================
  async function fetchEvents() {
    try {
      const event = $("#filter-event").value;
      const severity = $("#filter-severity").value;
      const params = new URLSearchParams({ limit: 100 });
      if (event) params.set("event", event);
      if (severity) params.set("severity", severity);

      const res = await fetch("/api/events?" + params.toString());
      const data = await res.json();

      // Check for new UNKNOWN_PERSON events
      if (data && data.length > 0) {
        const newUnknowns = data.filter(
          (e) => e.event === "UNKNOWN_PERSON" && e.id > lastUnknownEventId
        );
        if (newUnknowns.length > 0) {
          lastUnknownEventId = Math.max(...newUnknowns.map((e) => e.id));
          playAlertSound();
          showToast(
            `${newUnknowns.length} unknown person${newUnknowns.length > 1 ? "s" : ""} detected!`,
            "danger"
          );
        }
      }

      renderEvents(data);
    } catch (err) {
      console.error("Failed to fetch events", err);
    }
  }

  // =========================================================================
  //  FETCH STATS
  // =========================================================================
  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      const s = await res.json();

      animateNumber($("#stat-total"), s.total || 0);

      // Sum critical events
      const criticalCount =
        (s.by_severity?.critical || 0) + (s.by_severity?.high || 0);
      animateNumber($("#stat-critical"), criticalCount);

      // Intrusion count
      animateNumber(
        $("#stat-intrusion"),
        s.by_type?.INTRUSION || 0
      );

      // Face events
      const faceCount =
        (s.by_type?.KNOWN_PERSON || 0) + (s.by_type?.UNKNOWN_PERSON || 0);
      animateNumber($("#stat-faces"), faceCount);

      // Update event count badge
      if (eventBadge) eventBadge.textContent = s.total || 0;

      prevStats = s;
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  }

  // =========================================================================
  //  RENDER EVENTS TABLE
  // =========================================================================
  function renderEvents(events) {
    if (!events || events.length === 0) {
      tbody.innerHTML = `
        <tr class="placeholder-row">
          <td colspan="9">
            <div class="empty-state">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.3">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              <p>No events detected yet</p>
            </div>
          </td>
        </tr>`;
      if (eventCounter) eventCounter.textContent = "0 events";
      return;
    }

    if (eventCounter) {
      eventCounter.textContent = `${events.length} event${events.length !== 1 ? "s" : ""}`;
    }

    // Build rows
    const fragment = document.createDocumentFragment();

    for (const e of events) {
      const tr = document.createElement("tr");

      // Row class for known/unknown highlight
      if (e.event === "KNOWN_PERSON") tr.classList.add("row-known");
      if (e.event === "UNKNOWN_PERSON") tr.classList.add("row-unknown");

      // Snapshot
      const snap = e.snapshot_path
        ? `<img src="/snapshots/${basename(e.snapshot_path)}" loading="lazy" alt="Snapshot" />`
        : '<span style="color: var(--text-muted);">—</span>';

      // Confidence display
      const confPercent = e.confidence != null ? (e.confidence * 100).toFixed(0) : null;
      const confHtml = confPercent != null
        ? `<div class="confidence-bar">
             <div class="conf-track"><div class="conf-fill" style="width:${confPercent}%"></div></div>
             <span class="conf-value">${confPercent}%</span>
           </div>`
        : '<span style="color: var(--text-muted);">—</span>';

      // Truncated explanation
      const explanation = e.explanation || "—";
      const shortExplanation =
        explanation.length > 50
          ? explanation.substring(0, 50) + "…"
          : explanation;

      tr.innerHTML = `
        <td style="font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-muted);">${e.id}</td>
        <td style="font-size:12px; white-space:nowrap;">${formatTime(e.timestamp)}</td>
        <td>${e.camera || "—"}</td>
        <td style="font-weight:500;">${e.person || "—"}</td>
        <td><span class="ev-badge ev-${e.event}">${formatEventName(e.event)}</span></td>
        <td><span class="sev-badge sev-${e.severity}">${e.severity || "—"}</span></td>
        <td>${confHtml}</td>
        <td style="max-width:200px; color:var(--text-secondary);" title="${escapeHtml(explanation)}">${shortExplanation}</td>
        <td class="snap">${snap}</td>
      `;

      tr.addEventListener("click", (ev) => {
        if (ev.target.tagName !== "IMG") showDetail(e.id);
      });

      fragment.appendChild(tr);
    }

    tbody.innerHTML = "";
    tbody.appendChild(fragment);
  }

  // =========================================================================
  //  EVENT DETAIL MODAL
  // =========================================================================
  async function showDetail(id) {
    try {
      const res = await fetch(`/api/events/${id}`);
      const e = await res.json();

      modalTitle.textContent = `Event #${e.id} — ${formatEventName(e.event)}`;

      // Build XAI block
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
            </div>`;
        }
      } catch (_) { }

      // Snapshot
      const snapHtml = e.snapshot_path
        ? `<img src="/snapshots/${basename(e.snapshot_path)}" class="modal-snapshot" alt="Event snapshot" />`
        : "";

      // Confidence
      const confPercent = e.confidence != null ? (e.confidence * 100).toFixed(1) : "—";

      modalBody.innerHTML = `
        <div class="detail-grid">
          <div class="detail-item">
            <span class="detail-item-label">Timestamp</span>
            <span class="detail-item-value">${new Date(e.timestamp).toLocaleString()}</span>
          </div>
          <div class="detail-item">
            <span class="detail-item-label">Camera</span>
            <span class="detail-item-value">${e.camera || "—"}</span>
          </div>
          <div class="detail-item">
            <span class="detail-item-label">Subject</span>
            <span class="detail-item-value">${e.person || "—"}</span>
          </div>
          <div class="detail-item">
            <span class="detail-item-label">Severity</span>
            <span class="detail-item-value"><span class="sev-badge sev-${e.severity}">${e.severity}</span></span>
          </div>
        </div>
        <p><b>Event:</b> <span class="ev-badge ev-${e.event}">${formatEventName(e.event)}</span></p>
        <p><b>Confidence:</b> ${confPercent}%</p>
        <p><b>Overview:</b> ${e.explanation || "—"}</p>
        ${xaiBlock}
        ${snapHtml}
      `;

      modal.classList.add("active");
    } catch (err) {
      console.error("Failed to load event details", err);
    }
  }

  // Modal close handlers
  closeBtn.addEventListener("click", () => modal.classList.remove("active"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("active");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      modal.classList.remove("active");
    }
  });

  // =========================================================================
  //  UTILITY FUNCTIONS
  // =========================================================================
  function basename(p) {
    return p.split(/[\\/]/).pop();
  }

  function formatTime(ts) {
    const d = new Date(ts);
    const h = String(d.getHours()).padStart(2, "0");
    const m = String(d.getMinutes()).padStart(2, "0");
    const s = String(d.getSeconds()).padStart(2, "0");
    const month = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${month}/${day} ${h}:${m}:${s}`;
  }

  function formatEventName(name) {
    if (!name) return "—";
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // =========================================================================
  //  FILTER / REFRESH / AUTO-REFRESH
  // =========================================================================
  $("#refresh").addEventListener("click", () => {
    fetchEvents();
    fetchStats();
    // Pulse animation on refresh button
    const btn = $("#refresh");
    btn.style.transform = "rotate(360deg)";
    setTimeout(() => (btn.style.transform = ""), 500);
  });

  $("#filter-event").addEventListener("change", fetchEvents);
  $("#filter-severity").addEventListener("change", fetchEvents);

  let timer = null;
  function setupAuto() {
    if (timer) clearInterval(timer);
    if ($("#auto").checked) {
      timer = setInterval(() => {
        fetchEvents();
        fetchStats();
      }, 5000);
    }
  }
  $("#auto").addEventListener("change", setupAuto);

  // =========================================================================
  //  SOURCE CONTROL
  // =========================================================================
  async function refreshSource() {
    try {
      const r = await fetch("/api/source");
      const s = await r.json();
      if (sourceEl) sourceEl.textContent = s.current || "—";
      if (s.progress && progressEl) {
        progressEl.textContent = `▶ ${s.progress.current}/${s.progress.total} (${s.progress.percent}%)`;
      } else if (progressEl) {
        progressEl.textContent = "";
      }

      if (stopCameraBtn) {
        stopCameraBtn.style.display = s.active ? "inline-flex" : "none";
      }
    } catch (_) { }
  }

  // Webcam
  $("#btn-webcam").addEventListener("click", async () => {
    try {
      const r = await fetch("/api/source/webcam", {
        method: "POST",
        body: new URLSearchParams({ index: "0" }),
      });
      const data = await r.json();
      showToast(`Switched to webcam: ${data.source}`, "success");
      refreshSource();
    } catch (err) {
      showToast("Failed to switch to webcam", "danger");
    }
  });

  // RTSP
  $("#btn-rtsp").addEventListener("click", async () => {
    const url = $("#rtsp-input").value.trim();
    if (!url) {
      showToast("Please enter an RTSP URL", "warning");
      return;
    }
    try {
      const r = await fetch("/api/source/rtsp", {
        method: "POST",
        body: new URLSearchParams({ url }),
      });
      await r.json();
      showToast("Connected to RTSP stream", "success");
      refreshSource();
    } catch (err) {
      showToast("Failed to connect RTSP", "danger");
    }
  });

  // Video upload
  $("#video-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (uploadStatus) uploadStatus.textContent = "⏳ Uploading…";
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch("/api/source/video", { method: "POST", body: fd });
      const data = await r.json();
      if (uploadStatus) uploadStatus.textContent = "";
      showToast(`Video loaded: ${data.source}`, "success");
      refreshSource();
    } catch (err) {
      if (uploadStatus) uploadStatus.textContent = "";
      showToast("Upload failed", "danger");
      console.error(err);
    }
  });

  // Stop Camera
  if (stopCameraBtn) {
    stopCameraBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/source/stop", { method: "POST" });
        showToast("Camera disconnected", "success");
        refreshSource();
      } catch (err) {
        showToast("Failed to disconnect camera", "danger");
      }
    });
  }

  // =========================================================================
  //  FULLSCREEN VIDEO
  // =========================================================================
  const fsBtn = $("#btn-fullscreen");
  if (fsBtn) {
    fsBtn.addEventListener("click", () => {
      videoPanel.classList.toggle("fullscreen");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && videoPanel.classList.contains("fullscreen")) {
        videoPanel.classList.remove("fullscreen");
      }
    });
  }

  // =========================================================================
  //  SIDEBAR TOGGLE
  // =========================================================================
  const sidebarToggle = $("#sidebar-toggle");
  const mobileMenuBtn = $("#mobile-menu-btn");

  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
    });
  }
  if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
    });
  }

  // Close sidebar on outside click (mobile)
  document.addEventListener("click", (e) => {
    if (
      window.innerWidth <= 768 &&
      sidebar.classList.contains("mobile-open") &&
      !sidebar.contains(e.target) &&
      e.target !== mobileMenuBtn
    ) {
      sidebar.classList.remove("mobile-open");
    }
  });

  // =========================================================================
  //  NAV ITEM CLICKS (section switching)
  // =========================================================================
  $$(".nav-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      $$(".nav-item").forEach((n) => n.classList.remove("active"));
      item.classList.add("active");
    });
  });

  // =========================================================================
  //  SYSTEM HEALTH CHECK
  // =========================================================================
  async function checkHealth() {
    const dot = $("#system-health-dot");
    try {
      const r = await fetch("/api/health");
      const d = await r.json();
      if (dot) {
        dot.style.background = d.status === "ok" ? "var(--c-known)" : "var(--c-unknown)";
        dot.style.boxShadow = d.status === "ok"
          ? "0 0 8px var(--c-known)"
          : "0 0 8px var(--c-unknown)";
      }
    } catch (_) {
      if (dot) {
        dot.style.background = "var(--c-unknown)";
        dot.style.boxShadow = "0 0 8px var(--c-unknown)";
      }
    }
  }

  // =========================================================================
  //  INITIAL LOAD
  // =========================================================================
  fetchEvents();
  fetchStats();
  setupAuto();
  refreshSource();
  checkHealth();

  setInterval(refreshSource, 3000);
  setInterval(checkHealth, 15000);
})();
