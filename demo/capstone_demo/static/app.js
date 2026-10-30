"use strict";
// All server data is rendered with textContent / DOM nodes only.
const $ = (id) => document.getElementById(id);
const LABELS = { BF16: "BF16", W8A16: "GPTQ W8A16", W4A16: "GPTQ W4A16" };
const STATUS_TEXT = {
  pending: "waiting", starting_server: "verifying + loading", parsing: "parsing page",
  done: "done", failed: "failed", skipped: "skipped",
};
let selectedFile = null;
let pollTimer = null;
let ready = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function seconds(value) {
  return typeof value === "number" && isFinite(value) ? value.toFixed(2) + " s" : "—";
}

function setMessage(text, isError) {
  $("message").textContent = text || "";
  $("message").className = "message" + (isError ? " error" : "");
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function loadStatus() {
  try {
    const status = await getJSON("/api/status");
    ready = status.problems.length === 0;
    const pill = $("host-status");
    pill.textContent = ready ? (status.busy ? "Host ready · run in progress" : "Host ready") : "Host not ready";
    pill.className = "pill " + (ready ? "ok" : "bad");
    pill.title = status.problems.join("\n");
    $("adapter-note").textContent = status.adapter_note;
    if (!ready) setMessage("Not ready: " + status.problems.join("; "), true);
    updateRunButton(status.busy);
  } catch (error) {
    setMessage("Cannot reach demo server: " + error.message, true);
  }
}

function updateRunButton(busy) {
  $("run").disabled = !(ready && selectedFile && !busy && !pollTimer);
}

async function loadHistory() {
  const { runs } = await getJSON("/api/runs");
  const select = $("history");
  while (select.options.length > 1) select.remove(1);
  for (const run of runs) {
    const option = el("option", null, `${run.created_utc} · ${run.status} · ${run.run_id.slice(0, 8)}`);
    option.value = run.run_id;
    select.appendChild(option);
  }
}

function renderBars(variants) {
  const bars = $("bars");
  bars.replaceChildren();
  const values = variants.map((v) => v.timings && v.timings.measured_parse_s).filter((x) => typeof x === "number");
  const max = values.length ? Math.max(...values) : 1;
  const base = variants[0] && variants[0].timings && variants[0].timings.measured_parse_s;
  for (const v of variants) {
    const t = v.timings && v.timings.measured_parse_s;
    const row = el("div", "bar-row");
    row.appendChild(el("span", null, LABELS[v.variant] || v.variant));
    const track = el("div", "bar-track");
    const bar = el("div", "bar " + v.variant);
    bar.style.width = typeof t === "number" ? Math.max(2, (100 * t) / max) + "%" : "0%";
    track.appendChild(bar);
    row.appendChild(track);
    let label = typeof t === "number" ? seconds(t) : STATUS_TEXT[v.status] || v.status;
    if (typeof t === "number" && typeof base === "number" && v.variant !== "BF16") {
      label += ` · ${(base / t).toFixed(2)}×`;
    }
    row.appendChild(el("span", null, label));
    bars.appendChild(row);
  }
}

function card(v) {
  const node = el("article", "card");
  const head = el("div", "card-head");
  head.appendChild(el("h3", null, LABELS[v.variant] || v.variant));
  const statusClass = v.status === "done" ? "done" : v.status === "failed" ? "failed" : "running";
  head.appendChild(el("span", "status " + statusClass, STATUS_TEXT[v.status] || v.status));
  node.appendChild(head);
  node.appendChild(el("div", "muted small", "served as " + v.served_name));

  const t = v.timings || {};
  const rows = [
    ["Page parse (measured)", seconds(t.measured_parse_s)],
    ["Page parse (first pass)", seconds(t.first_parse_s)],
    ["Server verify + ready", seconds(t.server_ready_s)],
    ["Client pipeline init", seconds(t.pipeline_init_s)],
    ["Parsed regions", v.json_regions ?? "—"],
    ["Text similarity to BF16", typeof v.similarity_to_bf16 === "number" ? (100 * v.similarity_to_bf16).toFixed(1) + "%" : "—"],
    ["Peak GPU memory (device)", v.gpu_memory ? `${v.gpu_memory.peak_used_mib} / ${v.gpu_memory.total_mib} MiB` : "—"],
  ];
  const dl = el("dl");
  for (const [k, value] of rows) {
    dl.appendChild(el("dt", null, k));
    dl.appendChild(el("dd", null, value));
  }
  node.appendChild(dl);

  if (v.error) {
    node.appendChild(el("pre", "error", v.error + (v.error_log_tail ? "\n\n--- log tail ---\n" + v.error_log_tail : "")));
  }
  if (typeof v.markdown === "string") {
    node.appendChild(el("div", "muted small", "Parsed Markdown (raw text, not rendered)" + (v.markdown_truncated ? " · truncated" : "")));
    node.appendChild(el("pre", null, v.markdown));
  }
  return node;
}

function render(result) {
  $("summary").hidden = false;
  const s = result.settings || {};
  $("settings").textContent = result.status === "queued" ? "Queued…" :
    `Run ${result.run_id.slice(0, 8)} · ${result.status} · layout on ${s.layout_device} · ` +
    `gpu_mem_util ${s.gpu_memory_utilization} · max_num_seqs ${s.max_num_seqs} · warmup ${s.warmup ? "on" : "off"}. ` +
    "Speedup = BF16 measured / variant measured on this page and this machine only; similarity is a text diff, not accuracy.";
  renderBars(result.variants || []);
  const columns = $("columns");
  columns.replaceChildren(...(result.variants || []).map(card));
}

function showPreview(runId, kind) {
  const img = $("preview");
  if (kind === "pdf") {
    img.hidden = true;
    $("preview-note").textContent = "PDF input (no inline preview)";
  } else {
    img.hidden = false;
    img.src = `/api/runs/${runId}/input`;
    $("preview-note").textContent = "";
  }
}

async function follow(runId) {
  clearTimeout(pollTimer);
  const tick = async () => {
    try {
      const result = await getJSON(`/api/runs/${runId}`);
      render(result);
      if (result.input) showPreview(runId, result.input.kind);
      if (["running", "queued"].includes(result.status)) {
        pollTimer = setTimeout(tick, 1000);
        return;
      }
      pollTimer = null;
      setMessage(`Run finished: ${result.status}`, result.status !== "done");
      loadHistory();
      loadStatus();
    } catch (error) {
      pollTimer = null;
      setMessage(error.message, true);
    }
  };
  pollTimer = setTimeout(tick, 0);
}

$("file").addEventListener("change", (event) => {
  selectedFile = event.target.files[0] || null;
  $("file-label").textContent = selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1048576).toFixed(2)} MB)` : "Choose a page";
  updateRunButton(false);
});

$("run").addEventListener("click", async () => {
  if (!selectedFile) return;
  $("run").disabled = true;
  setMessage("Uploading…");
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "X-Capstone-Demo": "1", "Content-Type": "application/octet-stream" },
      body: selectedFile,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    setMessage("Running BF16 → W8A16 → W4A16 sequentially. Each model server is started, verified, used, and stopped in turn.");
    follow(data.run_id);
  } catch (error) {
    setMessage(error.message, true);
    loadStatus();
  }
});

$("history").addEventListener("change", (event) => {
  if (event.target.value) follow(event.target.value);
});

loadStatus();
loadHistory().catch(() => {});
