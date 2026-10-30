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
let busy = false;
let variantProblems = {};
let viewSuppressed = false;
let selectedPreviewUrl = null;

function selectedVariants() {
  return [...document.querySelectorAll('input[name="variant"]:checked')].map((input) => input.value);
}

function selectionProblems() {
  return selectedVariants().flatMap((variant) => variantProblems[variant] || []);
}

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
    variantProblems = status.variant_problems || {};
    busy = status.busy;
    ready = selectedVariants().length > 0 && selectionProblems().length === 0;
    const pill = $("host-status");
    pill.textContent = ready ? (busy ? "Host ready · run in progress" : "Host ready") : "Selected models not ready";
    pill.className = "pill " + (ready ? "ok" : "bad");
    pill.title = selectionProblems().join("\n");
    $("adapter-note").textContent = status.adapter_note;
    if (!ready) setMessage(selectedVariants().length ? "Not ready: " + selectionProblems().join("; ") : "Select at least one model", true);
    else if ($("message").textContent.startsWith("Not ready:") || $("message").textContent === "Select at least one model") setMessage("");
    updateRunButton(busy);
  } catch (error) {
    setMessage("Cannot reach demo server: " + error.message, true);
  }
}

function updateRunButton(busy) {
  $("run").disabled = !(ready && selectedFile && selectedVariants().length && !busy && !pollTimer);
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
  const bf16 = variants.find((v) => v.variant === "BF16" && v.status === "done");
  const base = bf16 && bf16.timings && bf16.timings.measured_parse_s;
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
  const statusClass = ["done", "failed", "skipped"].includes(v.status) ? v.status : "running";
  const node = el("article", "card " + statusClass);
  const head = el("div", "card-head");
  const identity = el("div");
  identity.appendChild(el("h3", null, LABELS[v.variant] || v.variant));
  identity.appendChild(el("small", null, "served as " + v.served_name));
  head.appendChild(identity);
  head.appendChild(el("span", "status " + statusClass, STATUS_TEXT[v.status] || v.status));
  node.appendChild(head);

  if (v.status === "skipped") {
    node.appendChild(el("p", "muted small", "Not selected for this run."));
    return node;
  }
  const t = v.timings || {};
  const primary = el("div", "primary-metric");
  primary.appendChild(el("strong", null, seconds(t.measured_parse_s)));
  primary.appendChild(el("span", null, "warm page parse · c=1"));
  node.appendChild(primary);
  if (v.status === "pending" || v.status === "starting_server" || v.status === "parsing") {
    node.appendChild(el("p", "muted small", v.status === "starting_server" ? "Checking checkpoint and loading on GPU…" : v.status === "parsing" ? "Parsing the uploaded page…" : "Waiting for the previous model…"));
    return node;
  }
  const rows = [
    ["First page parse", seconds(t.first_parse_s)],
    ["Server ready", seconds(t.server_ready_s)],
    ["Pipeline setup", seconds(t.pipeline_init_s)],
    ["Parsed regions", v.json_regions ?? "—"],
    ["Text similarity to BF16", typeof v.similarity_to_bf16 === "number" ? (100 * v.similarity_to_bf16).toFixed(1) + "%" : "—"],
    ["Peak device memory", v.gpu_memory ? `${v.gpu_memory.peak_used_mib} / ${v.gpu_memory.total_mib} MiB` : "—"],
  ];
  const dl = el("dl", "metrics");
  for (const [key, value] of rows) {
    dl.appendChild(el("dt", null, key));
    dl.appendChild(el("dd", null, value));
  }
  node.appendChild(dl);

  if (v.error) node.appendChild(el("pre", "error", v.error + (v.error_log_tail ? "\n\n--- log tail ---\n" + v.error_log_tail : "")));
  if (typeof v.markdown === "string") {
    node.appendChild(el("p", "excerpt", v.markdown.slice(0, 170).trim() + (v.markdown.length > 170 ? "…" : "")));
    const details = el("details", "output-view");
    details.appendChild(el("summary", null, "Full Markdown"));
    details.appendChild(el("pre", null, v.markdown));
    node.appendChild(details);
  }
  if (v.structured_json !== null && v.structured_json !== undefined) {
    const details = el("details", "output-view");
    details.appendChild(el("summary", null, "Full structured JSON"));
    details.appendChild(el("pre", null, JSON.stringify(v.structured_json, null, 2)));
    node.appendChild(details);
  }
  if (v.status === "done" && Object.hasOwn(LABELS, v.variant) && v.outputs) {
    const links = el("div", "downloads");
    for (const [kind, label] of [["markdown", "Download Markdown"], ["json", "Download JSON"]]) {
      if (!v.outputs[kind]) continue;
      const link = el("a", null, label);
      link.href = `/api/runs/${currentRunId}/outputs/${v.variant}/${kind}`;
      link.download = `${v.variant}-page.${kind === "markdown" ? "md" : "json"}`;
      links.appendChild(link);
    }
    node.appendChild(links);
  }
  return node;
}

let currentRunId = null;
function render(result) {
  currentRunId = result.run_id;
  clearSelectedPreview();
  $("empty").hidden = true;
  $("results-heading").hidden = false;
  $("summary").hidden = false;
  const s = result.settings || {};
  $("settings").textContent = `Run ${result.run_id.slice(0, 8)} · ${result.status} · layout on ${s.layout_device} · ${s.gpu_memory_utilization} GPU memory budget · warmup ${s.warmup ? "on" : "off"}. Latency is for this page and machine, not the study benchmark; text similarity is not accuracy.`;
  renderBars(result.variants || []);
  const columns = $("columns");
  columns.replaceChildren(...(result.variants || []).map(card));
}

function showPreview(runId, kind) {
  const img = $("preview");
  if (kind !== "png" && kind !== "jpeg") {
    img.hidden = true;
    img.removeAttribute("src");
    $("preview-note").textContent = kind === "pdf" ? "PDF input · preview unavailable" : "Preview unavailable";
  } else {
    img.hidden = false;
    img.src = `/api/runs/${runId}/input`;
    $("preview-note").textContent = "";
  }
}

function clearSelectedPreview() {
  $("empty-preview").hidden = true;
  $("empty-preview").removeAttribute("src");
  if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
  selectedPreviewUrl = null;
}

function clearPreviousView() {
  clearSelectedPreview();
  viewSuppressed = true;
  currentRunId = null;
  $("summary").hidden = true;
  $("results-heading").hidden = true;
  $("empty").hidden = false;
  $("columns").replaceChildren();
  $("bars").replaceChildren();
  $("settings").textContent = "";
  $("preview").hidden = true;
  $("preview").removeAttribute("src");
  $("preview-note").textContent = "";
  $("history").value = "";
}

async function follow(runId) {
  clearTimeout(pollTimer);
  const tick = async () => {
    try {
      const result = await getJSON(`/api/runs/${runId}`);
      if (!viewSuppressed) {
        render(result);
        if (result.input) showPreview(runId, result.input.kind);
      }
      if (["running", "queued"].includes(result.status)) {
        pollTimer = setTimeout(tick, 1000);
        return;
      }
      pollTimer = null;
      setMessage(viewSuppressed ? "Previous run finished. Choose a page to run again." : `Run finished: ${result.status}`, !viewSuppressed && result.status !== "done");
      loadHistory();
      loadStatus();
    } catch (error) {
      pollTimer = null;
      setMessage(error.message, true);
    }
  };
  pollTimer = setTimeout(tick, 0);
}

$("empty-preview").addEventListener("error", (event) => {
  if (!selectedPreviewUrl || event.target.getAttribute("src") !== selectedPreviewUrl) return;
  clearSelectedPreview();
  $("empty-note").textContent = "Preview unavailable. The server will verify this file when you run it.";
});

$("file").addEventListener("change", (event) => {
  clearPreviousView();
  const file = event.target.files[0] || null;
  if (file && !/\.(png|jpe?g|pdf)$/i.test(file.name)) {
    selectedFile = null;
    event.target.value = "";
    $("file-label").textContent = "Choose a different page";
    $("empty-title").textContent = "This file cannot be parsed";
    $("empty-note").textContent = "Choose a PNG, JPEG, or single-page PDF. The previous input and results have been cleared.";
    setMessage("Unsupported file. Upload a PNG, JPEG, or single-page PDF instead.", true);
  } else {
    selectedFile = file;
    $("file-label").textContent = file ? `${file.name} (${(file.size / 1048576).toFixed(2)} MB)` : "Choose a page";
    $("empty-title").textContent = file ? "Page ready to compare" : "Your next page starts here";
    if (!file) {
      $("empty-note").textContent = "Choose a page, select the models, then compare their real outputs side by side.";
    } else if (file.name.toLowerCase().endsWith(".pdf")) {
      $("empty-note").textContent = "PDF selected. Inline preview is unavailable; the server will validate its single page before running.";
    } else {
      $("empty-note").textContent = "Run the selected models to see outputs for this new page.";
    }
    if (file && /\.(png|jpe?g)$/i.test(file.name) && /^image\/(png|jpeg)$/.test(file.type)) {
      selectedPreviewUrl = URL.createObjectURL(file);
      $("empty-preview").src = selectedPreviewUrl;
      $("empty-preview").hidden = false;
    }
    setMessage(file ? "Page selected. Run the chosen models to see new results." : "");
  }
  updateRunButton(busy);
});

$("models").addEventListener("change", () => { loadStatus(); });

$("run").addEventListener("click", async () => {
  const selected = selectedVariants();
  if (!selectedFile || !selected.length || !ready || busy) return;
  $("run").disabled = true;
  setMessage("Uploading…");
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "X-Capstone-Demo": "1", "X-Capstone-Variants": selected.join(","), "Content-Type": "application/octet-stream" },
      body: selectedFile,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    busy = true;
    viewSuppressed = false;
    setMessage(`Running ${selected.join(" → ")} sequentially. Each model server is started, verified, used, and stopped in turn.`);
    follow(data.run_id);
  } catch (error) {
    selectedFile = null;
    $("file").value = "";
    $("file-label").textContent = "Choose a different page";
    $("empty-title").textContent = "This page could not be used";
    $("empty-note").textContent = "Choose a valid PNG, JPEG, or single-page PDF. No model output was created for this upload.";
    setMessage(error.message, true);
    loadStatus();
  }
});

$("history").addEventListener("change", (event) => {
  if (event.target.value) {
    selectedFile = null;
    $("file").value = "";
    $("file-label").textContent = "Choose a page";
    viewSuppressed = false;
    follow(event.target.value);
    updateRunButton(busy);
  }
});

loadStatus();
loadHistory().catch(() => {});
