"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = { get(k, d) { try { const v = localStorage.getItem("mp." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }, set(k, v) { try { localStorage.setItem("mp." + k, JSON.stringify(v)); } catch (e) { } } };
const api = async (path, body) => { const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); const j = await r.json(); if (!r.ok) throw j; return j; };
const STATES = { done: "done", failed: "failed", cancelled: "cancelled", running: "running", stopped: "stopped (partial)", waiting: "waiting", idle: "idle" };
const badge = s => `<span class="badge ${esc(s)}">${esc(STATES[s] || s)}</span>`;
let saved = [], schema = null, examples = [], timer = null, perfTimer = null;

function setTab(t) {
  document.querySelectorAll(".seg button[data-tab]").forEach(b => { const on = b.dataset.tab === t; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
  ["samples", "run", "results"].forEach(n => { $("#tab-" + n).hidden = n !== t; });
  store.set("tab", t);
  if (t === "run") refreshRun(); if (t === "results") refreshResults();
}
function setH(el, html) {
  if (el.__h === html) return;
  const key = d => (d.querySelector("summary b") || d.querySelector("summary") || d).textContent.trim(), open = new Set([...el.querySelectorAll("details[open]")].map(key)), shut = new Set([...el.querySelectorAll("details:not([open])")].map(key));
  el.innerHTML = html; el.__h = html;
  el.querySelectorAll("details").forEach(d => { const k = key(d); if (open.has(k)) d.open = true; else if (shut.has(k)) d.open = false; });
}

function field(f, v, ro) {
  return `<div class="field"><label for="${f.id}_X">${esc(f.label)}${f.required ? " *" : ""}</label><input type="text" data-id="${f.id}" value="${esc(v || "")}" placeholder="${esc(f.help || "")}" ${ro ? "readonly" : ""} aria-describedby="${f.id}_X_h"><p class="hint" id="${f.id}_X_h"></p></div>`;
}
function card(s, ro) {
  const src = s.uniprot_query ? "uniprot" : "fasta";
  const groups = schema.groups.map(g => {
    if (!g.source) return `<div class="fields">${g.fields.map(f => field(f, s[f.id], ro)).join("")}</div>`;
    const [fa, uq] = g.fields;
    return `<fieldset class="group"><legend class="title small">${esc(g.name)}</legend>
      <div class="seg small" role="radiogroup" aria-label="Protein source">${[["fasta", "Local FASTA"], ["uniprot", "UniProtKB query"]].map(([k, l]) => `<button type="button" role="radio" data-src="${k}" aria-checked="${src === k}" class="${src === k ? "on" : ""}" ${ro ? "disabled" : ""}>${l}</button>`).join("")}</div>
      <div class="fields src-fasta" ${src === "fasta" ? "" : "hidden"}>${field(fa, s.fasta_path, ro)}</div><div class="fields src-uniprot" ${src === "uniprot" ? "" : "hidden"}>${field(uq, s.uniprot_query, ro)}</div></fieldset>`;
  }).join("");
  const el = document.createElement("div"); el.className = "card sample";
  el.innerHTML = `<div class="card-head"><span class="title">${esc(s.sample_id || "new sample")}</span>${ro ? "" : `<span><button type="button" class="pill plain dup" title="Duplicate this sample">Duplicate</button><button type="button" class="pill red del" title="Remove this sample">Remove</button></span>`}</div>${groups}`;
  el.querySelectorAll("input").forEach((inp, i) => { const id = inp.id = `${inp.dataset.id}_${Math.random().toString(36).slice(2, 8)}`; inp.previousElementSibling.htmlFor = id; inp.setAttribute("aria-describedby", id + "_h"); inp.nextElementSibling.id = id + "_h"; });
  el.querySelectorAll("[data-src]").forEach(b => b.onclick = () => { el.querySelectorAll("[data-src]").forEach(x => { const on = x === b; x.classList.toggle("on", on); x.setAttribute("aria-checked", on); }); el.querySelector(".src-fasta").hidden = b.dataset.src !== "fasta"; el.querySelector(".src-uniprot").hidden = b.dataset.src !== "uniprot"; });
  const sid = el.querySelector('[data-id="sample_id"]'); if (sid) sid.oninput = () => { el.querySelector(".title").textContent = sid.value || "new sample"; };
  const fp = el.querySelector('[data-id="fasta_path"]'); let t = null;
  if (fp && !ro) fp.oninput = () => { clearTimeout(t); t = setTimeout(() => checkPath(fp), 350); };
  if (!ro) { el.querySelector(".dup").onclick = () => { const c = collect([el])[0]; c.sample_id = (c.sample_id || "s") + "_copy"; el.after(card(c)); }; el.querySelector(".del").onclick = () => el.remove(); }
  return el;
}
async function checkPath(inp) {
  const h = inp.nextElementSibling; if (!inp.value) { h.textContent = ""; return; }
  const r = await api("/api/path?p=" + encodeURIComponent(inp.value));
  h.className = "hint " + (r.file ? "ok" : "err"); h.textContent = r.file ? `✓ file, ${(r.size / 1e6).toFixed(2)} MB` : r.exists ? "✕ a folder: pick a FASTA file" : `✕ not found${r.entries.length ? " — e.g. " + r.entries.slice(0, 3).join(", ") : ""}`;
}
function collect(cards) {
  return (cards || [...document.querySelectorAll("#samples .sample")]).map(el => {
    const s = {}; el.querySelectorAll("input[data-id]").forEach(i => { s[i.dataset.id] = i.value.trim(); });
    const src = el.querySelector("[data-src].on")?.dataset.src; if (src === "fasta") s.uniprot_query = ""; else s.fasta_path = ""; return s;
  });
}
function showSamples(list, ro) { const box = $("#samples"); box.innerHTML = ""; list.forEach(s => box.append(card(s, ro))); ["#add", "#save"].forEach(b => { $(b).disabled = !!ro; }); }
function clearSamples() { showSamples([]); $("#example").value = ""; const st = $("#samples-status"); st.hidden = false; st.textContent = "Cards cleared. Nothing is saved until you click Save samples."; }
const OPTS = ["dedup", "twonn", "pca", "tsne", "umap", "all_layers"];
function pickedOptions() { return Object.fromEntries(OPTS.map(k => [k, $("#opt-" + k).checked])); }
function showOptions(o, ro) { OPTS.forEach(k => { $("#opt-" + k).checked = !!o[k]; $("#opt-" + k).disabled = !!ro; }); }
function pickedBackends() { return [...$("#backends").querySelectorAll("input:checked")].map(x => x.value); }
function showBackends(names, ro) { $("#backends").querySelectorAll("input").forEach(i => { i.checked = names.includes(i.value); i.disabled = !!ro; }); }
async function loadSaved() { const r = await api("/api/samples"); showSamples(r.samples.length ? r.samples : [{}]); $("#example").value = ""; if ($("#backends").children.length) { showBackends(saved, false); showOptions(await api("/api/options"), false); } }
async function saveSamples() {
  const st = $("#samples-status"); st.hidden = false;
  try { const names = pickedBackends(); if (!names.length) throw { errors: ["switch on at least one model"] }; const r = await api("/api/samples", { samples: collect() }); await api("/api/backends", { backends: names }); await api("/api/options", pickedOptions()); saved = names; st.textContent = `✓ saved ${r.n} sample(s) to ${r.samples_tsv} and models ${names.join(", ")}`; }
  catch (e) { st.textContent = "✕ " + (e.errors || [JSON.stringify(e)]).join("\n✕ "); }
}

async function loadBackends() {
  const r = await api("/api/backends");
  $("#backend-why").textContent = `Default ask per GPU job: ${r.why}. GPU lane: ${r.gpus_max} GPU at a time.`;
  saved = r.selected;
  $("#backends").innerHTML = r.backends.map(b => `<div class="backend"><label class="switch"><input type="checkbox" value="${esc(b.name)}" ${r.selected.includes(b.name) ? "checked" : ""}><span></span><b>${esc(b.name)}</b> ${b.weights === "ready" ? '<span class="badge done">weights</span>' : '<span class="badge waiting" title="weights to download: the pinned revision is not on disk yet">no weights</span>'}</label>
    <span class="meta">${esc(b.family)} · ${esc(b.size_gb)} GB · ${esc(b.partition)} · ${b.cpus} cores ${(b.mem_mb / 1000).toFixed(0)} GB · rev ${esc(b.revision.slice(0, 10))}</span><span class="meta" title="licence">${esc(b.licence)}</span></div>`).join("");
}
function bar(parts) { const tot = parts.reduce((a, p) => a + p.n, 0) || 1; return `<div class="bar" role="img" aria-label="${esc(parts.map(p => `${p.n} ${p.label}`).join(", "))}">${parts.map(p => p.n ? `<i class="${p.cls}" style="width:${100 * p.n / tot}%" title="${p.n} ${esc(p.label)}"></i>` : "").join("")}</div>`; }
async function refreshRun() {
  const name = $("#analysis").value.trim() || store.get("analysis", "");
  const r = await api("/api/status?analysis=" + encodeURIComponent(name));
  $("#run-config").textContent = `Models from the saved configuration: ${saved.join(", ") || "none"}. Change them in the Samples tab.`;
  $("#runtime").textContent = `image ${r.runtime.sif || "not pulled (containers/build.sh pull)"} · sbatch ${r.runtime.sbatch ? "✓" : "✕ missing"} · singularity ${r.runtime.apptainer ? "✓" : "✕ missing"}`;
  const pick = $("#pick"), cur = pick.value; pick.innerHTML = '<option value="">—</option>' + r.analyses.map(a => `<option ${a.name === name ? "selected" : ""}>${esc(a.name)}</option>`).join(""); if (cur && !name) pick.value = cur;
  $("#res-pick").innerHTML = '<option value="">—</option>' + r.analyses.map(a => `<option ${a.name === store.get("res", "") ? "selected" : ""}>${esc(a.name)}</option>`).join("");
  const rs = $("#run-state"); rs.className = "badge " + r.outcome; rs.textContent = STATES[r.outcome] || r.outcome;
  $("#size").textContent = r.size ? `${r.size.level === "ok" ? "✓" : r.size.level === "warn" ? "▲ large:" : "✕ above the cap:"} estimated embeddings ${r.size.total_gb} GB (warn above ${r.size.warn_gb} GB, refuse above ${r.size.cap_gb} GB unless allowed)${r.size.hard && r.size.hard.length ? " ✕ " + r.size.hard.join("; ") + " (hard cap)" : ""}` : "";
  setH($("#progress"), r.progress ? `<p class="muted small">${r.progress.done}/${r.progress.total} jobs done (${r.progress.pct}%)</p>${bar([{ n: r.progress.done, cls: "ok", label: "done" }, { n: r.progress.total - r.progress.done, cls: "", label: "to do" }])}` : "");
  setH($("#steps"), r.steps.map(s => {
    const open = s.failed || s.cancelled ? "open" : "", state = s.failed ? "failed" : s.done === s.total ? "done" : s.running ? "running" : "waiting";
    return `<details class="card step" ${open}><summary>${badge(state)} <b>${esc(s.step)}</b> <span class="count">${s.done}/${s.total}${s.failed ? ` · ${s.failed} failed` : ""}${s.running ? ` · ${s.running} running` : ""}</span><span class="grow"></span><span style="width:180px">${bar([{ n: s.done, cls: "ok", label: "done" }, { n: s.running, cls: "run", label: "running" }, { n: s.failed + s.cancelled, cls: "bad", label: "failed or cancelled" }, { n: s.total - s.done - s.running - s.failed - s.cancelled, cls: "", label: "waiting" }])}</span></summary>
      ${s.jobs.map(j => `<div class="job">${badge(j.state)} ${j.gpu ? '<span class="badge gpu">GPU</span>' : ""} ${esc(j.job)} <span class="muted small">${esc(j.log)}</span>${j.tail ? `<pre class="log">${esc(j.tail)}</pre>` : ""}</div>`).join("")}</details>`;
  }).join(""));
  setH($("#queue"), r.queue.length ? `<table><tr><th>job</th><th>name</th><th>state</th><th>time</th><th>partition</th><th>where / reason</th></tr>${r.queue.map(q => `<tr><td>${esc(q.job)}</td><td>${esc(q.name)}</td><td>${esc(q.state)}</td><td>${esc(q.time)}</td><td>${esc(q.partition)}</td><td>${esc(q.where)}</td></tr>`).join("")}</table>` : '<p class="muted">no queued job</p>');
  $("#node-log").textContent = r.node_log || "—"; $("#run-log").textContent = r.log || "—";
}
async function run() {
  const st = $("#run-status"); st.hidden = false; st.textContent = "… downloading UniProtKB samples (if any), then submitting (this can take a minute)";
  const name = $("#analysis").value.trim(); store.set("analysis", name);
  const ex = $("#use-example").checked && $("#example").value && $("#example").value !== "__all" ? [$("#example").value] : $("#use-example").checked ? examples.map(e => e.name) : [];
  try { const r = await api("/api/run", { analysis: name, examples: ex, online_here: $("#online-here").checked, allow_large: $("#allow-large").checked }); st.textContent = `✓ started ${r.members ? r.members.join(", ") : r.analysis}`; if (r.members) { $("#analysis").value = r.members[0]; store.set("analysis", r.members[0]); } }
  catch (e) { st.textContent = "✕ " + (e.errors || [JSON.stringify(e)]).join("\n✕ "); }
  refreshRun();
}
async function stop(all) { const r = await api(all ? "/api/stop_all" : "/api/stop", { analysis: $("#analysis").value.trim() }); $("#run-status").hidden = false; $("#run-status").textContent = "■ " + JSON.stringify(r); refreshRun(); }

async function refreshResults() {
  const name = $("#res-pick").value || store.get("res", ""); if (!name) return;
  store.set("res", name);
  const r = await api("/api/results?analysis=" + encodeURIComponent(name));
  const cell = {}; r.embeddings.forEach(e => { cell[e.sample_id + "|" + e.backend] = e; });
  $("#matrix").innerHTML = `<div class="scroll"><table><tr><th>sample</th>${r.backends.map(b => `<th>${esc(b)}</th>`).join("")}</tr>${r.sample_ids.map(s => `<tr><td>${esc(s)}</td>${r.backends.map(b => { const e = cell[s + "|" + b]; return e ? `<td title="${esc(e.keys)}; stored ${esc(e.store_dtype)}, computed ${esc(e.compute_dtype)}; ${e.truncated} truncated; ${e.files} file(s)">✓ ${e.proteins} × ${e.dim}<br><span class="muted small">${esc(e.keys)}</span></td>` : "<td>○ not yet</td>"; }).join("")}</tr>`).join("")}</table></div><p class="muted small">Settings: layers ${esc(JSON.stringify(r.settings.layers))}, pooling ${esc(JSON.stringify(r.settings.pooling))}, ${esc(r.settings.store_dtype)}</p>`;
  charts(r);
  $("#records-legend").innerHTML = '<span><i class="sw ok"></i>kept (solid)</span><span><i class="sw bad"></i>invalid (striped)</span>';
  $("#records").innerHTML = r.samples.length ? r.samples.map(s => `<div class="row"><span style="min-width:160px">${esc(s.sample_id)}</span><span class="grow">${bar([{ n: +s.kept, cls: "ok", label: "kept" }, { n: +s.invalid, cls: "bad", label: "invalid" }])}</span><span class="muted small">${s.kept}/${s.records} kept${s.uniprot_release ? " · UniProt " + esc(s.uniprot_release) : ""}</span></div>`).join("") : '<p class="muted">no summary yet (03_summary)</p>';
  $("#weights").innerHTML = `<table><tr><th>backend</th><th>repo</th><th>revision</th><th>files</th><th>downloaded</th></tr>${Object.entries(r.weights).map(([b, w]) => `<tr><td>${esc(b)}</td><td>${esc(w ? w.repo : "")}</td><td>${esc(w ? w.revision.slice(0, 12) : "○ missing")}</td><td>${w ? Object.keys(w.files).length : ""}</td><td>${esc(w ? w.downloaded_at : "")}</td></tr>`).join("")}</table>`;
  $("#meta").textContent = JSON.stringify(r.meta, null, 1);
}

const SERIES = ["#1f5f4f", "#e0a526", "#0071e3", "#c46aa6", "#9c3a00", "#8ec9f0", "#4b2a7b"];
const SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star", "ring"];
function swatch(i) { const c = SERIES[i % SERIES.length]; return i < SERIES.length ? `background:${c}` : `background:repeating-linear-gradient(45deg,${c} 0 4px,#ffffff 4px 6px)`; }
function fill(i) { return SERIES[i % SERIES.length]; }
function legend(names) { return `<div class="legend">${names.map((n, i) => `<span><i class="sw" style="${swatch(i)}"></i>${esc(n)}</span>`).join("")}</div>`; }
function bars(title, cats, series, fmt = v => v) {
  const names = Object.keys(series), W = 640, H = 220, pad = 36, gw = (W - pad) / Math.max(cats.length, 1), bw = Math.max(2, (gw - 4) / Math.max(names.length, 1));
  const max = Math.max(1e-9, ...names.flatMap(n => series[n].filter(v => v !== null)));
  const rects = cats.map((c, ci) => names.map((n, si) => { const v = series[n][ci]; if (v === null || v === undefined) return ""; const h = (H - 40) * v / max, x = pad + ci * gw + 2 + si * bw, y = H - 24 - h;
    return `<rect x="${x}" y="${y}" width="${bw - 1}" height="${h}" fill="${fill(si)}"><title>${esc(n)} · ${esc(c)}\n${esc(fmt(v))}</title></rect>${si >= SERIES.length ? `<rect x="${x}" y="${y}" width="${bw - 1}" height="${h}" fill="url(#hatch)"/>` : ""}`; }).join("")).join("");
  const ticks = cats.map((c, ci) => ci % Math.ceil(cats.length / 10) ? "" : `<text x="${pad + ci * gw + gw / 2}" y="${H - 8}" font-size="12.5" text-anchor="middle" fill="currentColor">${esc(c)}</text>`).join("");
  return `<h4>${esc(title)}</h4>${legend(names)}<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}" style="width:100%;max-width:${W}px"><text x="2" y="14" font-size="12.5" fill="currentColor">${esc(fmt(max))}</text><line x1="${pad}" y1="${H - 24}" x2="${W}" y2="${H - 24}" stroke="currentColor"/>${rects}${ticks}</svg>`;
}
function heat(title, m) {
  const n = m.samples.length, c = 64, W = 160 + n * c;
  const cells = m.matrix.map((row, i) => row.map((v, j) => { const g = Math.round(255 - 180 * Math.max(0, v)); return `<rect x="${160 + j * c}" y="${24 + i * c}" width="${c - 2}" height="${c - 2}" fill="rgb(${g},${g},${g})"><title>${esc(m.samples[i])} · ${esc(m.samples[j])}\ncosine ${v}</title></rect><text x="${160 + j * c + c / 2}" y="${24 + i * c + c / 2 + 4}" font-size="12.5" text-anchor="middle" fill="${g < 140 ? "#ffffff" : "#1d1d1f"}">${v.toFixed(2)}</text>`; }).join("")).join("");
  const labels = m.samples.map((s, i) => `<text x="154" y="${24 + i * c + c / 2 + 4}" font-size="12.5" text-anchor="end" fill="currentColor">${esc(s)}</text><text x="${160 + i * c + c / 2}" y="16" font-size="12.5" text-anchor="middle" fill="currentColor"><title>${esc(s)}</title>${esc(s.length > 8 ? s.slice(0, 7) + "…" : s)}</text>`).join("");
  return `<h4>${esc(title)}</h4><p class="muted small">Each cell prints its cosine; darker = more similar.</p><svg viewBox="0 0 ${W} ${30 + n * c}" role="img" aria-label="${esc(title)}" style="width:100%;max-width:${W}px">${labels}${cells}</svg>`;
}
function shape(k, x, y, col, r = 3.5) { return [`<circle cx="${x}" cy="${y}" r="${r}" fill="${col}"/>`, `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" fill="${col}"/>`, `<path d="M${x} ${y - r}L${x + r} ${y + r}L${x - r} ${y + r}Z" fill="${col}"/>`, `<path d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z" fill="${col}"/>`, `<path d="M${x - r} ${y}H${x + r}M${x} ${y - r}V${y + r}" stroke="${col}" stroke-width="2"/>`, `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${col}" stroke-width="2"/>`, `<circle cx="${x}" cy="${y}" r="${r + 1}" fill="none" stroke="${col}" stroke-width="1"/>`][k % 7]; }
function lines(title, xs, series, opts = {}) {
  const names = Object.keys(series), W = 640, H = 240, L = 44, B = 34, logx = !!opts.logx;
  const fx = v => logx ? Math.log10(v) : v, xmin = Math.min(...xs.map(fx)), xmax = Math.max(...xs.map(fx)), ys = names.flatMap(n => series[n].filter(v => v !== null && v !== undefined));
  const ymax = Math.max(1e-9, ...ys, ...(opts.ymax ? [opts.ymax] : [])) * 1.08, ymin = Math.min(0, ...ys);
  const X = v => L + (W - L - 10) * (fx(v) - xmin) / ((xmax - xmin) || 1), Y = v => H - B - (H - B - 10) * (v - ymin) / ((ymax - ymin) || 1);
  const body = names.map((n, i) => { const pts = xs.map((x, j) => [x, series[n][j], (opts.err && opts.err[n] || [])[j]]).filter(p => p[1] !== null && p[1] !== undefined);
    return `<polyline fill="none" stroke="${fill(i)}" stroke-width="2" ${i >= SERIES.length ? 'stroke-dasharray="5 3"' : ""} points="${pts.map(p => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>` + pts.map(p => `<g><title>${esc(n)} · ${esc(opts.xlabel || "x")} ${p[0]}\n${esc(opts.ylabel || "y")} ${(+p[1]).toFixed(2)}${p[2] ? ` ± ${(+p[2]).toFixed(2)}` : ""}</title>${p[2] ? `<line x1="${X(p[0])}" x2="${X(p[0])}" y1="${Y(p[1] - p[2])}" y2="${Y(p[1] + p[2])}" stroke="${fill(i)}"/>` : ""}${shape(i, X(p[0]), Y(p[1]), fill(i))}</g>`).join(""); }).join("");
  let last = -1e9; const ticks = xs.map(x => { const px = X(x); if (px - last < 44) return ""; last = px; return `<text x="${px}" y="${H - 18}" font-size="12.5" text-anchor="middle" fill="currentColor">${x}</text>`; }).join("");
  const yt = [ymin, (ymin + ymax) / 2, ymax].map(v => `<text x="${L - 4}" y="${Y(v) + 4}" font-size="12.5" text-anchor="end" fill="currentColor">${v.toFixed(1)}</text>`).join("");
  return `<h4>${esc(title)}</h4><div class="legend">${names.map((n, i) => `<span><svg width="14" height="14" aria-hidden="true">${shape(i, 7, 7, fill(i))}</svg>${esc(n)}</span>`).join("")}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}" style="width:100%;max-width:${W}px"><line x1="${L}" y1="${H - B}" x2="${W}" y2="${H - B}" stroke="currentColor"/><line x1="${L}" y1="10" x2="${L}" y2="${H - B}" stroke="currentColor"/>${yt}${ticks}<text x="${W / 2}" y="${H - 2}" font-size="12.5" text-anchor="middle" fill="currentColor">${esc(opts.xlabel || "")}</text>${body}</svg>`;
}
function crossDepth(T) {
  const bs = [...new Set(T.filter(r => r.pooling === "mean" && r.sample === "all samples").map(r => r.backend))];
  if (bs.length < 2) return "";
  const depth = [0.25, 0.5, 0.75, 1]; return lines("All samples: d at relative depth (nearest layer), per backend", depth, Object.fromEntries(bs.map(x => { const rs = T.filter(r => r.backend === x && r.pooling === "mean" && r.sample === "all samples"), n = Math.max(...rs.map(r => r.layer)); return [x, depth.map(f => { const t = Math.max(1, Math.round(f * n)); const hit = rs.reduce((p, r) => Math.abs(r.layer - t) < Math.abs(p.layer - t) ? r : p, rs[0]); return hit ? hit.d : null; })]; })), { xlabel: "layer / number of layers", ylabel: "d" });
}
function twonnCharts(A, b, key, out, cross = true) {
  const T = (A.twonn || {}).rows || [], box = out || $("#twonn");
  if (!T.length) { box.innerHTML = '<p class="muted">not computed (summarize.twonn.enabled is off, or fewer than 20 points per sample)</p>'; return; }
  const mine = T.filter(r => r.backend === b && r.pooling === "mean"), groups = [...new Set(T.map(r => r.sample))];
  const layers = [...new Set(mine.map(r => r.layer))].sort((x, y) => x - y);
  let html = layers.length > 1 ? lines(`${b}: intrinsic dimension by layer (mean vectors)`, layers, Object.fromEntries(groups.map(g => [g, layers.map(l => (mine.find(r => r.layer === l && r.sample === g) || {}).d ?? null)])), { xlabel: "layer", ylabel: "d", err: Object.fromEntries(groups.map(g => [g, layers.map(l => (mine.find(r => r.layer === l && r.sample === g) || {}).error)])) }) : "";
  html += cross ? crossDepth(T) : "";
  const at = T.filter(r => r.backend === b && r.key === key);
  if (at.length) { const Ns = [...new Set(at.flatMap(r => r.decimation.map(l => l.points)))].sort((x, y) => x - y); html += lines(`${b} ${key}: d(N) by number of points (random subsets, mean ± standard error)`, Ns, Object.fromEntries(at.map(r => [r.sample, Ns.map(N => (r.decimation.find(l => l.points === N) || {}).d ?? null)])), { logx: true, xlabel: "N points (log scale)", ylabel: "d", err: Object.fromEntries(at.map(r => [r.sample, Ns.map(N => (r.decimation.find(l => l.points === N) || {}).spread)])) });
    html += `<div class="scroll"><table><tr><th>sample</th><th>points</th><th>d</th><th>± SE (N/2)</th><th>duplicates dropped</th><th>note</th></tr>${at.map(r => `<tr><td>${esc(r.sample)}</td><td>${r.points}</td><td>${r.d ?? "—"}</td><td>${r.error ?? "—"}</td><td>${r.dropped}${r.dropped_warning ? " ▲ above 10%: unreliable" : ""}</td><td>${r.stratified_tokens ? "residues as points, stratified by protein" : "one mean vector per protein"}</td></tr>`).join("")}</table></div>`; }
  box.innerHTML = html || '<p class="muted">no TWO-NN estimate for this backend</p>';
}
function scatter(title, series, W = 360, H = 300) {
  const names = Object.keys(series), all = names.flatMap(n => series[n]), xs = all.map(p => p[0]), ys = all.map(p => p[1]), q = (a, f) => [...a].sort((u, v) => u - v)[Math.min(a.length - 1, Math.max(0, Math.round(f * (a.length - 1))))], x0 = q(xs, .005), x1 = q(xs, .995), y0 = q(ys, .005), y1 = q(ys, .995), cl = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const X = v => 10 + (W - 20) * (v - x0) / ((x1 - x0) || 1), Y = v => H - 10 - (H - 20) * (v - y0) / ((y1 - y0) || 1);
  const order = names.map((n, i) => i).sort((a, b) => series[names[b]].length - series[names[a]].length), dots = order.map(i => `<g opacity=".7">${series[names[i]].map(p => `<g><title>${esc(names[i])}</title>${shape(i, Math.round(cl(X(p[0]), 4, W - 4) * 10) / 10, Math.round(cl(Y(p[1]), 4, H - 4) * 10) / 10, fill(i), 2.6)}</g>`).join("")}</g>`).join("");
  return `<figure class="panel"><figcaption>${esc(title)}</figcaption><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}" style="width:100%;max-width:${W}px;border:1px solid var(--hair)">${dots}</svg></figure>`;
}
const METHODS = [["pca", "PCA"], ["tsne", "t-SNE"], ["umap", "UMAP"]];
function projData(A, b, m) {
  const k = Object.keys((A.embed2d || {})[b] || {}).concat(Object.keys((A.pca || {})[b] || {})).filter(x => x.endsWith("/mean")).sort((x, y) => parseInt(x.split("_")[1]) - parseInt(y.split("_")[1])).pop();
  const E = ((A.embed2d || {})[b] || {})[k], P = ((A.pca || {})[b] || {})[k];
  if (m === "pca") return P && P.points ? P.points : null;
  if (!E || !E[m]) return null;
  const o = {}; E[m].forEach((p, i) => (o[E.labels[i]] = o[E.labels[i]] || []).push(p)); return o;
}
function pearson(a, b) { const n = a.length, ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb || 1); }
function agreementHtml(A) {
  const G = A.embed2d_agreement; if (!G) return "";
  const names = G.backends, pts = b => { const P = ((A.pca || {})[b] || {})[G.key]; return P && P.points; }, sel = store.get("pair", [names[0], names[1]]);
  const [x, y] = sel.every(n => names.includes(n)) && sel[0] !== sel[1] ? sel : [names[0], names[1]];
  let pair = "";
  if (pts(x) && pts(y)) {
    const pc = c => { const o = {}; let ax = [], ay = []; Object.keys(pts(x)).forEach(sm => { o[sm] = pts(x)[sm].map((p, i) => [p[c], pts(y)[sm][i][c]]); pts(x)[sm].forEach((p, i) => { ax.push(p[c]); ay.push(pts(y)[sm][i][c]); }); }); return [o, Math.abs(pearson(ax, ay))]; };
    const [o1, r1] = pc(0), [o2, r2] = pc(1);
    pair = `<div class="grid-row" style="--n:2">${scatter(`PC1: ${x} vs ${y} (|r| = ${r1.toFixed(2)})`, o1)}${scatter(`PC2: ${x} vs ${y} (|r| = ${r2.toFixed(2)})`, o2)}</div><p class="muted small">Each dot is one protein: its PCA coordinate in ${esc(x)} (horizontal) against ${esc(y)} (vertical); the sign of a component is arbitrary, so |r| is shown.</p>`;
  } else pair = '<p class="muted">Per-protein PCA scatter needs PCA on (and fewer than 1000 points).</p>';
  const opt = v => names.map(n => `<option ${n === v ? "selected" : ""}>${esc(n)}</option>`).join("");
  return `<h4>Agreement between models (${G.proteins} proteins, ${esc(G.key)})</h4><div class="grid-row" style="--n:2">${heat("Distance correlation (RSA): pairwise cosine distances of the same proteins", { samples: names, matrix: G.rsa })}${heat(`Neighbour overlap: share of the ${G.k} nearest neighbours in common`, { samples: names, matrix: G.knn })}</div>
    <div class="row"><div class="field"><label for="ag-a">Model A</label><select id="ag-a" class="inline">${opt(x)}</select></div><div class="field"><label for="ag-b">Model B</label><select id="ag-b" class="inline">${opt(y)}</select></div></div>${pair}`;
}
function projections(A) {
  const box = $("#embed2d"), bs = Object.keys(A.backends || {}).concat(Object.keys(A.embed2d || {}), Object.keys(A.pca || {})).filter((b, i, a) => a.indexOf(b) === i);
  const have = bs.filter(b => METHODS.some(([m]) => projData(A, b, m))), ms = METHODS.filter(([m]) => have.some(b => projData(A, b, m)));
  if (!have.length) { box.innerHTML = '<p class="muted">no projections (switch on PCA, t-SNE or UMAP in the Samples tab; t-SNE and UMAP need fewer than 1000 points)</p>'; return; }
  const lay = store.get("proj", "models"), cell = (b, m, t, W) => projData(A, b, m) ? scatter(t, projData(A, b, m), W) : `<figure class="panel"><figcaption>${esc(t)}</figcaption><p class="muted small">not available</p></figure>`;
  const names = [...new Set(have.flatMap(b => Object.keys(projData(A, b, ms[0][0]) || projData(A, b, ms.find(([m]) => projData(A, b, m))[0]))))];
  let body = "";
  if (lay === "models") body = have.map(b => `<div class="by-model"><h4>${esc(b)}</h4><div class="grid-row" style="--n:${ms.length}">${ms.map(([m, l]) => cell(b, m, l, 360)).join("")}</div></div>`).join("");
  else if (lay === "methods") body = ms.map(([m, l]) => `<div class="by-model"><h4>${l}</h4><div class="grid-row" style="--n:${Math.min(have.length, 4)}">${have.map(b => cell(b, m, b, 300)).join("")}</div></div>`).join("");
  else { const sb = store.get("pb", have[0]), sm = store.get("pm", ms[0][0]), bb = have.includes(sb) ? sb : have[0], mm = ms.some(([m]) => m === sm) ? sm : ms[0][0]; body = `<div class="row"><div class="field"><label for="pj-b">Model</label><select id="pj-b" class="inline">${have.map(b => `<option ${b === bb ? "selected" : ""}>${esc(b)}</option>`).join("")}</select></div><div class="field"><label for="pj-m">Method</label><select id="pj-m" class="inline">${ms.map(([m, l]) => `<option value="${m}" ${m === mm ? "selected" : ""}>${l}</option>`).join("")}</select></div></div>${cell(bb, mm, ms.find(([m]) => m === mm)[1], 640)}`; }
  box.innerHTML = `<div class="row"><div class="field"><label>Layout</label><div id="proj-layout" class="seg small" role="group" aria-label="Projection layout">${[["models", "Row per model"], ["methods", "Row per method"], ["single", "Single large"]].map(([k, l]) => `<button type="button" data-l="${k}" class="${lay === k ? "on" : ""}">${l}</button>`).join("")}</div></div></div>${legend(Object.keys(projData(A, have[0], ms.find(([m]) => projData(A, have[0], m))[0])))}${body}<p class="muted small">All points shown, no subsampling. Distances between far-apart clusters are not meaningful in t-SNE or UMAP; use them to see grouping, not to measure.</p>${agreementHtml(A)}`;
  box.querySelectorAll("#proj-layout button").forEach(x => x.onclick = () => { store.set("proj", x.dataset.l); projections(A); });
  const pb = $("#pj-b"), pm = $("#pj-m"); if (pb) { pb.onchange = () => { store.set("pb", pb.value); projections(A); }; pm.onchange = () => { store.set("pm", pm.value); projections(A); }; }
  const aa = $("#ag-a"), ab = $("#ag-b"); if (aa) aa.onchange = ab.onchange = () => { store.set("pair", [aa.value, ab.value]); projections(A); };
}
function hexPanel(name, P, cells, max, W = 230) {
  const [x0, x1, y0, y1] = P.box, n = P.bins, sx = (x1 - x0) / n, sy = (y1 - y0) / n * Math.sqrt(3), H = W * (y1 - y0) / ((x1 - x0) || 1) || W, kx = W / (x1 - x0), ky = H / (y1 - y0), r = sx * kx / Math.sqrt(3) * 1.05;
  const hexPath = (cx, cy) => Array.from({ length: 6 }, (_, i) => { const a = Math.PI / 3 * i + Math.PI / 6; return `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`; }).join(" ");
  const shade = c => { const t = Math.sqrt(c / max); return `rgb(${Math.round(232 - 221 * t)},${Math.round(240 - 179 * t)},${Math.round(250 - 143 * t)})`; };
  const hx = cells.map(([c, rw, l, k]) => { const cx = (c + 0.5 * l) * sx * kx, cy = H - (rw + 0.5 * l) * sy * ky; return `<polygon points="${hexPath(cx, cy)}" fill="${shade(k)}"><title>${esc(name)}: ${k} vector(s)</title></polygon>`; }).join("");
  return `<figure class="panel"><figcaption>${esc(name)}</figcaption><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="PCA density of ${esc(name)}" style="width:${W}px;max-width:100%;border:1px solid var(--hair)">${hx}</svg></figure>`;
}
function pcaCharts(A, b, key, out) {
  const box = out || $("#pca"), P = ((A.pca || {})[b] || {})[key];
  if (!P) { box.innerHTML = '<p class="muted">not computed for this representation (summarize.pca.enabled is off)</p>'; return ""; }
  const max = Math.max(1, ...Object.values(P.hex).flatMap(cs => cs.map(c => c[3])));
  let html = `<p class="muted small">Shared axes and shared density scale (lightest = 1, darkest = ${max} vectors per hexagon; hover for counts). Fit on ${Object.entries(P.fit_points).map(([s, n]) => `${n} ${esc(s)}`).join(", ")}; clipped to the 0.5–99.5% range: ${Object.entries(P.clipped).map(([s, n]) => `${esc(s)} ${n}`).join(", ")}.</p><div class="panels">${P.samples.map(s => hexPanel(s, P, P.hex[s], max)).join("")}</div>`;
  const cum = P.explained.reduce((a, v) => (a.push((a.length ? a[a.length - 1] : 0) + v), a), []);
  html += lines(`Cumulative explained variance (${P.components_90} components reach 90%)`, cum.map((_, i) => i + 1), { cumulative: cum, "90%": cum.map(() => 0.9) }, { xlabel: "components", ylabel: "fraction", ymax: 1 });
  box.innerHTML = html; html = "";
  if (P.per_sample) html += `<h4>Per-sample PCA (own axes)</h4><div class="panels">${P.samples.map(s => hexPanel(s, Object.assign({ bins: P.bins }, P.per_sample[s]), P.per_sample[s].hex, Math.max(1, ...P.per_sample[s].hex.map(c => c[3])))).join("")}</div>`;
  const bins = A.length_edges.slice(0, -1).map((e, i) => `${e}–${A.length_edges[i + 1]}`);
  html += `<h4>PC1 versus protein length ${P.pc1_length.r !== null && Math.abs(P.pc1_length.r) > 0.5 ? "▲ warning" : ""}</h4><p class="muted small">Pearson r = ${P.pc1_length.r ?? "—"}${P.pc1_length.r !== null && Math.abs(P.pc1_length.r) > 0.5 ? ": the first component largely tracks length, not biology" : ""}.</p>` + bars("Mean PC1 score by length", bins, { "mean PC1": P.pc1_length.bins }, v => v === null ? "—" : Number(v).toFixed(3));
  html += `<h4>Nearest-centroid assignment (rows: true sample, columns: nearest centroid)</h4><div class="scroll"><table><tr><th></th>${P.samples.map(s => `<th>${esc(s)}</th>`).join("")}<th>within-sample spread (mean 1 − cosine to own centroid)</th></tr>${P.confusion.map((row, i) => { const t = row.reduce((a, v) => a + v, 0) || 1; return `<tr><th>${esc(P.samples[i])}</th>${row.map((v, j) => `<td>${i === j ? "✓ " : ""}${v} (${(100 * v / t).toFixed(0)}%)</td>`).join("")}<td>${P.within_spread[P.samples[i]]}</td></tr>`; }).join("")}</table></div>`;
  return html;
}
function charts(r) {
  const A = r.aggregates, box = $("#charts"), sel = $("#agg-backend");
  if (!A) { box.innerHTML = '<p class="muted">no aggregates yet (04_summary)</p>'; $("#more").innerHTML = ""; sel.innerHTML = ""; return; }
  const samples = Object.keys(A.samples), bins = A.length_edges.slice(0, -1).map((e, i) => `${e}–${A.length_edges[i + 1]}`);
  const bnames = [...new Set([...Object.keys(A.backends), ...Object.keys(A.pca || {}), ...((A.twonn || {}).rows || []).map(x => x.backend)])];
  const cur = sel.value; sel.innerHTML = bnames.map(b => `<option ${b === cur ? "selected" : ""}>${esc(b)}</option>`).join(""); sel.onchange = () => charts(r);
  const lsel = $("#agg-layer"), lcur = lsel.value, keys = [...new Set([...Object.keys((A.pca || {})[sel.value] || {}), ...((A.twonn || {}).rows || []).filter(x => x.backend === sel.value).map(x => x.key)])].sort((x, y) => parseInt(x.split("_")[1]) - parseInt(y.split("_")[1]));
  lsel.innerHTML = keys.map(k => `<option ${k === lcur ? "selected" : ""}>${esc(k)}</option>`).join(""); if (!keys.includes(lcur) && keys.length) lsel.value = keys[keys.length - 1]; lsel.onchange = () => charts(r);
  const view = store.get("view", "model"), tsel = $("#agg-type"); tsel.value = store.get("vtype", "twonn");
  document.querySelectorAll("#view-mode button").forEach(x => { x.classList.toggle("on", x.dataset.view === view); x.onclick = () => { store.set("view", x.dataset.view); charts(r); }; });
  tsel.onchange = () => { store.set("vtype", tsel.value); charts(r); };
  $("#field-type").hidden = view !== "type"; $("#field-backend").hidden = $("#field-layer").hidden = view === "type";
  const last = b => { const ks = [...Object.keys((A.pca || {})[b] || {}), ...Object.keys((A.embed2d || {})[b] || {}), ...((A.twonn || {}).rows || []).filter(x => x.backend === b && x.pooling === "mean").map(x => x.key)].filter(k => k.endsWith("/mean")); return ks.sort((x, y) => parseInt(x.split("_")[1]) - parseInt(y.split("_")[1])).pop(); };
  const blocks = fn => bnames.map(b => { const d = document.createElement("div"); d.className = "by-model"; d.innerHTML = `<h4>${esc(b)} <span class="muted small">${esc(last(b) || "")}</span></h4><div></div>`; fn(b, last(b), d.lastChild); return d; });
  ["twonn", "pca", "embed2d"].forEach(t => { $("#card-" + t).hidden = view === "type" && tsel.value !== t; });
  let pcaMore = "";
  if (view === "type") {
    const t = tsel.value, tgt = $("#" + t), L = t === "embed2d" ? [] : blocks((b, k, o) => { if (t === "twonn") twonnCharts(A, b, k, o, false); else if (t === "pca") pcaCharts(A, b, k, o); }); 
    if (t === "embed2d") projections(A); else { tgt.innerHTML = t === "twonn" ? crossDepth((A.twonn || {}).rows || []) : ""; L.forEach(d => tgt.append(d)); }
  } else { twonnCharts(A, sel.value, lsel.value); pcaMore = pcaCharts(A, sel.value, lsel.value); projections(A); }
  const qc = bars("Protein length (kept records)", bins, Object.fromEntries(samples.map(s => [s, A.samples[s].length_hist])), v => `${v} proteins`);
  let html = "";
  if (samples.some(s => A.samples[s].dedup)) html += `<h4>MMseqs2 deduplication</h4><table><tr><th>sample</th><th>proteins</th><th>representatives</th><th>represented by another sample</th><th>settings</th></tr>${samples.map(s => { const d = A.samples[s].dedup; return d ? `<tr><td>${esc(s)}</td><td>${d.proteins}</td><td>${d.representatives}</td><td>${d.represented_in_other_samples}</td><td>${esc(d.mode)} id ${d.min_seq_id} cov ${d.coverage} (${esc(d.scope)}, mmseqs ${esc(d.mmseqs_version)})</td></tr>` : ""; }).join("")}</table>`;
  html += `<h4>Status per sample</h4><table><tr><th>sample</th>${[...new Set(samples.flatMap(s => Object.keys(A.samples[s].status)))].map(k => `<th>${esc(k)}</th>`).join("")}<th>embedded</th></tr>${samples.map(s => { const st = A.samples[s].status, keys = [...new Set(samples.flatMap(x => Object.keys(A.samples[x].status)))]; return `<tr><td>${esc(s)}</td>${keys.map(k => `<td>${st[k] || 0}</td>`).join("")}<td>${esc(Object.entries(A.samples[s].embedded || {}).map(([k, v]) => `${k} ${v}`).join(" · ") || "—")}</td></tr>`; }).join("")}</table>`;
  const B = A.backends[sel.value];
  if (B) {
    const centers = B.norm_edges.slice(0, -1).map((e, i) => ((e + B.norm_edges[i + 1]) / 2).toFixed(2));
    html += `<p class="muted small">Vectors: ${esc(B.key)} (mean of real residues).</p>` + bars("L2 norm of the mean vectors", centers, B.norm_hist, v => `${v} proteins`) + bars("Mean norm by protein length", bins, B.length_vs_norm, v => v === null ? "—" : Number(v).toFixed(3)) + heat("Cosine between sample centroids", B.centroid_cosine);
    const cbins = Array.from({ length: 20 }, (_, i) => (-1 + i * 0.1 + 0.05).toFixed(2));
    html += bars(`Cosine of random pairs: within sample (mean ${B.cosine.within.mean ?? "—"}, n ${B.cosine.within.n}) vs between samples (mean ${B.cosine.between.mean ?? "—"}, n ${B.cosine.between.n})`, cbins, { within: B.cosine.within.hist, between: B.cosine.between.hist }, v => `${v} pairs`);
  }
  box.innerHTML = qc; $("#more").innerHTML = html + (pcaMore || "");
}
function mini(pct, warnBelow = 50) { if (pct === null || pct === undefined) return '<span class="muted">—</span>'; const cls = pct >= 80 ? "ok" : pct >= warnBelow ? "warn" : "low"; return `<span class="mini" title="${pct}%"><i class="${cls}" style="width:${Math.min(100, pct)}%"></i></span> ${pct}%`; }
function ownBar(total, ours, used, unit) { const others = Math.max(0, used - ours); return `<div class="bar" role="img" aria-label="${ours} ${unit} ours, ${others} other users, ${Math.max(0, total - used)} free" title="${ours} ${unit} ours · ${others} others · ${Math.max(0, total - used)} free of ${total}">${ours ? `<i class="ours" style="width:${100 * ours / (total || 1)}%"></i>` : ""}<i class="others" style="width:${100 * others / (total || 1)}%"></i></div>`; }
async function refreshPerf() {
  const r = await api("/api/perf"), c = r.cluster, o = c.ours;
  $("#perf-at").textContent = "updated " + r.at;
  $("#perf-summary").textContent = c.available ? `◆ Our running jobs now: ${o.jobs} job(s), ${o.cpus} cores, ${o.gpus} GPU(s) on ${o.nodes.join(", ") || "no node"}.` : "sinfo is not available here: run the console on the cluster login node.";
  $("#partitions").innerHTML = `<div class="scroll"><table><tr><th>partition</th><th>nodes</th><th>cores (ours / others / free)</th><th>GPUs (ours / others / free)</th></tr>${c.partitions.map(p => `<tr><td>${p.ours_jobs ? "◆ " : ""}${esc(p.partition)}</td><td>${p.nodes}</td><td>${ownBar(p.cpus, p.ours_cpus, p.alloc, "cores")}</td><td>${p.gpus ? ownBar(p.gpus, p.ours_gpus, p.gpus_used, "GPUs") : '<span class="muted">no GPU</span>'}</td></tr>`).join("")}</table></div>`;
  $("#nodes").innerHTML = c.nodes.map(n => `<div class="tile ${n.ours_jobs.length ? "ours" : ""}" title="${esc(n.node)}: ${n.alloc}/${n.cpus} cores, ${n.gpus_used}/${n.gpus} GPUs, ${n.state}"><div class="name">${esc(n.node)}</div><div class="muted">${esc(n.partitions.join(", "))} · ${esc(n.state)}</div>${ownBar(n.cpus, n.ours_cpus, n.alloc, "cores")}${n.gpus ? ownBar(n.gpus, n.ours_gpus, n.gpus_used, "GPUs") : ""}${n.ours_jobs.length ? `<div>◆ ${n.ours_jobs.length} of our jobs</div>` : ""}</div>`).join("");
  const agg = rows => `<div class="scroll"><table><tr><th>name</th><th>jobs</th><th>✓</th><th>✕</th><th>OOM</th><th>timeout</th><th>CPU eff.</th><th>memory eff.</th><th>wait (min)</th><th>core-h</th><th>GPU-h</th></tr>${rows.map(t => `<tr><td>${esc(t.name)}</td><td>${t.n}</td><td>${t.completed}</td><td>${t.failed}</td><td>${t.oom}</td><td>${t.timeout}</td><td>${mini(t.cpu_eff)}</td><td>${mini(t.mem_eff)}</td><td>${t.wait_min ?? "—"}</td><td>${t.core_h}</td><td>${t.gpu_h}</td></tr>`).join("")}</table></div>`;
  $("#throughput").innerHTML = Object.keys(r.throughput).length ? `<table><tr><th>backend</th><th>proteins</th><th>shards</th><th>GPU-job hours</th><th>proteins / GPU-job hour</th></tr>${Object.entries(r.throughput).map(([b, t]) => `<tr><td>${esc(b)}</td><td>${t.proteins}</td><td>${t.shards}</td><td>${(t.gpu_s / 3600).toFixed(2)}</td><td>${t.proteins_per_gpu_hour ?? "—"}</td></tr>`).join("")}</table>` : '<p class="muted">no finished embedding yet</p>';
  $("#per-tool").innerHTML = agg(r.runs.tools); $("#per-analysis").innerHTML = agg(r.runs.analyses);
  $("#jobs").innerHTML = `<div class="scroll"><table><tr><th>analysis</th><th>job</th><th>state</th><th>node</th><th>cores</th><th>memory used / asked</th><th>time used / asked</th><th>CPU eff.</th><th>wait</th><th>GPU-h</th></tr>${r.runs.jobs.map(j => `<tr><td>${esc(j.analysis)}</td><td>${j.gpu ? "▣ " : ""}${esc(j.key)}</td><td>${esc(j.state)}</td><td>${esc(j.node)}</td><td>${j.cpus}</td><td>${mini(j.mem_eff)} ${j.used_mb}/${j.req_mb} MB</td><td>${mini(j.time_eff)} ${j.elapsed_min}/${j.req_min ?? "—"} min</td><td>${mini(j.cpu_eff)}</td><td>${j.wait_min ?? "—"}</td><td>${j.gpu_h}</td></tr>`).join("")}</table></div>`;
  document.querySelectorAll("#partitions table, #per-tool table, #per-analysis table, #jobs table").forEach(t => { if (t.rows.length < 2) t.insertAdjacentHTML("afterend", '<p class="muted">no data yet</p>'); });
}
function setPerf(on) {
  $("#perf").hidden = !on; $("#console").hidden = on; document.body.classList.toggle("perf-on", on); $("#perf-toggle").setAttribute("aria-pressed", on);
  clearInterval(perfTimer); if (on) { refreshPerf(); perfTimer = setInterval(refreshPerf, 30000); if (location.hash !== "#perf") history.replaceState(null, "", "#perf"); } else if (location.hash === "#perf") history.replaceState(null, "", location.pathname);
}

async function init() {
  schema = await api("/api/schema"); examples = (await api("/api/examples")).examples;
  $("#info").innerHTML = `<p>${esc(schema.info)}</p>`;
  const hide = store.get("hideInfo", false); $("#info").hidden = hide; $("#toggle-info").textContent = hide ? "Show info" : "Hide info"; $("#toggle-info").setAttribute("aria-expanded", !hide);
  $("#toggle-info").onclick = () => { const h = !$("#info").hidden; $("#info").hidden = h; store.set("hideInfo", h); $("#toggle-info").textContent = h ? "Show info" : "Hide info"; $("#toggle-info").setAttribute("aria-expanded", !h); };
  $("#example").innerHTML = '<option value="">Saved samples (editable)</option>' + examples.map(e => `<option value="${esc(e.name)}">${esc(e.title)}</option>`).join("") + (examples.length > 1 ? '<option value="__all">All examples (read-only)</option>' : "");
  $("#example").onchange = () => { const v = $("#example").value; if (!v) return loadSaved(); const st = $("#samples-status"); st.hidden = false;
    if (v === "__all") { showSamples(examples.flatMap(e => e.samples), true); showBackends([...new Set(examples.flatMap(e => e.backends))], true); showOptions(Object.fromEntries(OPTS.map(k => [k, examples.some(e => e.options[k])])), true); st.textContent = "All examples (read-only): each runs as its own analysis with its own parameters. Go to Run, switch on 'Run the selected example', and start."; return; }
    const e = examples.find(x => x.name === v); showSamples(e.samples.map(s => ({ ...s }))); showBackends(e.backends, false); showOptions(e.options, false); st.textContent = `${e.title} — models ${e.backends.join(", ")} — ${e.credit}\n\nThe cards are editable. Click Save samples to use them (it replaces the saved samples and models).`; };
  document.querySelectorAll(".seg button[data-tab]").forEach(b => b.onclick = () => setTab(b.dataset.tab));
  $("#clear").onclick = clearSamples; $("#add").onclick = () => $("#samples").append(card({})); $("#reload").onclick = loadSaved; $("#save").onclick = saveSamples;
  $("#run").onclick = run; $("#stop").onclick = () => stop(false); $("#stop-all").onclick = () => stop(true);
  $("#pick").onchange = () => { $("#analysis").value = $("#pick").value; store.set("analysis", $("#pick").value); refreshRun(); };
  $("#res-pick").onchange = refreshResults;
  $("#analysis").value = store.get("analysis", "");
  $("#perf-toggle").onclick = () => setPerf($("#perf").hidden); $("#perf-back").onclick = () => setPerf(false);
  timer = setInterval(() => { if ($("#auto").checked && !$("#tab-run").hidden) refreshRun(); }, 5000);
  const q = new URLSearchParams(location.search); if (q.get("analysis")) store.set("res", q.get("analysis"));
  await loadSaved(); await loadBackends(); showOptions(await api("/api/options"), false); setTab(q.get("tab") || store.get("tab", "samples")); if (location.hash === "#perf") setPerf(true);
}
init().catch(e => { document.body.insertAdjacentHTML("afterbegin", `<pre class="note">✕ console failed to load: ${esc(e.message || JSON.stringify(e))}</pre>`); });
