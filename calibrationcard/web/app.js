"use strict";
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const COLORS = { A: "#2e7d5b", B: "#4f8a3a", C: "#c98a16", D: "#d0662b", F: "#d64541", I: "#5e6b78" };
let upload = null;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function status(id, text, err = false) { const el = $(id); el.textContent = text; el.classList.toggle("error", err); }

async function loaded(promise) {
  status("#load-status", "Reading…");
  try {
    upload = await promise;
    status("#load-status", `${upload.name}: ${upload.rows.toLocaleString()} rows, ${upload.columns.length} columns.`);
    showMapping();
  } catch (e) { status("#load-status", e.message, true); }
}

function showMapping() {
  const cols = upload.columns;
  $("#label-col").innerHTML = cols.map((c) => `<option ${c === upload.label_guess ? "selected" : ""}>${esc(c)}</option>`).join("");
  $("#prob-cols").innerHTML = cols.map((c) => `<label><input type="checkbox" value="${esc(c)}" ${upload.prob_guess.includes(c) ? "checked" : ""}> ${esc(c)}</label>`).join("");
  $("#head").innerHTML = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
    upload.head.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c])}</td>`).join("")}</tr>`).join("");
  $("#prob-cols").onchange = togglePositive;
  togglePositive();
  $("#map").classList.remove("hidden");
  $("#result").classList.add("hidden");
}

function chosenProbs() { return [...document.querySelectorAll("#prob-cols input:checked")].map((i) => i.value); }
function togglePositive() { $("#pos-wrap").classList.toggle("hidden", chosenProbs().length !== 1); }

async function grade() {
  const btn = $("#grade");
  btn.disabled = true;
  status("#grade-status", "Grading…");
  try {
    const r = await postJSON("/api/analyze", {
      upload_id: upload.upload_id, label: $("#label-col").value, prob_columns: chosenProbs(),
      positive: $("#positive").value, bins: +$("#bins").value, recalibrate: $("#recal").checked,
    });
    const h = r.headline;
    const g = $("#grade-letter");
    g.textContent = h.grade;
    g.style.color = g.style.borderColor = COLORS[h.grade[0]];
    $("#headline").textContent = r.result.grade.headline;
    const pct = (x) => (x * 100).toFixed(1) + "%";
    $("#numbers").innerHTML = [["ECE", pct(h.ece)], ["Brier", h.brier.toFixed(4)], ["Log loss", h.log_loss.toFixed(4)],
      ["Accuracy", pct(h.accuracy)], ["Rows", h.rows.toLocaleString()]].map(([k, v]) => `<div><b>${v}</b>${k}</div>`).join("");
    $("#dl-html").href = `/api/report/${r.result_id}.html?download=1`;
    $("#dl-pdf").href = `/api/report/${r.result_id}.pdf`;
    $("#dl-json").href = `/api/report/${r.result_id}.json`;
    $("#report").src = `/api/report/${r.result_id}.html`;
    $("#result").classList.remove("hidden");
    status("#grade-status", `Done in ${r.result.seconds}s.`);
    if (!new URLSearchParams(location.search).get("demo")) $("#result").scrollIntoView({ behavior: "smooth" });
  } catch (e) { status("#grade-status", e.message, true); } finally { btn.disabled = false; }
}

function init() {
  $("#file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const form = new FormData(); form.append("file", f);
    loaded(api("/api/upload", { method: "POST", body: form }));
  });
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const form = new FormData(); form.append("file", f);
    loaded(api("/api/upload", { method: "POST", body: form }));
  });
  $("#open-path").addEventListener("click", () => loaded(postJSON("/api/upload", { path: $("#path").value })));
  $("#demo").addEventListener("click", () => loaded(postJSON("/api/upload", { demo: true })));
  $("#grade").addEventListener("click", grade);
  api("/api/meta").then((m) => {
    $("#version").textContent = "v" + m.version;
    if (!m.pdf) { $("#dl-pdf").title = "Needs Chrome, Chromium or Edge. Use the HTML report and Print → Save as PDF."; }
  }).catch(() => {});
  if (new URLSearchParams(location.search).get("demo") === "1") {
    loaded(postJSON("/api/upload", { demo: true })).then(() => grade());
  }
}
init();
