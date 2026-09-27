/* Overview page controller — wires the honest, live OceanEmbed dashboard.
 * All values come from the FastAPI backend; nothing is hard-coded. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { date: null, di: 0, floats: [], meta: null, busy: false };

  async function boot() {
    let meta;
    try {
      meta = await OE.api.meta();
    } catch (e) {
      console.error("meta failed", e);
      return;
    }
    state.meta = meta;
    const h = meta.headline, m = meta.model;

    // headline KPIs
    OE.fill("kpi-rmse", OE.fmt.c(h.model_rmse));
    OE.fill("kpi-impr", OE.fmt.c(h.improvement_pct, 1) + "%");
    OE.fill("kpi-obs", OE.fmt.int(h.n_obs));
    OE.fill("kpi-obs-sub", "ARGO obs · " + h.n_floats + " floats");

    // header + sidebar + pipeline (model facts)
    OE.fill("hdr-epoch", m.epoch);
    OE.fill("pipe-epoch", m.epoch);
    OE.fill("side-params", (m.parameters / 1e6).toFixed(2) + " M");
    OE.fill("pipe-params", OE.fmt.int(m.parameters));
    OE.fill("pipe-in-ch", m.input_channels);
    OE.fill("pipe-out-ch", m.output_channels);
    OE.fill("pipe-emb", m.embedding_dim);
    OE.fill("tele-emb-dim", m.embedding_dim);
    OE.fill("pipe-grid", m.output_size[0] + " × " + m.output_size[1]);

    // validation trio
    OE.fill("val-r", OE.fmt.c(h.r_max));
    OE.fill("val-beat", h.depths_beat + " / " + h.depths_total);
    OE.fill("val-gap", (h.ceiling_gap >= 0 ? "+" : "") + OE.fmt.c(h.ceiling_gap));
    const p = h.p_value < 0.001 ? "p < 0.001" : "p = " + OE.fmt.c(h.p_value, 3);
    OE.fill(
      "val-ttest",
      "Paired t-test vs climatology: t = " + OE.fmt.c(h.t_stat, 1) + ", " + p +
        " across " + OE.fmt.int(h.n_obs) + " matched observations from " + h.n_floats + " floats."
    );

    // selectors
    const dsel = $("map-date");
    (meta.predicted_dates || []).forEach((d, i) => dsel.appendChild(OE.option(d, d, i === 0)));
    state.date = dsel.value || (meta.predicted_dates || [])[0];
    const zsel = $("map-depth");
    meta.depths.forEach((d, i) => zsel.appendChild(OE.option(i, d + " m", i === 0)));

    dsel.addEventListener("change", () => { state.date = dsel.value; runLive(); });
    zsel.addEventListener("change", () => { state.di = +zsel.value; runLive(); });
    $("btn-recon").addEventListener("click", runLive);

    // instant paint from the stored field, then hand off to live inference
    try {
      const f = await OE.api.field(state.date, state.di);
      OE.heatmap($("map-chart"), f);
    } catch (e) { console.error("field failed", e); }

    loadCharts();
    loadInputs();
    runLive();
  }

  async function runLive() {
    if (state.busy) return;
    state.busy = true;
    const btn = $("btn-recon");
    const label = btn.querySelector("span:last-child");
    const prev = label.textContent;
    label.textContent = "Running…";
    btn.disabled = true;
    try {
      const r = await OE.api.reconstruct(state.date, state.di);
      OE.heatmap($("map-chart"), r);
      OE.fill("tele-source", "Live U-Net forward pass");
      OE.fill("tele-latency", OE.fmt.c(r.latency_ms, 1) + " ms");
      OE.fill("tele-emb", OE.fmt.c(r.embedding_norm, 3));
      OE.fill("pipe-latency", OE.fmt.c(r.latency_ms, 0) + " ms");
      OE.fill("side-latency", OE.fmt.c(r.latency_ms, 0) + " ms");
      if (r.agreement)
        OE.fill("tele-agree", OE.fmt.c(r.agreement.rmse_vs_stored, 3) + " °C · " +
          OE.fmt.int(r.agreement.cells) + " cells");
    } catch (e) {
      console.error("reconstruct failed", e);
      OE.fill("tele-source", "Live inference unavailable");
    } finally {
      label.textContent = prev;
      btn.disabled = false;
      state.busy = false;
    }
  }

  async function loadCharts() {
    try {
      const m = await OE.api.metrics();
      OE.rmseByDepth($("rmse-chart"), m);
    } catch (e) { console.error("metrics failed", e); }
    try {
      const data = await OE.api.floats();
      state.floats = data.floats || [];
      OE.fill("float-count", (data.count || state.floats.length) + " floats · " +
        state.floats.reduce((s, f) => s + f.n, 0) + " casts");
      const sel = $("float-sel");
      state.floats.forEach((f, i) =>
        sel.appendChild(OE.option(i, "#" + f.id + " · " + f.n + " obs", i === 0)));
      sel.addEventListener("change", () => OE.profile($("profile-chart"), state.floats[+sel.value]));
      if (state.floats.length) OE.profile($("profile-chart"), state.floats[0]);
      renderFloatTable();
    } catch (e) { console.error("floats failed", e); }
  }

  function renderFloatTable() {
    const tb = $("float-list");
    if (!tb) return;
    const rows = state.floats.slice().sort((a, b) => b.n - a.n);
    tb.innerHTML = "";
    rows.forEach((f) => {
      const tr = document.createElement("tr");
      tr.className = "border-b border-outline-variant/20 hover:bg-surface-container/40 transition-colors";
      tr.innerHTML =
        '<td class="py-space-xs pr-space-md font-body-sm text-primary">#' + f.id + "</td>" +
        '<td class="py-space-xs pr-space-md font-body-sm text-on-surface-variant">' +
          f.lat.toFixed(2) + "°N, " + f.lon.toFixed(2) + "°E</td>" +
        '<td class="py-space-xs pr-space-md font-body-sm text-on-surface text-right">' + OE.fmt.int(f.n) + "</td>" +
        '<td class="py-space-xs pr-space-md font-body-sm text-secondary-fixed text-right">' + f.rmse.toFixed(2) + "</td>";
      tb.appendChild(tr);
    });
  }

  async function loadInputs() {
    try {
      const d = await OE.api.inputs(state.date);
      OE.fill("inputs-channels", d.n_channels);
      OE.fill("inputs-note", d.variables.length + " variables · 2-day window");
      const grid = $("inputs-grid");
      grid.innerHTML = "";
      d.variables.forEach((v) => {
        const el = document.createElement("div");
        el.className =
          "group rounded-xl bg-surface-container-low/70 backdrop-blur-xl p-space-md flex flex-col justify-between hover:bg-surface-container/80 transition-all shadow-md";
        el.innerHTML =
          '<div class="flex items-center justify-between pb-space-xs"><span class="font-label-mono text-label-mono text-on-surface-variant uppercase tracking-wider">' +
            v.key + '</span><span class="font-label-mono text-label-mono text-secondary">' + v.unit + "</span></div>" +
          '<div class="font-title-sm text-title-sm text-on-surface-variant">' + v.label + "</div>" +
          '<div class="flex items-baseline gap-space-xs mt-space-2xs"><span class="font-telemetry-value text-telemetry-value text-primary">' +
            v.mean + '</span><span class="font-label-mono text-label-mono text-on-surface-variant">mean</span></div>' +
          '<div class="pt-space-xs font-label-mono text-label-mono text-on-surface-variant">range ' +
            v.min + " – " + v.max + "</div>";
        grid.appendChild(el);
      });
    } catch (e) { console.error("inputs failed", e); }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
