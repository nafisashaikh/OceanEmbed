/* Depth Reconstruction Explorer — honest, live controller.
 * Every value comes from the FastAPI backend (real model output + ARGO skill). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { date: null, di: 0, floats: [], metrics: null, busy: false };

  // local vertical-profile builder (domain column mean)
  function colProfile(div, depths, col) {
    const C = OE.C;
    Plotly.react(
      div,
      [{
        x: col, y: depths, mode: "lines+markers",
        line: { color: C.model, width: 3, shape: "spline" },
        marker: { size: 6, color: C.model },
        hovertemplate: "%{x:.2f} °C @ %{y:.0f} m<extra></extra>",
      }],
      {
        font: { family: "Inter, sans-serif", color: C.mut, size: 12 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 54, r: 14, t: 10, b: 42 },
        xaxis: { title: "Temperature (°C)", gridcolor: C.grid, zeroline: false },
        yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: C.grid },
        hoverlabel: { bgcolor: "#0B1220", bordercolor: C.line,
          font: { family: "Inter", size: 12, color: C.ink } },
      },
      { displaylogo: false, responsive: true, displayModeBar: false }
    );
  }

  function renderSkillCard(di) {
    if (!state.metrics) return;
    const m = state.metrics.model[di] || {}, c = state.metrics.climatology[di] || {};
    const depth = state.metrics.depths[di];
    OE.fill("dr-depth-label", OE.fmt.c(depth, 0) + " m");
    OE.fill("dr-rmse-val", OE.fmt.c(m.rmse));
    OE.fill("dr-mae-val", OE.fmt.c(m.mae));
    OE.fill("dr-bias-val", (m.bias >= 0 ? "+" : "") + OE.fmt.c(m.bias));
    OE.fill("dr-corr-val", OE.fmt.c(m.corr));
    OE.fill("dr-n-val", OE.fmt.int(m.n));
    const beats = m.rmse != null && c.rmse != null && m.rmse < c.rmse;
    const icon = $("dr-beat-icon");
    if (icon) {
      icon.textContent = beats ? "check_circle" : "cancel";
      icon.className = "material-symbols-outlined text-[18px] " +
        (beats ? "text-secondary-fixed" : "text-error");
    }
    OE.fill("dr-beat-note", beats
      ? "Beats climatology here (" + OE.fmt.c(c.rmse) + " °C) by " + OE.fmt.c(c.rmse - m.rmse) + " °C."
      : (c.rmse != null ? "Climatology is stronger at this level (" + OE.fmt.c(c.rmse) + " °C)." : "No baseline at this level."));
  }
  async function boot() {
    let meta;
    try { meta = await OE.api.meta(); }
    catch (e) { console.error("meta failed", e); return; }
    const m = meta.model;
    OE.fill("dr-levels", meta.depths.length);
    OE.fill("dr-grid", m.output_size[0] + "×" + m.output_size[1]);
    OE.fill("hdr-epoch", m.epoch);
    OE.fill("dr-emb-dim", m.embedding_dim);
    OE.fill("side-params", (m.parameters / 1e6).toFixed(2) + " M");

    const dsel = $("dr-date");
    (meta.predicted_dates || []).forEach((d, i) => dsel.appendChild(OE.option(d, d, i === 0)));
    state.date = dsel.value || (meta.predicted_dates || [])[0];
    const zsel = $("dr-depth");
    meta.depths.forEach((d, i) => zsel.appendChild(OE.option(i, d + " m", i === 0)));

    dsel.addEventListener("change", () => { state.date = dsel.value; runLive(); });
    zsel.addEventListener("change", () => { state.di = +zsel.value; renderSkillCard(state.di); runLive(); });
    $("dr-run").addEventListener("click", runLive);

    try {
      const f = await OE.api.field(state.date, state.di);
      OE.heatmap($("dr-map"), f);
    } catch (e) { console.error("field failed", e); }

    loadMetrics();
    loadFloats();
    runLive();
  }

  async function loadMetrics() {
    try {
      state.metrics = await OE.api.metrics();
      OE.rmseByDepth($("dr-rmse"), state.metrics);
      renderSkillCard(state.di);
    } catch (e) { console.error("metrics failed", e); }
  }

  async function loadFloats() {
    try {
      const data = await OE.api.floats();
      state.floats = data.floats || [];
      const sel = $("dr-float");
      state.floats.forEach((f, i) =>
        sel.appendChild(OE.option(i, "#" + f.id + " · " + f.n + " obs", i === 0)));
      sel.addEventListener("change", () => OE.profile($("dr-profile"), state.floats[+sel.value]));
      if (state.floats.length) OE.profile($("dr-profile"), state.floats[0]);
    } catch (e) { console.error("floats failed", e); }
  }

  async function runLive() {
    if (state.busy) return;
    state.busy = true;
    const btn = $("dr-run");
    const label = btn.querySelector("span:last-child");
    const prev = label.textContent;
    label.textContent = "Running…";
    btn.disabled = true;
    try {
      const r = await OE.api.reconstruct(state.date, state.di);
      OE.heatmap($("dr-map"), r);
      colProfile($("dr-col"), r.depths, r.column_mean);
      OE.fill("dr-source", "Live U-Net forward pass");
      OE.fill("dr-latency", OE.fmt.c(r.latency_ms, 1) + " ms");
      OE.fill("side-latency", OE.fmt.c(r.latency_ms, 0) + " ms");
      OE.fill("dr-emb", OE.fmt.c(r.embedding_norm, 3));
      if (r.agreement)
        OE.fill("dr-agree", OE.fmt.c(r.agreement.rmse_vs_stored, 3) + " °C · " +
          OE.fmt.int(r.agreement.cells) + " cells");
    } catch (e) {
      console.error("reconstruct failed", e);
      OE.fill("dr-source", "Live inference unavailable");
    } finally {
      label.textContent = prev;
      btn.disabled = false;
      state.busy = false;
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
