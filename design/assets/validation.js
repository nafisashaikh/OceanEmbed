/* Scientific Validation page controller — honest, live.
 * All figures come from the FastAPI backend (real ARGO skill + matchups). */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { floats: [] };

  async function boot() {
    let meta;
    try { meta = await OE.api.meta(); }
    catch (e) { console.error("meta failed", e); return; }
    const h = meta.headline, m = meta.model;

    // headline KPIs
    OE.fill("kpi-rmse", OE.fmt.c(h.model_rmse));
    OE.fill("kpi-impr", OE.fmt.c(h.improvement_pct, 1) + "%");
    OE.fill("kpi-obs", OE.fmt.int(h.n_obs));
    OE.fill("kpi-obs-sub", "ARGO obs · " + h.n_floats + " floats");

    // shell
    OE.fill("hdr-epoch", m.epoch);
    OE.fill("side-params", (m.parameters / 1e6).toFixed(2) + " M");
    OE.fill("side-obs", OE.fmt.int(h.n_obs));
    OE.fill("side-floats", h.n_floats + " FLOATS");

    // skill gauges
    OE.fill("v-rmse", OE.fmt.c(h.model_rmse));
    OE.fill("v-rmse-sub", "vs climatology " + OE.fmt.c(h.clim_rmse) +
      " · GLORYS ceiling " + OE.fmt.c(h.glorys_rmse));
    OE.fill("v-r", OE.fmt.c(h.r_max));
    OE.fill("v-beat", h.depths_beat + " / " + h.depths_total);

    // significance statement
    const p = h.p_value < 0.001 ? "p < 0.001" : "p = " + OE.fmt.c(h.p_value, 3);
    OE.fill("v-ttest",
      "Paired t-test vs climatology: t = " + OE.fmt.c(h.t_stat, 1) + ", " + p +
      " across " + OE.fmt.int(h.n_obs) + " matched observations from " + h.n_floats +
      " independent floats — the error reduction is statistically overwhelming.");

    loadScatter();
    loadMetrics();
    loadFloats();
  }
  async function loadScatter() {
    try {
      const s = await OE.api.scatter();
      OE.parity($("v-parity"), s);
      OE.fill("v-n", OE.fmt.int(s.n));
      OE.fill("v-r2", OE.fmt.c(s.r2));
      OE.fill("v-prmse", OE.fmt.c(s.rmse));
      OE.fill("v-mae", OE.fmt.c(s.mae));
      OE.fill("v-bias", (s.bias >= 0 ? "+" : "") + OE.fmt.c(s.bias));
    } catch (e) { console.error("scatter failed", e); }
  }

  async function loadMetrics() {
    try {
      const m = await OE.api.metrics();
      OE.rmseByDepth($("v-rmse-depth"), m);
    } catch (e) { console.error("metrics failed", e); }
  }

  async function loadFloats() {
    try {
      const data = await OE.api.floats();
      state.floats = data.floats || [];
      const casts = state.floats.reduce((s, f) => s + f.n, 0);
      OE.fill("v-float-count", (data.count || state.floats.length) + " floats · " +
        OE.fmt.int(casts) + " casts");
      const tb = $("v-float-list");
      tb.innerHTML = "";
      state.floats.slice().sort((a, b) => b.n - a.n).forEach((f) => {
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
    } catch (e) { console.error("floats failed", e); }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
