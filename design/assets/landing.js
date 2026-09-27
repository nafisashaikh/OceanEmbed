/* Landing / cover page controller — honest, live.
 * Every headline number and the hero field come from the FastAPI backend
 * (real epoch-81 checkpoint + ARGO validation summary). Nothing is hard-coded. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  async function boot() {
    let meta;
    try { meta = await OE.api.meta(); }
    catch (e) { console.error("meta failed", e); return; }
    const h = meta.headline, m = meta.model;

    const rmse = OE.fmt.c(h.model_rmse);
    const impr = OE.fmt.c(h.improvement_pct, 1) + "%";
    const floats = h.n_floats;
    const obs = OE.fmt.int(h.n_obs);
    const gap = OE.fmt.c(Math.abs(h.ceiling_gap));

    // hero chip + centerpiece stat cards
    OE.fill("chip-rmse", rmse);
    OE.fill("home-rmse", rmse);
    OE.fill("home-impr", impr);
    OE.fill("home-floats", floats);
    OE.fill("home-obs", obs);
    OE.fill("home-gap", gap);

    // pipeline validation card + verification strip
    OE.fill("card-rmse", rmse);
    OE.fill("card-beat", h.depths_beat + "/" + h.depths_total);
    OE.fill("card-r", OE.fmt.c(h.r_max));
    OE.fill("strip-obs", obs);
    OE.fill("strip-rmse", rmse);
    OE.fill("strip-impr", impr);

    // shell
    OE.fill("hdr-epoch", m.epoch);
    OE.fill("side-params", (m.parameters / 1e6).toFixed(2) + " M");
    OE.fill("side-floats", floats);

    // pipeline bento (stage 02 detail)
    OE.fill("bento-params", (m.parameters / 1e6).toFixed(2) + " M");
    OE.fill("bento-epoch", m.epoch);

    // hero live field — real reconstructed surface slice, first test date
    const date = (meta.predicted_dates || [])[0];
    OE.fill("home-date", date || "");
    if (date) {
      try {
        const f = await OE.api.field(date, 0);
        OE.heatmap($("home-map"), f, { unit: "°C" });
      } catch (e) { console.error("field failed", e); }
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
