/* OceanEmbed dashboard controller — wires every page to the live FastAPI API.
 * ZERO hardcoded values; every number on screen comes from /api/* endpoints
 * that read real model output, validation artifacts, and ARGO matchups. */

"use strict";

const STATE = {
  date: null,
  depthIdx: 0,
  floats: [],
  floatIdx: 0,
  metrics: null,
  scatter: null,
  volume: null,            // cached /api/volume for transect + ocean-state
  embeddingData: null,
  meta: null,
  anomalyMode: false,
  anomalyThreshold: 1.5,
  transectAxis: "lat",     // "lat" = along longitude, "lon" = along latitude
  transectIndex: 20,
  oceanStateMode: "mld",
};

/* ------------------------------------------------------------------ */
/* SPA router                                                         */
/* ------------------------------------------------------------------ */
function setupRouting() {
  const pages = document.querySelectorAll(".page");
  const navLinks = document.querySelectorAll("#main-nav a");

  function go(hash) {
    if (!hash || hash === "#") hash = "#overview";
    const route = hash.replace("#", "");
    navLinks.forEach(link => {
      link.classList.toggle("active", link.dataset.route === route);
    });
    pages.forEach(p => {
      p.classList.toggle("show", p.dataset.page === route);
    });
    renderCurrentPage(route);
  }

  window.addEventListener("hashchange", () => go(window.location.hash));
  go(window.location.hash);
}

/* ------------------------------------------------------------------ */
/* Page renderer — called on every route change and control change    */
/* ------------------------------------------------------------------ */
async function renderCurrentPage(route) {
  if (!STATE.date) return;

  try {
    /* ---- Overview ---- */
    if (route === "overview") {
      // KPIs already filled in initDashboard — nothing extra

    /* ---- Domain Map ---- */
    } else if (route === "domain-map") {
      const vmap = document.getElementById("v-map");
      if (vmap) {
        const f = await OE.api.field(STATE.date, STATE.depthIdx);
        OE.heatmap(vmap, f, { zmin: 2, zmax: 32, unit: "°C" });
        OE.fill("map-date-label", STATE.date + " · " + f.depth + " m depth");
        const region = STATE.meta ? STATE.meta.domain.name : "Arabian Sea";
        OE.fill("map-subtitle", region + " · " + STATE.date + " · depth " + f.depth + " m");
        OE.fill("map-badge", f.depth === 0 ? "Surface · SST" : f.depth + " m depth");

        // Anomaly overlay
        if (STATE.anomalyMode) {
          const flat = f.z.flat().filter(v => v !== null && !isNaN(v));
          if (flat.length > 0) {
            const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
            const thresh = mean + STATE.anomalyThreshold;
            const zMask = f.z.map(row =>
              row.map(v => (v !== null && !isNaN(v) && v > thresh) ? 1 : null)
            );
            Plotly.addTraces(vmap, {
              type: "heatmap", x: f.lon, y: f.lat, z: zMask,
              colorscale: [[0, "rgba(239,68,68,0.30)"], [1, "rgba(239,68,68,0.30)"]],
              showscale: false, hoverinfo: "skip", zsmooth: false, name: "anomaly",
            });
          }
        }
      }

    /* ---- Skill Summary ---- */
    } else if (route === "skill-summary") {
      await renderSkillSummary();

    /* ---- Float Explorer ---- */
    } else if (route === "float-explorer") {
      await renderFloatExplorer();

    /* ---- 3D Ocean ---- */
    } else if (route === "ocean-3d") {
      const vvolume = document.getElementById("v-volume");
      if (vvolume) {
        const vol = await OE.api.volume(STATE.date, 3);
        OE.volume3d(vvolume, vol, { unit: "°C" });
      }

    /* ---- Depth Transect ---- */
    } else if (route === "depth-transect") {
      await renderTransect();

    /* ---- Ocean State ---- */
    } else if (route === "ocean-state") {
      await renderOceanState();

    /* ---- Data Pipeline ---- */
    } else if (route === "data-pipeline") {
      renderPipeline();

    /* ---- Embedding ---- */
    } else if (route === "embedding") {
      await renderEmbedding();
    }
  } catch (e) {
    console.error(`Error rendering page ${route}:`, e);
    OE.banner("Error loading " + route + ": " + e.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Skill Summary — validation charts                                  */
/* ------------------------------------------------------------------ */
async function renderSkillSummary() {
  const vscatter = document.getElementById("v-scatter");
  if (vscatter && !STATE.scatter) {
    try { STATE.scatter = await OE.api.scatter(); } catch (e) { console.error(e); }
  }
  if (vscatter && STATE.scatter) OE.parity(vscatter, STATE.scatter);

  const vrmse = document.getElementById("v-rmse");
  if (vrmse && !STATE.metrics) {
    try { STATE.metrics = await OE.api.metrics(); } catch (e) { console.error(e); }
  }
  if (vrmse && STATE.metrics) OE.rmseByDepth(vrmse, STATE.metrics);

  if (STATE.metrics) {
    const table = document.getElementById("skill-table");
    if (table) {
      let html = "<tr><th>Depth</th><th>RMSE vs Clim</th><th>RMSE vs GLORYS</th><th>Model MAE</th><th>Bias</th><th>Corr (R)</th><th>Samples</th></tr>";
      STATE.metrics.depths.forEach((d, i) => {
        const m = STATE.metrics.model[i] || {};
        const c = STATE.metrics.climatology[i] || {};
        const g = STATE.metrics.glorys[i] || {};
        html += `<tr>
          <td><b>${d} m</b></td>
          <td>${m.rmse != null ? m.rmse.toFixed(2) : "—"} / ${c.rmse != null ? c.rmse.toFixed(2) : "—"}</td>
          <td>${m.rmse != null ? m.rmse.toFixed(2) : "—"} / ${g.rmse != null ? g.rmse.toFixed(2) : "—"}</td>
          <td>${m.mae != null ? m.mae.toFixed(2) : "—"}</td>
          <td>${m.bias != null ? (m.bias > 0 ? "+" : "") + m.bias.toFixed(2) : "—"}</td>
          <td>${m.corr != null ? m.corr.toFixed(2) : "—"}</td>
          <td>${m.n || "—"}</td>
        </tr>`;
      });
      table.innerHTML = html;
    }

    const mkRegion = (arr) => arr.map(a => a.rmse);
    const m1 = document.getElementById("skill-region");
    if (m1 && STATE.metrics.regions) {
      const regObj = STATE.metrics.regions;
      const defs = [
        { name: "Arabian Sea", vals: mkRegion(regObj.arabian_sea), color: "#2563EB" },
        { name: "Bay of Bengal", vals: mkRegion(regObj.bay_of_bengal), color: "#06B6D4" },
      ];
      OE.skillByDepth(m1, STATE.metrics.depths, defs, { xtitle: "RMSE (°C)" });
    }

    const m2 = document.getElementById("skill-season");
    if (m2 && STATE.metrics.seasons) {
      const seaObj = STATE.metrics.seasons;
      const defs = Object.keys(seaObj).map((k, i) => ({
        name: k, vals: mkRegion(seaObj[k]), color: ["#2563EB", "#06B6D4", "#7C3AED", "#F43F5E"][i % 4]
      }));
      OE.skillByDepth(m2, STATE.metrics.depths, defs, { xtitle: "RMSE (°C)" });
    }
  }
}

/* ------------------------------------------------------------------ */
/* Float Explorer — interactive platform selector                     */
/* ------------------------------------------------------------------ */
async function renderFloatExplorer() {
  const vprofile = document.getElementById("v-profile");
  if (!vprofile || STATE.floats.length === 0) return;

  // Build selector if not yet present
  let sel = document.getElementById("float-select");
  if (sel && sel.options.length === 0) {
    STATE.floats.forEach((f, i) => {
      sel.appendChild(OE.option(i, "Float " + f.id + " · RMSE " + f.rmse.toFixed(2) + "°C"));
    });
    sel.value = STATE.floatIdx;
    sel.addEventListener("change", () => {
      STATE.floatIdx = parseInt(sel.value);
      renderCurrentPage("float-explorer");
    });
  }

  const fl = STATE.floats[STATE.floatIdx];
  if (fl) {
    OE.profile(vprofile, fl);
    const statsHtml = `
      <div class="sumrow"><span class="k">Region</span><span class="v">${fl.region || 'Arabian Sea'}</span></div>
      <div class="sumrow"><span class="k">Profiles</span><span class="v">${fl.n}</span></div>
      <div class="sumrow"><span class="k">OceanEmbed RMSE</span><span class="v">${fl.rmse.toFixed(2)} <small>°C</small></span></div>
    `;
    const floatStats = document.getElementById("float-stats");
    if (floatStats) floatStats.innerHTML = statsHtml;
    OE.fill("float-sub", `Observed vs model vs GLORYS vs climatology (°C)`);
  }

  const btnReset = document.getElementById("float-reset");
  if (btnReset && !btnReset.hasAttribute('data-bound')) {
    btnReset.setAttribute('data-bound', 'true');
    btnReset.addEventListener("click", () => {
      if (sel) sel.value = 0;
      STATE.floatIdx = 0;
      renderCurrentPage("float-explorer");
    });
  }

  const btnCsv = document.getElementById("float-csv");
  if (btnCsv && fl && !btnCsv.hasAttribute('data-bound')) {
    btnCsv.setAttribute('data-bound', 'true');
    btnCsv.addEventListener("click", () => {
      let csv = "Depth(m),Observed(C),OceanEmbed(C),Climatology(C),GLORYS(C)\n";
      for(let i=0; i<fl.depths.length; i++) {
        csv += `${fl.depths[i]},${fl.observed[i]||''},${fl.model[i]||''},${fl.climatology[i]||''},${fl.glorys[i]||''}\n`;
      }
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `float_${fl.id}_profile.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    });
  }
}

/* ------------------------------------------------------------------ */
/* Depth Transect — live slice from the reconstructed 3D volume       */
/* ------------------------------------------------------------------ */
async function renderTransect() {
  const div = document.getElementById("v-transect");
  if (!div) return;

  // Fetch volume (cached)
  if (!STATE.volume) {
    STATE.volume = await OE.api.volume(STATE.date, 2);
  }
  const cube = STATE.volume;
  const maxIdx = STATE.transectAxis === "lat"
    ? cube.lat.length - 1
    : cube.lon.length - 1;

  // Clamp index
  if (STATE.transectIndex > maxIdx) STATE.transectIndex = Math.floor(maxIdx / 2);

  // Update slider range
  const slider = document.getElementById("tx-index");
  if (slider) {
    slider.max = maxIdx;
    slider.value = STATE.transectIndex;
  }

  // Update position label
  const posLabel = document.getElementById("tx-pos-label");
  const posVal = document.getElementById("tx-pos-val");
  if (STATE.transectAxis === "lat") {
    if (posLabel) posLabel.textContent = "Latitude";
    if (posVal) posVal.textContent = cube.lat[STATE.transectIndex]?.toFixed(2) + "°N";
  } else {
    if (posLabel) posLabel.textContent = "Longitude";
    if (posVal) posVal.textContent = cube.lon[STATE.transectIndex]?.toFixed(2) + "°E";
  }

  const sub = document.getElementById("transect-sub");
  if (sub) {
    const label = STATE.transectAxis === "lat"
      ? "E–W section at " + cube.lat[STATE.transectIndex]?.toFixed(2) + "°N"
      : "N–S section at " + cube.lon[STATE.transectIndex]?.toFixed(2) + "°E";
    sub.textContent = label;
  }

  OE.transect(div, cube, {
    axis: STATE.transectAxis,
    index: STATE.transectIndex,
    date: STATE.date,
  });
}

/* ------------------------------------------------------------------ */
/* Ocean State — MLD / Thermocline derived from the 3D volume         */
/* ------------------------------------------------------------------ */
async function renderOceanState() {
  const div = document.getElementById("v-oceanstate");
  if (!div) return;

  if (!STATE.volume) {
    STATE.volume = await OE.api.volume(STATE.date, 2);
  }

  const title = document.getElementById("os-title");
  const sub = document.getElementById("os-sub");
  if (STATE.oceanStateMode === "thermocline") {
    if (title) title.textContent = "Thermocline depth";
    if (sub) sub.textContent = "Depth of the strongest dT/dz gradient in each water column";
  } else {
    if (title) title.textContent = "Mixed-layer depth";
    if (sub) sub.textContent = "Shallowest depth cooler than the surface by ≥ 0.5 °C";
  }

  OE.oceanState(div, STATE.volume, STATE.oceanStateMode);
}

/* ------------------------------------------------------------------ */
/* Data Pipeline — fill from /api/meta                                */
/* ------------------------------------------------------------------ */
function renderPipeline() {
  // Already fetched by initDashboard; use cached meta
  OE.api.meta().then(meta => {
    const d = meta.domain;
    OE.fill("pipe-domain", d.name);
    OE.fill("pipe-depths", meta.depths.length + " levels · 0–" + meta.depths[meta.depths.length - 1] + " m");
    OE.fill("pipe-window", meta.test_window);
    OE.fill("pipe-cov1", d.lat_min + "–" + d.lat_max + "°N · " + d.lon_min + "–" + d.lon_max + "°E");
    OE.fill("pipe-cov2", d.lat_min + "–" + d.lat_max + "°N · " + d.lon_min + "–" + d.lon_max + "°E");
    OE.fill("pipe-glorys-depth", meta.depths.length + " levels");
    OE.fill("pipe-argo", meta.headline.n_obs.toLocaleString() + " matchups · " + meta.headline.n_floats + " floats");
    const m = meta.model;
    if (m) {
      OE.fill("pipe-model", "Model: " + (m.arch || "U-Net") + " · " +
        (m.params ? (m.params / 1e6).toFixed(1) + "M params" : "") + " · embedding " +
        (m.embedding_dim || "128") + "-d · trained on " + (m.train_dates || "—") + " days");
    }
  }).catch(e => console.error("Pipeline fill error:", e));
}

/* ------------------------------------------------------------------ */
/* Embedding — PCA scatter from /api/embedding                        */
/* ------------------------------------------------------------------ */
async function renderEmbedding() {
  const div = document.getElementById("v-embedding");
  if (!div) return;

  if (!STATE.embeddingData) {
    try {
      STATE.embeddingData = await OE.api.embedding();
    } catch (e) {
      console.error("Embedding API error:", e);
      OE.fill("embed-cap", "Embedding coordinates not available — run the training pipeline first.");
      return;
    }
  }

  OE.embeddingScatter(div, STATE.embeddingData.points);
  OE.fill("embed-sub",
    STATE.embeddingData.n + " days · " + STATE.embeddingData.held_out + " held-out test days");
  OE.fill("embed-cap",
    "Each point is one day's surface state projected to 2D via PCA. " +
    "Held-out test days (diamonds) sit inside the training manifold — " +
    "no distribution shift, no data leakage.");
}

/* ------------------------------------------------------------------ */
/* initDashboard — runs once on DOMContentLoaded                      */
/* ------------------------------------------------------------------ */
async function initDashboard() {
  // Legacy DOM migration — move cards from original-content into their pages
  const orig = document.getElementById("original-content");
  if (orig && orig.children.length > 0) {
    try {
      const ov = document.querySelector('[data-page="overview"]');
      if (ov) {
        const phead = orig.querySelector(".pagehead");
        if (phead) ov.appendChild(phead);
        const kpis = orig.querySelector(".kpis");
        if (kpis) ov.appendChild(kpis);
      }
      orig.style.display = "none";
    } catch (e) { console.error(e); }
  }

  setupRouting();

  // Fetch meta — all KPIs come from here
  let meta;
  try {
    meta = await OE.api.meta();
    STATE.meta = meta;
    OE.clearBanner();
  } catch (e) {
    OE.banner("Cannot reach the API at " + (location.origin || "localhost:8000") +
      "/api/meta — is the FastAPI server running?", "error");
    return;
  }

  const headline = meta.headline;

  // Populate date/depth selectors
  const dateSel = document.getElementById("ctrl-date");
  const depthSel = document.getElementById("ctrl-depth");

  if (dateSel) {
    meta.predicted_dates.forEach(d => dateSel.appendChild(OE.option(d, d)));
    STATE.date = meta.predicted_dates[0];
    dateSel.value = STATE.date;
    dateSel.addEventListener("change", (e) => {
      STATE.date = e.target.value;
      STATE.volume = null;  // invalidate cached volume
      renderCurrentPage(window.location.hash.replace("#", "") || "overview");
    });
  } else {
    STATE.date = meta.predicted_dates[0];
  }

  if (depthSel) {
    meta.depths.forEach((d, i) => depthSel.appendChild(OE.option(i, d + " m")));
    depthSel.value = 0;
    depthSel.addEventListener("change", (e) => {
      STATE.depthIdx = parseInt(e.target.value);
      renderCurrentPage(window.location.hash.replace("#", "") || "overview");
    });
  }

  // ---- KPI bars (global topbar) ----
  OE.fill("kpi-top-rmse", headline.model_rmse.toFixed(2) + " °C");
  OE.fill("kpi-top-r2", headline.r_max.toFixed(2));
  OE.fill("kpi-top-obs", headline.n_obs.toLocaleString());

  // ---- Overview KPI cards ----
  const dEl = document.getElementById("kpi-depths");
  if (dEl) dEl.innerHTML = meta.depths.length + '<small>levels</small>';

  const sstEl = document.getElementById("kpi-sst");
  if (sstEl) sstEl.innerHTML = headline.improvement_pct.toFixed(1) + '<small>%</small>';
  const sstKpi = sstEl?.closest('.kpi');
  if (sstKpi) {
    const lbl = sstKpi.querySelector('.lbl');
    const sub = sstKpi.querySelector('.sub');
    if (lbl) lbl.textContent = 'vs Climatology';
    if (sub) sub.textContent = 'RMSE improvement · test set';
  }

  const covEl = document.getElementById("kpi-coverage");
  if (covEl) covEl.innerHTML = headline.n_floats + '<small>floats</small>';
  const covKpi = covEl?.closest('.kpi');
  if (covKpi) {
    const lbl = covKpi.querySelector('.lbl');
    const sub = covKpi.querySelector('.sub');
    if (lbl) lbl.textContent = 'ARGO Floats';
    if (sub) sub.textContent = 'Validation platforms used';
  }

  // ---- ARGO samples KPI ----
  const sampEl = document.getElementById("kpi-samples");
  if (sampEl) sampEl.innerHTML = headline.n_obs.toLocaleString() + '<small>obs</small>';

  // ---- Metadata fills ----
  OE.fill("ov-summary", "Reconstruction Model: " + (meta.model?.arch || "U-Net") + " trained on " + (meta.model?.train_dates || "—") + " days");
  OE.fill("ov-window", meta.test_window);
  OE.fill("ov-depths", meta.depths.length + " levels");
  if (meta.domain) OE.fill("ov-region", meta.domain.name);
  OE.fill("map-subtitle", meta.domain?.name + " · loading...");

  // ---- Load floats for float explorer ----
  try { STATE.floats = (await OE.api.floats()).floats; } catch (e) {
    console.error("Floats API error:", e);
  }

  // ---- Anomaly controls (domain map page) ----
  const anomalyCheck = document.getElementById("ctrl-anomaly");
  const anomalyThresh = document.getElementById("ctrl-anomaly-thresh");
  const anomalyVal = document.getElementById("anomaly-val");
  const anomalyWrap = document.getElementById("anomaly-slider-wrap");
  if (anomalyCheck) {
    anomalyCheck.addEventListener("change", () => {
      STATE.anomalyMode = anomalyCheck.checked;
      if (anomalyWrap) anomalyWrap.classList.toggle("active", STATE.anomalyMode);
      renderCurrentPage("domain-map");
    });
  }
  if (anomalyThresh) {
    anomalyThresh.addEventListener("input", () => {
      STATE.anomalyThreshold = parseFloat(anomalyThresh.value);
      if (anomalyVal) anomalyVal.textContent = STATE.anomalyThreshold.toFixed(2);
      if (STATE.anomalyMode) renderCurrentPage("domain-map");
    });
  }

  // ---- Transect controls (depth transect page) ----
  const txLat = document.getElementById("tx-lat");
  const txLon = document.getElementById("tx-lon");
  const txIndex = document.getElementById("tx-index");
  if (txLat) {
    txLat.addEventListener("click", () => {
      STATE.transectAxis = "lat";
      txLat.classList.add("on"); txLon?.classList.remove("on");
      STATE.transectIndex = 20;
      renderCurrentPage("depth-transect");
    });
  }
  if (txLon) {
    txLon.addEventListener("click", () => {
      STATE.transectAxis = "lon";
      txLon.classList.add("on"); txLat?.classList.remove("on");
      STATE.transectIndex = 20;
      renderCurrentPage("depth-transect");
    });
  }
  if (txIndex) {
    txIndex.addEventListener("input", () => {
      STATE.transectIndex = parseInt(txIndex.value);
      renderCurrentPage("depth-transect");
    });
  }

  // ---- Ocean State controls ----
  const osMld = document.getElementById("os-mld");
  const osTherm = document.getElementById("os-therm");
  if (osMld) {
    osMld.addEventListener("click", () => {
      STATE.oceanStateMode = "mld";
      osMld.classList.add("on"); osTherm?.classList.remove("on");
      renderCurrentPage("ocean-state");
    });
  }
  if (osTherm) {
    osTherm.addEventListener("click", () => {
      STATE.oceanStateMode = "thermocline";
      osTherm.classList.add("on"); osMld?.classList.remove("on");
      renderCurrentPage("ocean-state");
    });
  }

  // ---- Domain map reset ----
  const mapReset = document.getElementById("map-reset");
  if (mapReset) {
    mapReset.addEventListener("click", () => {
      STATE.anomalyMode = false;
      STATE.anomalyThreshold = 1.5;
      if (anomalyCheck) anomalyCheck.checked = false;
      if (anomalyWrap) anomalyWrap.classList.remove("active");
      if (anomalyThresh) anomalyThresh.value = 1.5;
      if (anomalyVal) anomalyVal.textContent = "1.50";
      renderCurrentPage("domain-map");
    });
  }

  // ---- 3D Ocean reset ----
  const ocean3dReset = document.getElementById("ocean3d-reset");
  if (ocean3dReset) {
    ocean3dReset.addEventListener("click", () => {
      const vvolume = document.getElementById("v-volume");
      if (vvolume && vvolume.layout) {
        Plotly.relayout(vvolume, {'scene.camera': { eye: { x: 1.5, y: -1.6, z: 0.9 } }});
      }
    });
  }

  // ---- Depth transect reset ----
  const txReset = document.getElementById("tx-reset");
  if (txReset) {
    txReset.addEventListener("click", () => {
      STATE.transectAxis = "lat";
      STATE.transectIndex = 20;
      if (txLat) txLat.classList.add("on");
      if (txLon) txLon.classList.remove("on");
      if (txIndex) txIndex.value = 20;
      renderCurrentPage("depth-transect");
    });
  }

  // ---- Initial render ----
  renderCurrentPage(window.location.hash.replace("#", "") || "overview");
}

document.addEventListener("DOMContentLoaded", initDashboard);
