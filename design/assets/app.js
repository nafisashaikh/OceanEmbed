/* ============================================================
   OceanEmbed — app shell: data load, routing, section renderers.
   Every figure is read from data/*.json (and live /api where the
   backend is running). Nothing on this page is hard-coded.
   ============================================================ */
(async function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const fmt = (n, d = 2) => (n == null || Number.isNaN(+n) ? "—" : (+n).toFixed(d));
  const col = (arr, key) => arr.map(r => r[key]);

  const LABELS = {
    overview: "Overview", validation: "Validation & skill", floats: "Float explorer",
    map: "Domain map", ocean3d: "3D ocean", transect: "Depth transect", pipeline: "Data & pipeline",
  };
  const REGCOLOR = { "Arabian Sea": "#2563EB", "Bay of Bengal": "#06B6D4" };

  let meta, metrics, field, floats;
  try {
    [meta, metrics, field, floats] = await Promise.all([
      fetch("data/meta.json").then(r => r.json()),
      fetch("data/metrics.json").then(r => r.json()),
      fetch("data/field.json").then(r => r.json()),
      fetch("data/floats.json").then(r => r.json()).then(d => d.floats || d),
    ]);
  } catch (e) {
    console.error("Data load failed", e);
    $("#content").insertAdjacentHTML("afterbegin",
      '<p style="padding:24px;color:#b91c1c">Could not load data/*.json — run <code>python design/build_data.py</code> first.</p>');
    return;
  }
  const h = meta.headline, D = meta.depths;

  // per-float RMSE from real observed-vs-model pairs (if not already provided)
  const floatRmse = f => {
    let se = 0, k = 0;
    f.observed.forEach((o, i) => { const m = f.model[i]; if (o != null && m != null) { se += (m - o) ** 2; k++; } });
    return k ? Math.sqrt(se / k) : null;
  };
  floats.forEach(f => { if (f.rmse == null) f.rmse = floatRmse(f); });

  // ---------- shared tiny builders ----------
  const secHead = (t, p) => `<div class="sec"><h2>${t}</h2>${p ? `<p>${p}</p>` : ""}</div>`;
  const mini = items => `<div class="mini">${items.map(i =>
    `<div class="m"><div class="mv${i.good ? " good" : ""}"${i.id ? ` id="${i.id}"` : ""}>${i.v}${i.s ? `<small>${i.s}</small>` : ""}</div><div class="mk">${i.k}</div></div>`).join("")}</div>`;
  const readout = items => `<div class="readout">${items.map(i =>
    `<div class="r"><div class="rv">${i.v}</div><div class="rk">${i.k}</div></div>`).join("")}</div>`;
  const setTxt = (id, v) => { const el = $("#" + id); if (el) el.textContent = v; };
  const trow = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;

  // ============================================================
  // OVERVIEW
  // ============================================================
  $("#m-region").textContent = meta.domain.name;
  $("#m-window").textContent = meta.test_window;
  $("#m-depths").textContent = `${D.length} levels · 0–1000 m`;
  $("#hero-rmse").textContent = fmt(h.model_rmse);
  $("#hero-impr").textContent = h.improvement_pct + "%";
  $("#foot-note").textContent =
    `Figures computed from processed_data/predicted_field.zarr and argo_validation_summary.json · representative day ${field.date}.`;
  setTxt("ov-rmse-badge", `${h.depths_beat} / ${h.depths_total} win`);
  const ovStory = $("#ov-story-result");
  if (ovStory) ovStory.innerHTML =
    `${fmt(h.model_rmse)}&nbsp;°C RMSE against real ARGO floats — beating the baseline at ` +
    `${h.depths_beat} of ${h.depths_total} depth levels, and within ${fmt(h.ceiling_gap)}&nbsp;°C of the GLORYS reanalysis ceiling.`;

  const kpis = [
    { v: fmt(h.model_rmse), s: " °C", k: "RMSE vs. real ARGO floats", good: true },
    { v: h.improvement_pct, s: "%", k: "Better than climatology baseline", tag: "vs. baseline" },
    { v: `${h.depths_beat}/${h.depths_total}`, k: "Depth levels beat the baseline" },
    { v: h.n_obs.toLocaleString(), k: `Matched ARGO observations · ${h.n_floats} floats` },
    { v: fmt(h.ceiling_gap), s: " °C", k: "Gap to the GLORYS reanalysis ceiling", tag: "near ceiling" },
    { v: fmt(h.r_max), k: "Peak correlation (near surface)" },
  ];
  $("#kpis").innerHTML = kpis.map(c => `
    <div class="kpi${c.good ? " good" : ""}">
      <div class="v">${c.v}${c.s ? `<small>${c.s}</small>` : ""}</div>
      <div class="k">${c.k}</div>
      ${c.tag ? `<span class="tag">${c.tag}</span>` : ""}
    </div>`).join("");
  const depthSel = $("#map-depth");
  depthSel.innerHTML = D.map((d, i) => `<option value="${i}">${d} m</option>`).join("");
  const drawMap = () => {
    const di = +depthSel.value;
    OE.heatmap("map-chart", field, di);
    $("#map-sub").textContent = `Predicted field · ${field.date} · ${D[di]} m depth`;
  };
  depthSel.addEventListener("change", drawMap);
  const volIdx = [0, 30, 100, 300, 1000].map(w => D.indexOf(w)).filter(i => i >= 0);
  let drewOverview = false;
  function renderOverview() {
    if (drewOverview) return;
    drawMap();
    OE.rmseByDepth("rmse-chart", metrics);
    OE.volume("vol-chart", field, volIdx);
    drewOverview = true;
  }

  // ============================================================
  // VALIDATION & SKILL
  // ============================================================
  let _scatter;
  async function loadScatter() {
    if (_scatter) return _scatter;
    try {
      const r = await fetch("/api/scatter");
      if (!r.ok) throw 0;
      const s = await r.json();
      if (!s.observed || !s.observed.length) throw 0;
      s._source = "api";
      return (_scatter = s);
    } catch {
      const observed = [], model = [], depth = [];
      floats.forEach(f => f.depths.forEach((dd, i) => {
        const o = f.observed[i], m = f.model[i];
        if (o != null && m != null) { observed.push(o); model.push(m); depth.push(dd); }
      }));
      const n = observed.length; let se = 0, ae = 0, be = 0;
      const mo = observed.reduce((a, b) => a + b, 0) / n, mm = model.reduce((a, b) => a + b, 0) / n;
      let cov = 0, vo = 0, vm = 0;
      for (let i = 0; i < n; i++) {
        const e = model[i] - observed[i]; se += e * e; ae += Math.abs(e); be += e;
        const a = observed[i] - mo, b = model[i] - mm; cov += a * b; vo += a * a; vm += b * b;
      }
      const r = vo > 0 && vm > 0 ? cov / Math.sqrt(vo * vm) : NaN;
      const all = observed.concat(model);
      _scatter = {
        observed, model, depth, n, _source: "floats",
        rmse: Math.sqrt(se / n), mae: ae / n, bias: be / n, r, r2: r * r,
        tmin: Math.min.apply(null, all), tmax: Math.max.apply(null, all),
      };
      return _scatter;
    }
  }
  function skillTable() {
    const rows = D.map((d, i) => {
      const m = metrics.model[i], c = metrics.climatology[i], g = metrics.glorys[i];
      const win = m.rmse != null && c.rmse != null && m.rmse < c.rmse;
      return `<tr><td>${d} m</td><td>${m.n ?? "—"}</td>` +
        `<td class="${win ? "win" : "lose"}">${fmt(m.rmse)}</td>` +
        `<td>${fmt(c.rmse)}</td><td>${fmt(g.rmse)}</td><td>${fmt(m.mae)}</td>` +
        `<td>${m.bias == null ? "—" : (m.bias >= 0 ? "+" : "") + fmt(m.bias)}</td>` +
        `<td>${fmt(m.corr)}</td></tr>`;
    }).join("");
    return `<table class="tbl"><thead><tr><th>Depth</th><th>n</th><th>Model RMSE</th>` +
      `<th>Clim RMSE</th><th>GLORYS RMSE</th><th>Model MAE</th><th>Bias</th><th>r</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  }

  async function renderValidation(sec) {
    sec.innerHTML = `<div class="wrap">
      ${secHead("Validation &amp; skill", "Per-depth skill against real ARGO floats — model vs. the climatology baseline and the GLORYS reanalysis ceiling, with the paired t-test.")}
      ${mini([
        { v: fmt(h.model_rmse), s: " °C", k: "Model RMSE", good: true },
        { v: fmt(h.clim_rmse), s: " °C", k: "Climatology RMSE" },
        { v: fmt(h.glorys_rmse), s: " °C", k: "GLORYS ceiling" },
        { v: h.improvement_pct + "%", k: "Better than baseline" },
        { v: h.t_stat, k: "Paired t-statistic" },
        { v: h.p_value < 1e-6 ? "<1e-6" : fmt(h.p_value, 3), k: `p-value · n=${h.n_obs.toLocaleString()}` },
      ])}
      <div class="grid g-2">
        <div class="card pad">
          <div class="ch"><div><h3>RMSE by depth</h3><p>Lower is better — model beats climatology at ${h.depths_beat}/${h.depths_total} levels</p></div></div>
          <div class="chart" id="val-rmse" style="height:400px"></div>
        </div>
        <div class="card pad">
          <div class="ch"><div><h3>Correlation by depth</h3><p>Model vs. baseline vs. ceiling</p></div><span class="badge2">r up to ${fmt(h.r_max)}</span></div>
          <div class="chart" id="val-corr" style="height:400px"></div>
        </div>
      </div>
      <div class="grid g-2" style="margin-top:16px">
        <div class="card pad">
          <div class="ch"><div><h3>Model RMSE by region</h3><p>Arabian Sea vs. Bay of Bengal</p></div></div>
          <div class="chart" id="val-region" style="height:360px"></div>
        </div>
        <div class="card pad">
          <div class="ch"><div><h3>Model bias by depth</h3><p>Warm (+) or cold (−) tendency vs. ARGO</p></div></div>
          <div class="chart" id="val-bias" style="height:360px"></div>
        </div>
      </div>
      <div class="card pad" style="margin-top:16px">
        <div class="ch"><div><h3>Observed vs. model parity</h3><p id="val-parity-sub">Matched ARGO casts — color encodes depth</p></div></div>
        <div class="chart" id="val-parity" style="height:440px"></div>
        <div id="val-parity-read"></div>
      </div>
      <div class="card pad" style="margin-top:16px">
        <div class="ch"><div><h3>Per-depth skill table</h3><p>Green where the model beats the climatology baseline</p></div></div>
        <div class="tbl-wrap" id="val-table"></div>
      </div>
      <div class="honest"><b>Leak-free validation:</b> ARGO floats are held out from training entirely. Skill is measured against real in-situ casts — GLORYS is shown only as a practical ceiling, not ground truth.</div>
    </div>`;
    OE.skillByDepth("val-rmse", D, [
      { name: "Climatology", vals: col(metrics.climatology, "rmse"), color: "#94A3B8", dash: "dot" },
      { name: "GLORYS ceiling", vals: col(metrics.glorys, "rmse"), color: "#7C3AED", dash: "dash" },
      { name: "OceanEmbed", vals: col(metrics.model, "rmse"), color: "#2563EB", width: 3 },
    ], { xtitle: "RMSE (°C)" });
    OE.skillByDepth("val-corr", D, [
      { name: "Climatology", vals: col(metrics.climatology, "corr"), color: "#94A3B8", dash: "dot" },
      { name: "GLORYS ceiling", vals: col(metrics.glorys, "corr"), color: "#7C3AED", dash: "dash" },
      { name: "OceanEmbed", vals: col(metrics.model, "corr"), color: "#2563EB", width: 3 },
    ], { xtitle: "Correlation (r)" });
    OE.skillByDepth("val-region", D, [
      { name: "Arabian Sea", vals: col(metrics.region.arabian_sea, "rmse"), color: "#2563EB", width: 3 },
      { name: "Bay of Bengal", vals: col(metrics.region.bay_of_bengal, "rmse"), color: "#06B6D4", width: 3 },
    ], { xtitle: "Model RMSE (°C)" });
    OE.skillByDepth("val-bias", D, [
      { name: "Model bias", vals: col(metrics.model, "bias"), color: "#2563EB", width: 3 },
    ], { xtitle: "Bias (°C)" });

    $("#val-table").innerHTML = skillTable();

    const s = await loadScatter();
    OE.parity("val-parity", s);
    setTxt("val-parity-sub", s._source === "api"
      ? `Every one of ${s.n.toLocaleString()} matched ARGO casts — color encodes depth`
      : "Matched ARGO casts from float profiles — color encodes depth");
    $("#val-parity-read").innerHTML = readout([
      { v: (s.n || s.observed.length).toLocaleString(), k: "matched casts" },
      { v: fmt(s.rmse) + " °C", k: "RMSE" },
      { v: fmt(s.mae) + " °C", k: "MAE" },
      { v: (s.bias >= 0 ? "+" : "") + fmt(s.bias) + " °C", k: "bias" },
      { v: fmt(s.r2, 3), k: "R²" },
    ]);
  /* __APPEND_4__ */
  }
  // ============================================================
  // FLOAT EXPLORER
  // ============================================================
  function renderFloats(sec) {
    sec.innerHTML = `<div class="wrap">
      ${secHead("Float explorer", `All ${floats.length} real ARGO floats used for validation — click a float to compare its measured profile against the model, GLORYS and climatology.`)}
      <div class="grid g-2">
        <div class="card pad">
          <div class="ch"><div><h3>Float positions</h3><p>Arabian Sea &amp; Bay of Bengal — click to select</p></div><span class="badge2">${floats.length} floats</span></div>
          <div class="chart" id="fl-map" style="height:320px"></div>
          <div class="flist" id="fl-list"></div>
        </div>
        <div class="card pad">
          <div class="ch"><div><h3 id="fl-title">Vertical profile</h3><p id="fl-sub">—</p></div></div>
          <div class="chart" id="fl-profile" style="height:420px"></div>
          <div id="fl-read"></div>
        </div>
      </div>
    </div>`;
    let activeId = floats[0].id;
    const byId = id => floats.find(f => f.id === id);
    const listHtml = () => floats.map(f => `
      <div class="flrow${f.id === activeId ? " on" : ""}" data-id="${f.id}">
        <span class="dot" style="background:${REGCOLOR[f.region] || "#2563EB"}"></span>
        <span class="fid">${f.id}</span>
        <span class="fmeta">${f.region}<br>${f.n} profiles · ${fmt(f.rmse)} °C</span>
      </div>`).join("");
    const drawProfile = () => {
      const f = byId(activeId);
      OE.profile("fl-profile", f);
      setTxt("fl-title", `Float ${f.id}`);
      setTxt("fl-sub", `${f.region} · ${f.n} profiles · ${f.depths.length} depth levels`);
      $("#fl-read").innerHTML = readout([
        { v: fmt(f.rmse) + " °C", k: "float RMSE" },
        { v: f.lat.toFixed(2) + "°N", k: "latitude" },
        { v: f.lon.toFixed(2) + "°E", k: "longitude" },
        { v: f.n, k: "profiles" },
      ]);
    };
    const select = id => {
      if (!byId(id)) return;
      activeId = id;
      $("#fl-list").innerHTML = listHtml();
      OE.floatsMap("fl-map", floats, activeId);
      drawProfile();
    };
    $("#fl-list").innerHTML = listHtml();
    OE.floatsMap("fl-map", floats, activeId);
    drawProfile();
    $("#fl-list").addEventListener("click", e => {
      const row = e.target.closest(".flrow[data-id]"); if (row) select(row.dataset.id);
    });
    const md = $("#fl-map");
    if (md && md.on) md.on("plotly_click", ev => {
      const id = ev.points && ev.points[0] && ev.points[0].customdata;
      if (id != null) select(String(id));
    });
  }
  // ============================================================
  // DOMAIN MAP
  // ============================================================
  function layerStats(grid) {
    let mn = Infinity, mx = -Infinity, sum = 0, n = 0;
    for (const rowv of grid) for (const v of rowv) if (v != null) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; n++; }
    return { min: n ? mn : null, max: n ? mx : null, mean: n ? sum / n : null, n };
  }
  function renderMap(sec) {
    const res = field.lat.length > 1 ? Math.abs(field.lat[1] - field.lat[0]).toFixed(2) : "0.25";
    sec.innerHTML = `<div class="wrap">
      ${secHead("Domain temperature map", `Reconstructed 0–1000 m temperature for ${field.date} across ${meta.domain.name} — choose a depth to slice the field.`)}
      <div class="card pad">
        <div class="ch"><div><h3>Predicted field</h3><p id="dm-sub">—</p></div>
          <span class="ctrl"><label>Depth</label><select id="dm-depth"></select></span></div>
        <div class="chart" id="dm-chart" style="height:520px"></div>
        <div id="dm-read"></div>
      </div>
      ${mini([
        { v: `${meta.domain.lat[0]}–${meta.domain.lat[1]}°N`, k: "Latitude range" },
        { v: `${meta.domain.lon[0]}–${meta.domain.lon[1]}°E`, k: "Longitude range" },
        { v: `${field.lat.length}×${field.lon.length}`, k: "Grid (lat×lon)" },
        { v: D.length, k: "Depth levels" },
        { v: `${res}°`, k: "Resolution" },
        { v: field.date, k: "Representative day" },
      ])}
    </div>`;
    const sel = $("#dm-depth");
    sel.innerHTML = D.map((d, i) => `<option value="${i}">${d} m</option>`).join("");
    const draw = () => {
      const di = +sel.value;
      OE.heatmap("dm-chart", field, di);
      const st = layerStats(field.temp[di]);
      setTxt("dm-sub", `${field.date} · ${D[di]} m depth`);
      $("#dm-read").innerHTML = readout([
        { v: fmt(st.min) + " °C", k: "minimum" },
        { v: fmt(st.mean) + " °C", k: "mean" },
        { v: fmt(st.max) + " °C", k: "maximum" },
        { v: st.n.toLocaleString(), k: "ocean cells" },
      ]);
    };
    sel.addEventListener("change", draw);
    draw();
  }
  // ============================================================
  // 3D OCEAN
  // ============================================================
  function renderOcean3d(sec) {
    const keyIdx = [0, 30, 100, 300, 1000].map(w => D.indexOf(w)).filter(i => i >= 0);
    const allIdx = D.map((_, i) => i);
    sec.innerHTML = `<div class="wrap">
      ${secHead("3D ocean", "The reconstructed temperature volume as stacked depth layers on one shared color scale — drag to rotate, scroll to zoom, hover for values.")}
      <div class="card pad">
        <div class="ch"><div><h3>Reconstructed volume</h3><p id="o3-sub">—</p></div>
          <span class="seg" id="o3-toggle"><button class="on" data-mode="key">Key layers</button><button data-mode="all">All ${D.length}</button></span></div>
        <div class="chart" id="o3-chart" style="height:600px"></div>
        <div id="o3-read"></div>
      </div>
    </div>`;
    const draw = mode => {
      const idx = mode === "all" ? allIdx : keyIdx;
      OE.volume("o3-chart", field, idx);
      setTxt("o3-sub", mode === "all"
        ? `All ${idx.length} depth levels`
        : `${idx.length} representative layers · ${idx.map(i => D[i] + " m").join(", ")}`);
      let mn = Infinity, mx = -Infinity;
      idx.forEach(i => { const st = layerStats(field.temp[i]); if (st.min != null) { mn = Math.min(mn, st.min); mx = Math.max(mx, st.max); } });
      $("#o3-read").innerHTML = readout([
        { v: idx.length, k: "layers shown" },
        { v: `${D[idx[0]]}–${D[idx[idx.length - 1]]} m`, k: "depth span" },
        { v: `${fmt(mn)}–${fmt(mx)} °C`, k: "temperature range" },
        { v: field.date, k: "date" },
      ]);
    };
    $("#o3-toggle").addEventListener("click", e => {
      const b = e.target.closest("button[data-mode]"); if (!b) return;
      $$("#o3-toggle button").forEach(x => x.classList.toggle("on", x === b));
      draw(b.dataset.mode);
    });
    draw("key");
  }

  // ============================================================
  // DEPTH TRANSECT
  // ============================================================
  function renderTransect(sec) {
    const mid = Math.floor(field.lat.length / 2);
    sec.innerHTML = `<div class="wrap">
      ${secHead("Depth transect", "A vertical temperature section along a chosen latitude — the thermocline in cross-section, longitude × depth.")}
      <div class="card pad">
        <div class="ch"><div><h3>Longitude × depth section</h3><p id="tr-sub">—</p></div>
          <span class="ctrl"><label>Latitude</label><input type="range" id="tr-lat" min="0" max="${field.lat.length - 1}" value="${mid}" style="width:160px"><b id="tr-latval">—</b></span></div>
        <div class="chart" id="tr-chart" style="height:480px"></div>
        <div id="tr-read"></div>
      </div>
    </div>`;
    const rng = $("#tr-lat");
    const draw = () => {
      const li = +rng.value;
      OE.transect("tr-chart", field, li);
      const lat = field.lat[li];
      setTxt("tr-latval", `${lat.toFixed(2)}°N`);
      setTxt("tr-sub", `Latitude ${lat.toFixed(2)}°N · longitude ${field.lon[0]}–${field.lon[field.lon.length - 1]}°E`);
      const surf = layerStats([field.temp[0][li]]);
      const di = D.length - 1, deep = layerStats([field.temp[di][li]]);
      $("#tr-read").innerHTML = readout([
        { v: `${lat.toFixed(2)}°N`, k: "latitude" },
        { v: fmt(surf.mean) + " °C", k: `surface (${D[0]} m)` },
        { v: fmt(deep.mean) + " °C", k: `deep (${D[di]} m)` },
      ]);
    };
    rng.addEventListener("input", draw);
    rng.addEventListener("change", draw);
    draw();
  }
  // ============================================================
  // DATA & PIPELINE
  // ============================================================
  function renderPipeline(sec) {
    const dom = meta.domain || {};
    const lat = dom.lat || [5, 30], lon = dom.lon || [45, 105];
    sec.innerHTML = `<div class="wrap">
      ${secHead("Data & pipeline", "How daily satellite surface observations become a full 0–1000 m temperature field — inputs, model, and validation, end to end.")}
      <div class="card pad">
        <div class="ch"><div><h3>Reconstruction pipeline</h3><p>Surface → subsurface, trained on reanalysis, validated on real floats</p></div><span class="badge2">end-to-end</span></div>
        <div class="pipe">
          <div class="pcol">
            <div class="pnode accent"><div class="pt">Satellite surface</div><div class="pd" id="pp-inputs">SST · SSH · SSS · winds · currents</div></div>
          </div>
          <div class="parrow">→</div>
          <div class="pcol">
            <div class="pnode dark"><div class="pt">U-Net CNN</div><div class="pd" id="pp-model">Encoder–decoder · <span id="pp-params">—</span> params · <span id="pp-ch">—</span> channels</div></div>
          </div>
          <div class="parrow">→</div>
          <div class="pcol">
            <div class="pnode accent"><div class="pt">15 depth levels</div><div class="pd">0–1000 m temperature field, daily</div></div>
          </div>
          <div class="parrow">→</div>
          <div class="pcol">
            <div class="pnode"><div class="pt">ARGO validation</div><div class="pd">${h.n_obs.toLocaleString()} in-situ obs · ${h.n_floats} floats · leak-free</div></div>
          </div>
        </div>
        <div class="tags">
          <span class="tag2">GLORYS reanalysis (training truth)</span>
          <span class="tag2">Real ARGO floats (independent test)</span>
          <span class="tag2">Climatology (baseline)</span>
        </div>
      </div>

      <div class="mini">${[
        { v: `${lat[0]}–${lat[1]}°N`, k: "Latitude range" },
        { v: `${lon[0]}–${lon[1]}°E`, k: "Longitude range" },
        { v: D.length, k: "Depth levels · 0–1000 m" },
        { v: h.n_obs.toLocaleString(), k: "ARGO observations" },
        { v: h.n_floats, k: "Independent floats" },
        { v: fmt(h.model_rmse), s: " °C", k: "RMSE vs ARGO", good: true },
      ].map(i => `<div class="m"><div class="mv${i.good ? " good" : ""}">${i.v}${i.s ? `<small>${i.s}</small>` : ""}</div><div class="mk">${i.k}</div></div>`).join("")}</div>

      <div class="grid g-2">
        <div class="card pad">
          <div class="ch"><div><h3>Domain &amp; coverage</h3><p>Where the model runs and what it resolves</p></div></div>
          <div class="tbl-wrap"><table class="tbl"><tbody id="pp-domain"></tbody></table></div>
        </div>
        <div class="card pad">
          <div class="ch"><div><h3>Model configuration</h3><p id="pp-cfg-sub">Architecture &amp; training</p></div></div>
          <div class="tbl-wrap"><table class="tbl"><tbody id="pp-config"></tbody></table></div>
        </div>
      </div>
      <div class="honest"><b>Honest positioning:</b> surface→subsurface reconstruction is established in the literature. Our contribution is adapting it to the <b>Indian Ocean</b> with strict leak-free validation against real ARGO floats and an end-to-end, deployable pipeline.</div>
    </div>`;

    const domRows = [
      ["Region", meta.domain?.name || "Arabian Sea + Bay of Bengal"],
      ["Latitude", `${lat[0]}° – ${lat[1]}° N`],
      ["Longitude", `${lon[0]}° – ${lon[1]}° E`],
      ["Grid resolution", (field.lat.length > 1 ? Math.abs(field.lat[1] - field.lat[0]).toFixed(2) : "0.25") + "°"],
      ["Grid size", `${field.lat.length} × ${field.lon.length}`],
      ["Depth levels", `${D.length}  (${D[0]}–${D[D.length - 1]} m)`],
      ["Test window", meta.test_window || "Dec 2023 – Jan 2024"],
    ];
    $("#pp-domain").innerHTML = domRows.map(trow).join("");

    // baseline config from static meta; enriched live from /api/meta when available
    const cfgRows = [
      ["Architecture", "U-Net encoder–decoder (CNN)"],
      ["Output depths", `${D.length} levels`],
      ["Training truth", "GLORYS12 reanalysis"],
      ["Validation", `Real ARGO floats (${h.n_floats})`],
    ];
    $("#pp-config").innerHTML = cfgRows.map(trow).join("");

    fetch("/api/meta").then(r => r.ok ? r.json() : null).then(j => {
      const m = j && j.model; if (!m) return;
      if (m.parameters != null) setTxt("pp-params", (m.parameters / 1e6).toFixed(2) + "M");
      if (m.input_channels != null) setTxt("pp-ch", `${m.input_channels}→${m.output_channels}`);
      if (Array.isArray(m.input_variables) && m.input_variables.length)
        setTxt("pp-inputs", m.input_variables.join(" · "));
      const extra = [];
      if (m.parameters != null) extra.push(["Parameters", (m.parameters / 1e6).toFixed(2) + " M"]);
      if (m.input_channels != null) extra.push(["Input channels", m.input_channels]);
      if (m.output_channels != null) extra.push(["Output channels", m.output_channels]);
      if (m.embedding_dim != null) extra.push(["Embedding dim", m.embedding_dim]);
      if (m.base_channels != null) extra.push(["Base channels", m.base_channels]);
      if (m.output_size) extra.push(["Output size", Array.isArray(m.output_size) ? m.output_size.join(" × ") : m.output_size]);
      if (m.lag_days != null) extra.push(["Lag (days)", m.lag_days]);
      if (m.epoch != null) extra.push(["Checkpoint epoch", m.epoch]);
      if (extra.length) {
        $("#pp-config").innerHTML = cfgRows.concat(extra).map(trow).join("");
        setTxt("pp-cfg-sub", "Architecture & training · live from model checkpoint");
      }
    }).catch(() => {});
  }

  // ============================================================
  // ROUTING
  // ============================================================
  const RENDER = {
    validation: renderValidation,
    floats: renderFloats,
    map: renderMap,
    ocean3d: renderOcean3d,
    transect: renderTransect,
    pipeline: renderPipeline,
  };

  function go(route) {
    route = route || "overview";
    const nav = $$("#nav a[data-route]");
    nav.forEach(a => a.classList.toggle("active", a.dataset.route === route));
    $$(".page").forEach(p => p.classList.toggle("show", p.dataset.page === route));
    const active = nav.find(a => a.dataset.route === route);
    setTxt("crumb", active ? active.textContent.trim() : "Overview");

    if (route === "overview") { renderOverview(); }
    else {
      const sec = $(`.page[data-page="${route}"]`);
      if (sec && RENDER[route] && sec.dataset.filled !== "1") {
        try { RENDER[route](sec); sec.dataset.filled = "1"; }
        catch (err) {
          console.error("render error:", route, err);
          sec.innerHTML = `<div class="wrap"><div class="honest" style="background:#fef2f2;border-color:#fecaca;color:#991b1b">Could not render “${route}”: ${err.message}</div></div>`;
        }
      }
    }
    closeSidebar();
    $("#content").scrollTo({ top: 0 });
    window.scrollTo({ top: 0 });
  }

  function closeSidebar() {
    $("#sidebar")?.classList.remove("open");
    $("#scrim")?.classList.remove("show");
  }

  $("#nav")?.addEventListener("click", e => {
    const a = e.target.closest("a[data-route]"); if (!a) return;
    e.preventDefault();
    const r = a.dataset.route;
    if (location.hash.slice(2) === r) go(r);
    else location.hash = "/" + r;
  });
  window.addEventListener("hashchange", () => go(location.hash.slice(2) || "overview"));
  $("#burger")?.addEventListener("click", () => {
    $("#sidebar")?.classList.toggle("open");
    $("#scrim")?.classList.toggle("show");
  });
  $("#scrim")?.addEventListener("click", closeSidebar);

  go(location.hash.slice(2) || "overview");
})();
