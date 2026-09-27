/* ============================================================
   OceanEmbed — app shell: data load, routing, wiring
   ============================================================ */
(async function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  const LABELS = {
    overview: "Overview", validation: "Validation & skill", floats: "Float explorer",
    map: "Domain map", ocean3d: "3D ocean", transect: "Depth transect", pipeline: "Data & pipeline",
  };

  let meta, metrics, field;
  try {
    [meta, metrics, field] = await Promise.all([
      fetch("data/meta.json").then(r => r.json()),
      fetch("data/metrics.json").then(r => r.json()),
      fetch("data/field.json").then(r => r.json()),
    ]);
  } catch (e) {
    console.error("Data load failed", e);
    $("#content").insertAdjacentHTML("afterbegin",
      '<p style="padding:24px;color:#b91c1c">Could not load data/*.json — run <code>python design/build_data.py</code> first.</p>');
    return;
  }

  // ---- hero + meta ----
  const h = meta.headline;
  $("#m-region").textContent = meta.domain.name;
  $("#m-window").textContent = meta.test_window;
  $("#m-depths").textContent = `${meta.depths.length} levels · 0–1000 m`;
  $("#hero-rmse").textContent = h.model_rmse.toFixed(2);
  $("#hero-impr").textContent = h.improvement_pct + "%";
  $("#foot-note").textContent =
    `Figures computed from processed_data/predicted_field.zarr and argo_validation_summary.json · representative day ${field.date}.`;

  // ---- KPI strip ----
  const kpis = [
    { v: h.model_rmse.toFixed(2), s: " °C", k: "RMSE vs. real ARGO floats", good: true },
    { v: h.improvement_pct, s: "%", k: "Better than climatology baseline", tag: "vs. baseline" },
    { v: `${h.depths_beat}/${h.depths_total}`, k: "Depth levels beat the baseline" },
    { v: h.n_obs.toLocaleString(), k: `Matched ARGO observations · ${h.n_floats} floats` },
    { v: h.ceiling_gap.toFixed(2), s: " °C", k: "Gap to the GLORYS reanalysis ceiling", tag: "near ceiling" },
    { v: h.r_max.toFixed(2), k: "Peak correlation (near surface)" },
  ];
  $("#kpis").innerHTML = kpis.map(c => `
    <div class="kpi${c.good ? " good" : ""}">
      <div class="v">${c.v}${c.s ? `<small>${c.s}</small>` : ""}</div>
      <div class="k">${c.k}</div>
      ${c.tag ? `<span class="tag">${c.tag}</span>` : ""}
    </div>`).join("");

  // ---- overview charts ----
  const depthSel = $("#map-depth");
  depthSel.innerHTML = meta.depths.map((d, i) => `<option value="${i}">${d} m</option>`).join("");
  const drawMap = () => {
    const di = +depthSel.value;
    OE.heatmap("map-chart", field, di);
    $("#map-sub").textContent = `Predicted field · ${field.date} · ${meta.depths[di]} m depth`;
  };
  depthSel.addEventListener("change", drawMap);

  // pick a readable spread of layers for the 3D teaser
  const want = [0, 30, 100, 300, 1000];
  const volIdx = want.map(w => meta.depths.indexOf(w)).filter(i => i >= 0);

  let drewOverview = false;
  function renderOverview() {
    if (drewOverview) return;
    drawMap();
    OE.rmseByDepth("rmse-chart", metrics);
    OE.volume("vol-chart", field, volIdx);
    drewOverview = true;
  }
  // ---- stub content for pages still to come ----
  const STUB_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/></svg>';
  const STUB_COPY = {
    validation: "Per-depth RMSE, MAE, bias and correlation for model vs. climatology vs. the GLORYS ceiling, with the paired t-test and region / season breakdowns.",
    floats: `Explore all ${h.n_floats} real ARGO floats — click a float to see its measured profile against the model, GLORYS and climatology, depth by depth.`,
    map: "The full predicted temperature field animated across the test window, with a depth slider and play control.",
    ocean3d: "The complete 0–1000 m reconstructed volume as an interactive, rotatable 3D field.",
    transect: "Vertical temperature sections along a chosen latitude or longitude, revealing the thermocline structure.",
    pipeline: "The end-to-end pipeline: six satellite surface inputs → CNN encoder–decoder → 15 depth levels, trained on GLORYS, validated on real ARGO.",
  };
  function fillStub(page) {
    const sec = $(`.page[data-page="${page}"]`);
    if (!sec || sec.dataset.filled) return;
    sec.innerHTML = `<div class="wrap"><div class="stub">${STUB_ICON}
      <h2>${LABELS[page]}</h2>
      <p>${STUB_COPY[page] || ""}</p>
      <p style="margin-top:14px;font-size:12px;color:var(--faint)">Coming next — the Overview page is the fully wired flagship.</p>
    </div></div>`;
    sec.dataset.filled = "1";
  }

  // ---- hash routing ----
  function go(route) {
    if (!LABELS[route]) route = "overview";
    $$(".page").forEach(p => p.classList.toggle("show", p.dataset.page === route));
    $$("#nav a").forEach(a => a.classList.toggle("active", a.dataset.route === route));
    $("#crumb").textContent = LABELS[route];
    if (route === "overview") renderOverview(); else fillStub(route);
    $("#content").scrollTop = 0;
    closeSidebar();
  }
  window.addEventListener("hashchange", () => go(location.hash.slice(2)));

  $("#nav").addEventListener("click", e => {
    const a = e.target.closest("a[data-route]");
    if (!a) return;
    e.preventDefault();
    location.hash = "/" + a.dataset.route;
  });

  // ---- mobile sidebar ----
  const sidebar = $("#sidebar"), scrim = $("#scrim");
  const closeSidebar = () => { sidebar.classList.remove("open"); scrim.classList.remove("show"); };
  $("#burger").addEventListener("click", () => {
    sidebar.classList.toggle("open"); scrim.classList.toggle("show");
  });
  scrim.addEventListener("click", closeSidebar);

  // ---- initial route ----
  go(location.hash.slice(2) || "overview");
})();
