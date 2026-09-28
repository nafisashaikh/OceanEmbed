/* ============================================================
   OceanEmbed — themed Plotly chart builders
   ============================================================ */
const OE = (() => {
  const FONT = { family: "Inter, sans-serif", color: "#334155", size: 12 };
  const TURBO = "Turbo";
  const baseLayout = (over = {}) => Object.assign({
    font: FONT,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 52, r: 16, t: 10, b: 44 },
    hoverlabel: { bgcolor: "#0B1628", bordercolor: "#2563EB",
      font: { family: "Inter, sans-serif", size: 12, color: "#F8FAFC" } },
  }, over);
  const CONFIG = { displayModeBar: true, responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };

  function colorbar() {
    return { title: { text: "°C", font: { color: "#475569" }, side: "top" },
      tickfont: { color: "#475569", size: 10 },
      x: 1.02, xanchor: "left", xpad: 4,
      outlinecolor: "rgba(148,163,184,.5)", outlinewidth: 1, thickness: 12, len: .9 };
  }

  // ---- domain heatmap ----
  function heatmap(div, field, di) {
    const z = field.temp[di];
    const trace = {
      type: "heatmap", x: field.lon, y: field.lat, z,
      colorscale: TURBO, zsmooth: "best",
      colorbar: colorbar(),
      hovertemplate: "LON %{x:.2f}°  ·  LAT %{y:.2f}°<br>TEMP %{z:.2f} °C<extra></extra>",
    };
    Plotly.react(div, [trace], baseLayout({
      margin: { l: 52, r: 82, t: 10, b: 44 },
      xaxis: { title: "Longitude (°E)", gridcolor: "#EEF2F7", zeroline: false },
      yaxis: { title: "Latitude (°N)", gridcolor: "#EEF2F7", zeroline: false,
        scaleanchor: "x", scaleratio: 1 },
    }), CONFIG);
  }

  // ---- RMSE by depth ----
  function rmseByDepth(div, m) {
    const mk = (arr) => arr.map(r => r.rmse);
    const model = { x: mk(m.model), y: m.depths, name: "Model", mode: "lines+markers",
      line: { color: "#2563EB", width: 3, shape: "spline" },
      marker: { size: 7, color: "#2563EB" },
      hovertemplate: "Model<br>%{x:.2f} °C @ %{y:.0f} m<extra></extra>" };
    const clim = { x: mk(m.climatology), y: m.depths, name: "Climatology", mode: "lines+markers",
      line: { color: "#94A3B8", width: 2, dash: "dot", shape: "spline" },
      marker: { size: 6, color: "#94A3B8" },
      hovertemplate: "Climatology<br>%{x:.2f} °C @ %{y:.0f} m<extra></extra>" };
    Plotly.react(div, [clim, model], baseLayout({
      margin: { l: 60, r: 16, t: 10, b: 44 },
      xaxis: { title: "RMSE (°C)", gridcolor: "#EEF2F7", zeroline: false },
      yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#EEF2F7" },
      legend: { orientation: "h", y: 1.08, x: 1, xanchor: "right", bgcolor: "rgba(255,255,255,.6)" },
    }), CONFIG);
  }
  /* PLACEHOLDER-VOL */
  // ---- 3D stacked-surface teaser ----
  function volume(div, field, depthIdxs) {
    // shared colour scale across shown layers
    let vmin = Infinity, vmax = -Infinity;
    depthIdxs.forEach(di => field.temp[di].forEach(row => row.forEach(v => {
      if (v != null) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    })));
    const nlat = field.lat.length, nlon = field.lon.length;
    const traces = depthIdxs.map((di, k) => {
      const layer = field.temp[di];
      const d = field.depths[di];
      const zconst = Array.from({ length: nlat }, () => new Array(nlon).fill(-d));
      const text = layer.map((row, i) => row.map((v, j) =>
        `LON ${field.lon[j].toFixed(1)}°  ·  LAT ${field.lat[i].toFixed(1)}°` +
        `<br>TEMP ${v == null ? "land / no data" : v.toFixed(2) + " °C"}` +
        `<br>DEPTH ${d} m`));
      return {
        type: "surface", x: field.lon, y: field.lat, z: zconst,
        surfacecolor: layer, cmin: vmin, cmax: vmax, colorscale: TURBO,
        showscale: k === 0, colorbar: colorbar(), opacity: 0.9,
        text, hoverinfo: "text", name: "", showlegend: false,
      };
    });
    Plotly.react(div, traces, baseLayout({
      margin: { l: 0, r: 62, t: 10, b: 0 },
      scene: {
        aspectmode: "manual", aspectratio: { x: 1.4, y: 1, z: 0.9 },
        camera: { eye: { x: 1.5, y: -1.6, z: 0.9 } },
        xaxis: { title: "Lon", backgroundcolor: "rgba(0,0,0,0)", gridcolor: "#E2E8F0", color: "#64748B" },
        yaxis: { title: "Lat", backgroundcolor: "rgba(0,0,0,0)", gridcolor: "#E2E8F0", color: "#64748B" },
        zaxis: { title: "Depth (m)", backgroundcolor: "rgba(0,0,0,0)", gridcolor: "#E2E8F0", color: "#64748B",
          tickvals: depthIdxs.map(di => -field.depths[di]),
          ticktext: depthIdxs.map(di => field.depths[di] + " m") },
      },
    }), CONFIG);
  }
  // ---- skill-by-depth (generic multi-series, depth on y) ----
  function skillByDepth(div, depths, defs, opts) {
    opts = opts || {};
    const traces = defs.map(d => ({
      x: d.vals, y: depths, name: d.name,
      mode: d.markers ? "markers" : "lines+markers",
      line: { color: d.color, width: d.width || 2.5, dash: d.dash || "solid", shape: "spline" },
      marker: { size: d.markers ? 9 : 6, color: d.color,
        line: d.markers ? { color: "#fff", width: 1.3 } : undefined },
      connectgaps: false,
      hovertemplate: d.name + " · %{x:.2f} @ %{y:.0f} m<extra></extra>",
    }));
    Plotly.react(div, traces, baseLayout({
      margin: { l: 60, r: 16, t: 10, b: 44 },
      xaxis: { title: opts.xtitle || "RMSE (°C)", gridcolor: "#EEF2F7", zeroline: false, rangemode: "tozero" },
      yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#EEF2F7" },
      legend: { orientation: "h", y: 1.09, x: 1, xanchor: "right", bgcolor: "rgba(255,255,255,.6)" },
    }), CONFIG);
  }

  // ---- parity scatter (obs vs model, coloured by depth) ----
  function parity(div, s) {
    const lo = s.tmin != null ? s.tmin : 0, hi = s.tmax != null ? s.tmax : 32;
    const pts = { type: "scattergl", mode: "markers", x: s.observed, y: s.model,
      marker: { size: 4, color: s.depth, colorscale: TURBO, opacity: 0.5, reversescale: true,
        colorbar: { title: { text: "Depth (m)", font: { color: "#475569" }, side: "top" },
          tickfont: { color: "#475569", size: 10 }, thickness: 12, len: 0.9,
          x: 1.02, xanchor: "left", xpad: 4,
          outlinecolor: "rgba(148,163,184,.5)", outlinewidth: 1 } },
      hovertemplate: "obs %{x:.2f} °C · model %{y:.2f} °C<extra></extra>" };
    const ident = { type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi],
      line: { color: "#94A3B8", width: 1.5, dash: "dash" }, hoverinfo: "skip", showlegend: false };
    Plotly.react(div, [pts, ident], baseLayout({
      margin: { l: 60, r: 92, t: 10, b: 44 },
      xaxis: { title: "ARGO observed (°C)", gridcolor: "#EEF2F7", zeroline: false, range: [lo, hi] },
      yaxis: { title: "OceanEmbed (°C)", gridcolor: "#EEF2F7", zeroline: false, range: [lo, hi],
        scaleanchor: "x", scaleratio: 1 },
      showlegend: false,
    }), CONFIG);
  }

  // ---- ARGO float positions (clickable scatter over the domain) ----
  function floatsMap(div, floats, activeId) {
    const REG = { "Arabian Sea": "#2563EB", "Bay of Bengal": "#06B6D4" };
    const tr = { type: "scatter", mode: "markers",
      x: floats.map(f => f.lon), y: floats.map(f => f.lat),
      marker: {
        size: floats.map(f => (f.id === activeId ? 20 : 13)),
        color: floats.map(f => REG[f.region] || "#2563EB"),
        line: { color: floats.map(f => (f.id === activeId ? "#0F172A" : "#ffffff")),
          width: floats.map(f => (f.id === activeId ? 3 : 1.5)) },
      },
      text: floats.map(f => `Float ${f.id}<br>${f.region} · ${f.n} profiles`),
      hovertemplate: "%{text}<br>lon %{x:.2f}°E · lat %{y:.2f}°N<extra></extra>",
      customdata: floats.map(f => f.id) };
    Plotly.react(div, [tr], baseLayout({
      xaxis: { title: "Longitude (°E)", gridcolor: "#EEF2F7", zeroline: false },
      yaxis: { title: "Latitude (°N)", gridcolor: "#EEF2F7", zeroline: false, scaleanchor: "x", scaleratio: 1 },
    }), CONFIG);
  }

  // ---- single-float vertical profile ----
  function profile(div, fl) {
    const mk = (a, name, color, dash, markers) => ({
      x: a, y: fl.depths, name: name, mode: markers ? "markers" : "lines",
      line: { color: color, width: dash ? 2 : 2.5, dash: dash || "solid", shape: "spline" },
      marker: { size: markers ? 9 : 5, color: color, line: markers ? { color: "#fff", width: 1.4 } : undefined },
      hovertemplate: name + " · %{x:.2f} °C @ %{y:.0f} m<extra></extra>",
    });
    Plotly.react(div, [
      mk(fl.climatology, "Climatology", "#94A3B8", "dot"),
      mk(fl.glorys, "GLORYS", "#7C3AED", "dash"),
      mk(fl.model, "OceanEmbed", "#2563EB"),
      mk(fl.observed, "ARGO observed", "#F43F5E", null, true),
    ], baseLayout({
      xaxis: { title: "Temperature (°C)", gridcolor: "#EEF2F7", zeroline: false },
      yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#EEF2F7" },
      legend: { orientation: "h", y: 1.1, x: 1, xanchor: "right", bgcolor: "rgba(255,255,255,.6)" },
    }), CONFIG);
  }

  // ---- vertical transect: temperature vs (longitude × depth) at one latitude ----
  function transect(div, field, latIndex) {
    const z = field.depths.map((d, di) => field.temp[di][latIndex]); // one lon-row per depth
    const trace = { type: "heatmap", x: field.lon, y: field.depths, z: z,
      colorscale: TURBO, zsmooth: "best", colorbar: colorbar(),
      hovertemplate: "lon %{x:.2f}°E · %{y:.0f} m<br>%{z:.2f} °C<extra></extra>" };
    Plotly.react(div, [trace], baseLayout({
      margin: { l: 60, r: 82, t: 10, b: 44 },
      xaxis: { title: "Longitude (°E)", gridcolor: "#EEF2F7", zeroline: false },
      yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#EEF2F7" },
    }), CONFIG);
  }

  return { heatmap, rmseByDepth, volume, skillByDepth, parity, floatsMap, profile, transect };
})();
