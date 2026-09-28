/* OceanEmbed frontend runtime — live API client + dark-themed Plotly builders.
 * Everything rendered here comes from the FastAPI backend (real model output,
 * real ARGO validation). No numbers are hard-coded in the pages. */
(function (global) {
  "use strict";

  // ---- API client -------------------------------------------------------- //
  // The uvicorn server serves both /api/* and the static site from the same
  // origin. Use relative paths when on any http origin; fall back to :8000 for
  // file:// previews or other static servers.
  const API_BASE = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
  const j = (p) =>
    fetch(API_BASE + p).then((r) => {
      if (!r.ok) throw new Error(p + " → HTTP " + r.status);
      return r.json();
    });
  const api = {
    base: () => API_BASE,
    health: () => j("/api/health"),
    meta: () => j("/api/meta"),
    metrics: () => j("/api/metrics"),
    floats: () => j("/api/floats"),
    dates: () => j("/api/dates"),
    scatter: () => j("/api/scatter"),
    embedding: () => j("/api/embedding"),
    inputs: (d) => j("/api/inputs?date=" + encodeURIComponent(d)),
    field: (d, di) => j("/api/field?date=" + encodeURIComponent(d) + "&depth_index=" + di),
    reconstruct: (d, di) =>
      j("/api/reconstruct?date=" + encodeURIComponent(d) + "&depth_index=" + di),
    volume: (d, stride) =>
      j("/api/volume?date=" + encodeURIComponent(d) + (stride ? "&stride=" + stride : "")),
  };

  // ---- palette (mirrors the Tailwind tokens in each page) ---------------- //
  const C = {
    ink: "#0F172A",
    mut: "#64748B",
    grid: "rgba(148,163,184,0.22)",
    model: "#2563EB", // secondary-fixed (teal)
    clim: "#64748B", // muted
    glorys: "#7C3AED", // tertiary periwinkle
    obs: "#06B6D4", // coral
    surface: "#FFFFFF",
    line: "#E2E8F0",
  };
  const FONT = { family: "Inter, sans-serif", color: C.mut, size: 12 };
  const CONFIG = { displaylogo: false, responsive: true, displayModeBar: false };
  // Jet colorscale, hoisted so the transect / ocean-state builders match the map.
  const JETSCALE = [
    [0.0, "#00007F"], [0.125, "#0000FF"], [0.25, "#007FFF"], [0.375, "#00FFFF"],
    [0.5, "#7FFF7F"], [0.625, "#FFFF00"], [0.75, "#FF7F00"], [0.875, "#FF0000"],
    [1.0, "#7F0000"],
  ];
  // Non-blocking status banner (shown when the API is unreachable, cleared on
  // recovery) — replaces the old silent blank-page failure mode.
  function banner(msg, kind) {
    let el = document.getElementById("oe-banner");
    if (!el) {
      el = document.createElement("div");
      el.id = "oe-banner";
      el.style.cssText =
        "position:fixed;left:50%;top:14px;transform:translateX(-50%);z-index:9999;" +
        "max-width:680px;padding:11px 18px;border-radius:12px;font:600 13px Inter,sans-serif;" +
        "box-shadow:0 14px 34px -10px rgba(15,23,42,.45);line-height:1.45";
      document.body.appendChild(el);
    }
    const err = kind === "error";
    el.style.background = err ? "#fff1f2" : "#ecfdf5";
    el.style.border = "1px solid " + (err ? "#fecdd3" : "#a7f3d0");
    el.style.color = err ? "#9f1239" : "#065f46";
    el.innerHTML = msg;
    el.style.display = "block";
  }
  function clearBanner() {
    const el = document.getElementById("oe-banner");
    if (el) el.style.display = "none";
  }
  const base = (o) =>
    Object.assign(
      {
        font: FONT,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 54, r: 14, t: 10, b: 42 },
        hoverlabel: {
          bgcolor: "#FFFFFF",
          bordercolor: C.line,
          font: { family: "Inter", size: 12, color: C.ink },
        },
      },
      o || {}
    );
  const cbar = (unit) => ({
    title: { text: unit || "°C", font: { color: C.mut, size: 11 } },
    tickfont: { color: C.mut, size: 10 },
    outlinewidth: 0,
    thickness: 10,
    len: 0.92,
    x: 1.005,
  });

  // ---- builders ---------------------------------------------------------- //
  function heatmap(div, f, opts) {
    opts = opts || {};
    // Jet colorscale — matches the Streamlit screenshot exactly
    const JET = [
      [0.0,   "#00007F"],
      [0.125, "#0000FF"],
      [0.25,  "#007FFF"],
      [0.375, "#00FFFF"],
      [0.5,   "#7FFF7F"],
      [0.625, "#FFFF00"],
      [0.75,  "#FF7F00"],
      [0.875, "#FF0000"],
      [1.0,   "#7F0000"],
    ];
    const depthLabel = (f.depth != null && f.depth !== undefined) ? f.depth + " m" : "surface";
    const title = "Predicted temperature \u00a0|\u00a0 " + (f.date || opts.date || "") + " \u00a0|\u00a0 " + depthLabel;
    const trace = {
      type: "heatmap",
      x: f.lon,
      y: f.lat,
      z: f.z,
      colorscale: JET,
      zsmooth: "best",
      colorbar: {
        title: { text: "\u00b0C", font: { color: "#444", size: 12 } },
        tickfont: { color: "#555", size: 11 },
        outlinewidth: 0,
        thickness: 14,
        len: 0.9,
        x: 1.01,
      },
      hovertemplate: "lon %{x:.2f}\u00b0E \u00a0 lat %{y:.2f}\u00b0N<br>%{z:.2f} \u00b0C<extra></extra>",
    };
    if (opts.zmin != null) trace.zmin = opts.zmin;
    if (opts.zmax != null) trace.zmax = opts.zmax;
    Plotly.react(
      div,
      [trace],
      {
        font: { family: "Inter, sans-serif", color: C.mut, size: 12 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 60, r: 80, t: 44, b: 50 },
        title: {
          text: title,
          font: { family: "Inter, sans-serif", size: 14, color: "#f1f5f9" },
          x: 0,
          xref: "paper",
          pad: { l: 8 },
        },
        xaxis: {
          title: { text: "Longitude", standoff: 8, font: { color: C.mut } },
          gridcolor: C.grid,
          linecolor: C.grid,
          zeroline: false,
          tickfont: { size: 11, color: C.mut },
        },
        yaxis: {
          title: { text: "Latitude", standoff: 8, font: { color: C.mut } },
          gridcolor: C.grid,
          linecolor: C.grid,
          zeroline: false,
          scaleanchor: "x",
          scaleratio: 1,
          tickfont: { size: 11, color: C.mut },
        },
        hoverlabel: {
          bgcolor: "#1e293b",
          bordercolor: C.grid,
          font: { family: "Inter", size: 12, color: "#f1f5f9" },
        },
      },
      { displaylogo: false, responsive: true, displayModeBar: "hover",
        modeBarButtonsToRemove: ["select2d", "lasso2d", "toggleSpikelines"] }

    );
  }


  function rmseByDepth(div, m) {
    const y = m.depths;
    const mk = (rows, name, color, dash) => ({
      x: rows.map((r) => r.rmse),
      y: y,
      name: name,
      mode: "lines+markers",
      line: { color: color, width: dash ? 2 : 3, dash: dash || "solid", shape: "spline" },
      marker: { size: 6, color: color },
      hovertemplate: name + " · %{x:.2f} °C @ %{y:.0f} m<extra></extra>",
    });
    Plotly.react(
      div,
      [
        mk(m.climatology, "Climatology baseline", C.clim, "dot"),
        mk(m.glorys, "GLORYS ceiling", C.glorys, "dash"),
        mk(m.model, "OceanEmbed", C.model),
      ],
      base({
        xaxis: { title: "RMSE (°C)", gridcolor: C.grid, zeroline: false, rangemode: "tozero" },
        yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: C.grid },
        legend: { orientation: "h", y: 1.14, x: 1, xanchor: "right", font: { color: C.mut } },
      }),
      CONFIG
    );
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
    Plotly.react(div, traces, base({
      margin: { l: 60, r: 16, t: 10, b: 44 },
      xaxis: { title: opts.xtitle || "RMSE (°C)", gridcolor: C.grid, zeroline: false, rangemode: "tozero" },
      yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: C.grid },
      legend: { orientation: "h", y: 1.09, x: 1, xanchor: "right", font: { color: C.mut } },
    }), CONFIG);
  }

  function profile(div, fl) {
    const mk = (a, name, color, dash, markers) => ({
      x: a,
      y: fl.depths,
      name: name,
      mode: markers ? "markers" : "lines",
      line: { color: color, width: dash ? 2 : 3, dash: dash || "solid", shape: "spline" },
      marker: { size: markers ? 8 : 5, color: color },
      hovertemplate: name + " · %{x:.2f} °C @ %{y:.0f} m<extra></extra>",
    });
    Plotly.react(
      div,
      [
        mk(fl.climatology, "Climatology", C.clim, "dot"),
        mk(fl.glorys, "GLORYS", C.glorys, "dash"),
        mk(fl.model, "OceanEmbed", C.model),
        mk(fl.observed, "ARGO observed", C.obs, null, true),
      ],
      base({
        xaxis: { title: "Temperature (°C)", gridcolor: C.grid, zeroline: false },
        yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: C.grid },
        legend: { orientation: "h", y: 1.14, x: 1, xanchor: "right", font: { color: C.mut } },
      }),
      CONFIG
    );
  }

  function parity(div, s) {
    const lo = s.tmin != null ? s.tmin : 0,
      hi = s.tmax != null ? s.tmax : 32;
    const pts = {
      type: "scattergl",
      mode: "markers",
      x: s.observed,
      y: s.model,
      marker: {
        size: 4,
        color: s.depth,
        colorscale: [
          [0.0,  "#08306b"],
          [0.15, "#1565c0"],
          [0.3,  "#0288d1"],
          [0.45, "#00bcd4"],
          [0.6,  "#80deea"],
          [0.72, "#fff176"],
          [0.85, "#fb8c00"],
          [1.0,  "#b71c1c"]
        ],
        opacity: 0.55,
        colorbar: Object.assign(cbar("Depth m"), { x: 1.005 }),
        reversescale: true,
      },
      hovertemplate: "obs %{x:.2f} °C&nbsp; model %{y:.2f} °C<extra></extra>",
    };
    const ident = {
      type: "scatter",
      mode: "lines",
      x: [lo, hi],
      y: [lo, hi],
      line: { color: C.mut, width: 1.5, dash: "dash" },
      hoverinfo: "skip",
      showlegend: false,
    };
    Plotly.react(
      div,
      [pts, ident],
      base({
        xaxis: { title: "ARGO observed (°C)", gridcolor: C.grid, zeroline: false, range: [lo, hi] },
        yaxis: {
          title: "OceanEmbed (°C)",
          gridcolor: C.grid,
          zeroline: false,
          range: [lo, hi],
          scaleanchor: "x",
          scaleratio: 1,
        },
        showlegend: false,
      }),
      CONFIG
    );
  }

  // ---- 3D stacked-depth ocean volume ------------------------------------- //
  function volume3d(div, vol, opts) {
    opts = opts || {};
    const depths = vol.depths, lat = vol.lat, lon = vol.lon, layers = vol.layers;
    let zmin = Infinity, zmax = -Infinity;
    for (const g of layers) for (const row of g) for (const v of row)
      if (v != null) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
    if (!isFinite(zmin)) { zmin = 0; zmax = 30; }
    const traces = layers.map((g, k) => ({
      type: "surface",
      x: lon,
      y: lat,
      z: g.map((row) => row.map((v) => (v == null ? null : -k))),
      surfacecolor: g,
      cmin: zmin,
      cmax: zmax,
      colorscale: [
        [0.0,  "#08306b"],
        [0.15, "#1565c0"],
        [0.3,  "#0288d1"],
        [0.45, "#00bcd4"],
        [0.6,  "#80deea"],
        [0.72, "#fff176"],
        [0.85, "#fb8c00"],
        [1.0,  "#b71c1c"]
      ],
      showscale: k === 0,
      colorbar: cbar(opts.unit),
      opacity: opts.opacity != null ? opts.opacity : 0.72,
      text: g.map((row) =>
        row.map((v) => (v == null ? "" : v.toFixed(2) + " °C · " + depths[k] + " m"))
      ),
      hoverinfo: "x+y+text",
      lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 },
      showlegend: false,
    }));
    Plotly.react(
      div,
      traces,
      base({
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          aspectmode: "manual",
          aspectratio: { x: 1.5, y: 1.15, z: 1.12 },
          camera: { eye: { x: 1.55, y: -1.55, z: 0.92 } },
          xaxis: axis3d("Longitude °E"),
          yaxis: axis3d("Latitude °N"),
          zaxis: Object.assign(axis3d("Depth (m)"), {
            tickvals: depths.map((d, k) => -k),
            ticktext: depths.map((d) => "" + d),
          }),
        },
      }),
      CONFIG
    );
  }
  const axis3d = (title) => ({
    title: title,
    color: C.mut,
    gridcolor: C.grid,
    backgroundcolor: "rgba(11,18,32,0.35)",
    showbackground: true,
    zeroline: false,
  });

  // ---- small DOM/format helpers ------------------------------------------ //
  const fmt = {
    c: (v, d) => (v == null ? "—" : Number(v).toFixed(d == null ? 2 : d)),
    int: (v) => (v == null ? "—" : Number(v).toLocaleString("en-IN")),
  };
  function fill(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }
  function option(value, label, selected) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = label;
    if (selected) o.selected = true;
    return o;
  }

  // ---- vertical transect from the depth cube ----------------------------- //
  // Builds a temperature section (depth on y) from the /api/volume cube.
  // axis "lat": E–W slice at a fixed latitude row; axis "lon": N–S slice.
  function transect(div, cube, opts) {
    opts = opts || {};
    const axis = opts.axis || "lat";
    const idx = opts.index || 0;
    const depths = cube.depths, layers = cube.layers;
    let x, xtitle, z, atLabel;
    if (axis === "lat") {
      x = cube.lon;
      xtitle = "Longitude (°E)";
      z = depths.map((d, k) => layers[k][idx]);              // [depth][lon]
      atLabel = (cube.lat[idx]).toFixed(2) + "° N";
    } else {
      x = cube.lat;
      xtitle = "Latitude (°N)";
      z = depths.map((d, k) => layers[k].map((row) => row[idx])); // [depth][lat]
      atLabel = (cube.lon[idx]).toFixed(2) + "° E";
    }
    Plotly.react(
      div,
      [{
        type: "heatmap", x: x, y: depths, z: z,
        colorscale: JETSCALE, zsmooth: "best",
        colorbar: cbar("°C"),
        hovertemplate: "%{x:.2f} · %{y:.0f} m<br>%{z:.2f} °C<extra></extra>",
      }],
      base({
        margin: { l: 60, r: 74, t: 10, b: 46 },
        title: {
          text: "Temperature section  |  " + (opts.date || "") + "  |  " + atLabel,
          font: { family: "Inter, sans-serif", size: 13, color: C.mut },
          x: 0, xref: "paper", pad: { l: 6 },
        },
        xaxis: { title: { text: xtitle, standoff: 8, font: { color: C.mut } }, gridcolor: C.grid, zeroline: false, tickfont: { size: 11, color: C.mut } },
        yaxis: { title: { text: "Depth (m)", standoff: 8, font: { color: C.mut } }, autorange: "reversed", gridcolor: C.grid, tickfont: { size: 11, color: C.mut } },
      }),
      CONFIG
    );
  }

  // ---- derived ocean-state maps (MLD / thermocline) from the cube -------- //
  // Both are computed per water column from the real reconstructed profile:
  //   MLD  = shallowest depth cooler than the surface by ≥ 0.5 °C.
  //   Thermocline = depth of the strongest temperature gradient dT/dz.
  function oceanState(div, cube, mode) {
    const depths = cube.depths, layers = cube.layers;
    const ny = layers[0].length, nx = layers[0][0].length;
    const out = Array.from({ length: ny }, () => new Array(nx).fill(null));
    for (let i = 0; i < ny; i++) {
      for (let jx = 0; jx < nx; jx++) {
        const p = depths.map((d, k) => layers[k][i][jx]);
        if (p[0] == null) continue;                                   // land
        if (mode === "thermocline") {
          let best = -Infinity, bd = depths[0];
          for (let k = 0; k < p.length - 1; k++) {
            if (p[k] == null || p[k + 1] == null) continue;
            const g = (p[k] - p[k + 1]) / (depths[k + 1] - depths[k]);
            if (g > best) { best = g; bd = depths[k + 1]; }
          }
          out[i][jx] = bd;
        } else {
          const t0 = p[0];
          let mld = depths[depths.length - 1];
          for (let k = 1; k < p.length; k++) {
            if (p[k] == null) { mld = depths[k - 1]; break; }
            if (t0 - p[k] >= 0.5) { mld = depths[k]; break; }
          }
          out[i][jx] = mld;
        }
      }
    }
    const label = mode === "thermocline" ? "Thermocline (m)" : "Mixed-layer (m)";
    Plotly.react(
      div,
      [{
        type: "heatmap", x: cube.lon, y: cube.lat, z: out,
        colorscale: "YlGnBu", zsmooth: "best",
        colorbar: cbar(label),
        hovertemplate: "lon %{x:.2f}°E · lat %{y:.2f}°N<br>%{z:.0f} m<extra></extra>",
      }],
      base({
        margin: { l: 60, r: 92, t: 10, b: 46 },
        xaxis: { title: { text: "Longitude (°E)", standoff: 8, font: { color: C.mut } }, gridcolor: C.grid, zeroline: false, tickfont: { size: 11, color: C.mut } },
        yaxis: { title: { text: "Latitude (°N)", standoff: 8, font: { color: C.mut } }, gridcolor: C.grid, zeroline: false, scaleanchor: "x", scaleratio: 1, tickfont: { size: 11, color: C.mut } },
      }),
      CONFIG
    );
    return out;
  }

  // ---- pretrained-embedding PCA scatter ---------------------------------- //
  // One point per day from /api/embedding: training days as circles coloured by
  // month, held-out test days as larger diamonds — shows the test window sits
  // inside the learned manifold (no distribution shift / leakage).
  function embeddingScatter(div, points) {
    const train = points.filter((p) => !p.held_out);
    const test = points.filter((p) => p.held_out);
    const mk = (pts, name, symbol, size, linew, showscale) => ({
      type: "scattergl", mode: "markers", name: name,
      x: pts.map((p) => p.pc1), y: pts.map((p) => p.pc2),
      text: pts.map((p) => p.date),
      marker: {
        size: size, symbol: symbol,
        color: pts.map((p) => p.month),
        colorscale: "Turbo", cmin: 1, cmax: 12,
        opacity: 0.85, line: { color: "#fff", width: linew },
        showscale: showscale,
        colorbar: showscale
          ? Object.assign(cbar("Month"), { tickvals: [1, 4, 7, 10, 12] })
          : undefined,
      },
      hovertemplate: "%{text}<br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra>" + name + "</extra>",
    });
    Plotly.react(
      div,
      [
        mk(train, "Training days", "circle", 7, 0.4, true),
        mk(test, "Held-out test days", "diamond", 12, 1.5, false),
      ],
      base({
        margin: { l: 54, r: 80, t: 10, b: 46 },
        xaxis: { title: { text: "PC 1", standoff: 8, font: { color: C.mut } }, gridcolor: C.grid, zeroline: false, tickfont: { size: 11, color: C.mut } },
        yaxis: { title: { text: "PC 2", standoff: 8, font: { color: C.mut } }, gridcolor: C.grid, zeroline: false, tickfont: { size: 11, color: C.mut } },
        legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right", font: { color: C.mut } },
      }),
      CONFIG
    );
  }

  global.OE = { api, heatmap, rmseByDepth, profile, parity, volume3d, transect, oceanState, embeddingScatter, skillByDepth, banner, clearBanner, fmt, fill, option, C };
})(window);
