/* OceanEmbed frontend runtime — live API client + dark-themed Plotly builders.
 * Everything rendered here comes from the FastAPI backend (real model output,
 * real ARGO validation). No numbers are hard-coded in the pages. */
(function (global) {
  "use strict";

  // ---- API client (same-origin; backend serves this file too) ------------ //
  const j = (p) =>
    fetch(p).then((r) => {
      if (!r.ok) throw new Error(p + " → HTTP " + r.status);
      return r.json();
    });
  const api = {
    health: () => j("/api/health"),
    meta: () => j("/api/meta"),
    metrics: () => j("/api/metrics"),
    floats: () => j("/api/floats"),
    dates: () => j("/api/dates"),
    scatter: () => j("/api/scatter"),
    inputs: (d) => j("/api/inputs?date=" + encodeURIComponent(d)),
    field: (d, di) => j("/api/field?date=" + encodeURIComponent(d) + "&depth_index=" + di),
    reconstruct: (d, di) =>
      j("/api/reconstruct?date=" + encodeURIComponent(d) + "&depth_index=" + di),
    volume: (d, stride) =>
      j("/api/volume?date=" + encodeURIComponent(d) + (stride ? "&stride=" + stride : "")),
  };

  // ---- palette (mirrors the Tailwind tokens in each page) ---------------- //
  const C = {
    ink: "#E2E8F2",
    mut: "#9FB0C3",
    grid: "rgba(159,176,195,0.10)",
    model: "#3BC9B0", // secondary-fixed (teal)
    clim: "#9FB0C3", // muted
    glorys: "#AEB8E4", // tertiary periwinkle
    obs: "#F2A98F", // coral
    surface: "#0B1220",
    line: "#2F8FCE",
  };
  const FONT = { family: "Inter, sans-serif", color: C.mut, size: 12 };
  const CONFIG = { displaylogo: false, responsive: true, displayModeBar: false };
  const base = (o) =>
    Object.assign(
      {
        font: FONT,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { l: 54, r: 14, t: 10, b: 42 },
        hoverlabel: {
          bgcolor: "#0B1220",
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
    const trace = {
      type: "heatmap",
      x: f.lon,
      y: f.lat,
      z: f.z,
      colorscale: "Turbo",
      zsmooth: "best",
      colorbar: cbar(opts.unit),
      hovertemplate: "lon %{x:.2f}°E &nbsp; lat %{y:.2f}°N<br>%{z:.2f} °C<extra></extra>",
    };
    if (opts.zmin != null) trace.zmin = opts.zmin;
    if (opts.zmax != null) trace.zmax = opts.zmax;
    Plotly.react(
      div,
      [trace],
      base({
        xaxis: { title: "Longitude °E", gridcolor: C.grid, zeroline: false },
        yaxis: {
          title: "Latitude °N",
          gridcolor: C.grid,
          zeroline: false,
          scaleanchor: "x",
          scaleratio: 1,
        },
      }),
      CONFIG
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
        colorscale: "Turbo",
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
      colorscale: "Turbo",
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

  global.OE = { api, heatmap, rmseByDepth, profile, parity, volume3d, fmt, fill, option, C };
})(window);
