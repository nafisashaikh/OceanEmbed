/* OceanEmbed — 3D Ocean page controller.
 * Fetches the full 15-level reconstructed volume for a date and renders it as a
 * stacked-surface Plotly view. Everything shown comes from /api/volume (real
 * model output) — no values are hard-coded in the page. */
(function () {
  "use strict";
  var view = document.getElementById("o3-view");
  var dateSel = document.getElementById("o3-date");
  var STRIDE = 2;

  function note(msg) {
    // Clear any prior Plotly state before replacing the DOM, otherwise a later
    // Plotly.react() diffs against destroyed nodes and the plot never repaints.
    if (window.Plotly && view.data) { try { Plotly.purge(view); } catch (e) {} }
    view.innerHTML =
      '<div style="height:100%;display:flex;align-items:center;justify-content:center;' +
      'color:#9FB0C3;font:500 14px Inter,sans-serif">' + msg + "</div>";
  }

  // real surface/deep extremes + ocean-column count, straight from the volume
  function stats(vol) {
    var L = vol.layers, last = L[L.length - 1];
    var smax = -Infinity, dmin = Infinity, ocean = 0, r, c;
    for (r = 0; r < L[0].length; r++)
      for (c = 0; c < L[0][r].length; c++) {
        var s = L[0][r][c];
        if (s != null) { ocean++; if (s > smax) smax = s; }
      }
    for (r = 0; r < last.length; r++)
      for (c = 0; c < last[r].length; c++) {
        var d = last[r][c];
        if (d != null && d < dmin) dmin = d;
      }
    OE.fill("o3-grid", vol.lat.length + " × " + vol.lon.length);
    OE.fill("o3-surf", isFinite(smax) ? OE.fmt.c(smax, 1) + " °C" : "—");
    OE.fill("o3-deep", isFinite(dmin) ? OE.fmt.c(dmin, 1) + " °C" : "—");
    OE.fill("o3-layers", String(vol.depths.length));
    OE.fill("o3-cells", OE.fmt.int(ocean));
  }
  function render(date) {
    note("Reconstructing volume for " + date + " …");
    OE.api.volume(date, STRIDE).then(function (vol) {
      OE.volume3d(view, vol);
      stats(vol);
    }).catch(function (e) {
      note("Could not load volume: " + e.message);
    });
  }

  OE.api.meta().then(function (m) {
    var dates = m.predicted_dates || [];
    dates.forEach(function (d, i) { dateSel.appendChild(OE.option(d, d, i === 0)); });
    dateSel.addEventListener("change", function () { render(dateSel.value); });
    render(dateSel.value || dates[0]);
  }).catch(function (e) {
    note("Could not load metadata: " + e.message);
  });
})();
