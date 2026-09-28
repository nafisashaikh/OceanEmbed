import re

with open("design/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. KPIs
html = html.replace('<div class="val">28.4<small>°C</small></div>', '<div class="val" id="kpi-sst">28.4<small>°C</small></div>')
html = html.replace('<div class="val">12<small>levels</small></div>', '<div class="val" id="kpi-depths">12<small>levels</small></div>')
html = html.replace('<div class="val">0.47<small>°C</small></div>', '<div class="val" id="kpi-rmse">0.47<small>°C</small></div>')
html = html.replace('<div class="val">1,284</div>', '<div class="val" id="kpi-samples">1,284</div>')
html = html.replace('<div class="val">92.3<small>%</small></div>', '<div class="val" id="kpi-coverage">92.3<small>%</small></div>')

# 2. Top Bar
html = html.replace('<div class="crumb">OceanEmbed / <b>Overview</b></div>', '<div class="crumb" id="crumb">OceanEmbed / <b>Overview</b></div>')

# 3. Ocean Explorer Map
# The SVG is from <svg class="mapsvg" to </svg>
map_regex = re.compile(r'<svg class="mapsvg"[\s\S]*?</svg>')
html = map_regex.sub('<div id="v-map" style="width: 100%; height: 440px; background: transparent;"></div>', html, count=1)

# 4. Temperature Profile
prof_regex = re.compile(r'<div class="profchart">\s*<svg.*?preserveAspectRatio="none">[\s\S]*?</svg>\s*</div>')
html = prof_regex.sub('<div class="profchart" id="v-profile"></div>', html, count=1)

# 5. Temperature Cross-Section (Transect)
xsec_regex = re.compile(r'<div class="xsec">\s*<svg viewBox="0 0 900 320"[\s\S]*?</svg>\s*</div>')
html = xsec_regex.sub('<div class="xsec" id="v-transect" style="height: 320px;"></div>', html, count=1)

# 6. ARGO Validation Scatter Plot
scatter_regex = re.compile(r'<div class="scatterwrap">\s*<svg viewBox="0 0 450 250"[\s\S]*?</svg>\s*</div>')
html = scatter_regex.sub('<div class="scatterwrap" id="v-scatter"></div>', html, count=1)

# 7. ARGO Metrics
html = html.replace('<div class="mv">1,284</div>', '<div class="mv" id="v-n">1,284</div>')
html = html.replace('<div class="mv">0.96</div>', '<div class="mv" id="v-r2">0.96</div>')
html = html.replace('<div class="mv">0.52<small>°C</small></div>', '<div class="mv" id="v-m-rmse">0.52<small>°C</small></div>')

# 8. Add Script Tags at the end of body
scripts = """
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script src="assets/oe.js"></script>
<script src="assets/dashboard.js"></script>
"""
html = html.replace('</body>', scripts + '\n</body>')

# 9. Sidebar Navigation routing
# Make links use hashes so dashboard.js can pick them up
html = html.replace('href="#"', 'href="javascript:void(0)"')

with open("design/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Injected IDs into design/index.html")
