import re

with open("design/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Add CSS for pages
if ".page { display: none; }" not in html:
    html = html.replace('</style>', '.page { display: none; }\n.page.show { display: block; }\n</style>')

# 2. Replace sidebar
new_sidebar = """<aside class="sidebar">
  <div class="brand">
    <div class="logo"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12c1.5-2 3-2 4.5 0S9.5 14 11 12s3-2 4.5 0 3 2 4.5 0"/><path d="M2 17c1.5-2 3-2 4.5 0s3 2 4.5 0 3-2 4.5 0 3 2 4.5 0"/><path d="M2 7c1.5-2 3-2 4.5 0s3 2 4.5 0 3-2 4.5 0 3 2 4.5 0"/></svg></div>
    <div><b>OceanEmbed</b><span>AI Ocean Intelligence</span></div>
  </div>
  <div class="navgroup" style="margin-top: 10px;">
    <nav class="nav" id="main-nav">
      <a class="active" data-route="overview" href="#overview"><i></i> Overview</a>
      <a data-route="skill-summary" href="#skill-summary"><i></i> Skill summary</a>
      <a data-route="float-explorer" href="#float-explorer"><i></i> Float explorer</a>
      <a data-route="domain-map" href="#domain-map"><i></i> Domain map</a>
      <a data-route="ocean-3d" href="#ocean-3d"><i></i> 3D ocean</a>
      <a data-route="depth-transect" href="#depth-transect"><i></i> Depth transect</a>
      <a data-route="ocean-state" href="#ocean-state"><i></i> Ocean state</a>
      <a data-route="data-pipeline" href="#data-pipeline"><i></i> Data pipeline</a>
      <a data-route="embedding" href="#embedding"><i></i> Embedding</a>
    </nav>
  </div>
  <style>
    .nav a i { width: 10px; height: 10px; border-radius: 50%; border: 2px solid #cbd5e1; display: inline-block; background: #fff; margin-right: 2px; }
    .nav a.active i { border: 3px solid #2563EB; background: #fff; }
    .nav a::before { display: none; }
  </style>
  <div class="sysbox">
    <div class="st">System Status</div>
    <div class="sr"><i class="dot-ok"></i>FastAPI Backend<b>Online</b></div>
    <div class="sr"><i class="dot-ok"></i>OceanEmbed Model<b>Loaded</b></div>
  </div>
</aside>"""

html = re.sub(r'<aside class="sidebar">[\s\S]*?</aside>', new_sidebar, html)

# 3. Wrap sections into pages
# Find <main class="content">
main_start = html.find('<main class="content">') + len('<main class="content">')
main_inner = html[main_start:html.find('</main>')]

# We'll replace the main inner with separated sections.
# Luckily the original HTML has comments like <!-- KPI ROW -->, <!-- OCEAN EXPLORER + CONDITIONS -->
# Let's just create a generic layout.

new_main = """
  <div class="page show" data-page="overview">
    <div class="pagehead">
      <div>
        <h1>Mission Control</h1>
        <p>Explore satellite observations, reconstruct subsurface ocean temperatures, and validate model predictions.</p>
        <div class="exp-tag">Active experiment <b>arabian-sea-2024-v3</b><span class="sep"></span>Dataset <b style="color:#059669">synchronised</b></div>
      </div>
    </div>
    <!-- KPI ROW -->
    <div class="kpis" id="overview-kpis"></div>
    <div class="grid" id="overview-grid"></div>
  </div>

  <div class="page" data-page="skill-summary">
    <div class="pagehead"><h1>Validation & Skill</h1><p>ARGO Validation Matchups.</p></div>
    <div id="skill-content"></div>
  </div>

  <div class="page" data-page="float-explorer">
    <div class="pagehead"><h1>Float Explorer</h1><p>Temperature Profile vs Depth</p></div>
    <div id="float-content"></div>
  </div>

  <div class="page" data-page="domain-map">
    <div class="pagehead"><h1>Domain Map</h1><p>Sea Surface Temperature and predicted depths.</p></div>
    <div id="domain-content"></div>
  </div>

  <div class="page" data-page="ocean-3d">
    <div class="pagehead"><h1>3D Ocean Volume</h1><p>Interactive 3D scatter volume.</p></div>
    <div class="card pad"><div style="height: 500px;" id="v-volume"></div></div>
  </div>

  <div class="page" data-page="depth-transect">
    <div class="pagehead"><h1>Depth Transect</h1><p>Temperature Cross-Section</p></div>
    <div id="transect-content"></div>
  </div>

  <div class="page" data-page="ocean-state">
    <div class="pagehead"><h1>Ocean State</h1><p>Derived diagnostics: mixed-layer depth and thermocline depth.</p></div>
    <div class="card pad"><i>This python-only diagnostic is being migrated to the FastAPI backend.</i></div>
  </div>

  <div class="page" data-page="data-pipeline">
    <div class="pagehead"><h1>Data Pipeline</h1><p>Data Sources & Pipeline Status</p></div>
    <div id="pipeline-content"></div>
  </div>

  <div class="page" data-page="embedding">
    <div class="pagehead"><h1>Pretrained Surface Embedding</h1><p>PCA projection of the embedding space.</p></div>
    <div class="card pad"><i>This python-only diagnostic is being migrated to the FastAPI backend.</i></div>
  </div>
  
  <div id="original-content" style="display: none;">
"""

# We hide the original content and just move the pieces we need using JS to preserve their complex HTML structure.
html = html[:main_start] + new_main + main_inner + "\n</div>" + html[html.find('</main>'):]

with open("design/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Updated index.html layout and sidebar")
