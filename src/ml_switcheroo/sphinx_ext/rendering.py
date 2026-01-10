"""
HTML Rendering logic for the WASM demo.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.sphinx_ext.types import HierarchyMap

# Categorization Map based on Architecture Diagram
FRAMEWORK_GROUPS = {
  # Level 1
  "torch": "Level 1: High-Level",
  "mlx": "Level 1: High-Level",
  "tensorflow": "Level 1: High-Level",
  "keras": "Level 1: High-Level",
  # Level 2
  "jax": "Level 2: Numerics",
  "numpy": "Level 2: Numerics",
  # Level 0
  "html": "Level 0: Representations",
  "tikz": "Level 0: Representations",
  "latex_dsl": "Level 0: Representations",
  # Level 3
  "mlir": "Level 3: Standard IR",
  "stablehlo": "Level 3: Standard IR",
  # Level 4 - Updated to include SASS/RDNA
  "sass": "Level 4: ASM",
  "rdna": "Level 4: ASM",
}

GROUP_ORDER = [
  "Level 1: High-Level",
  "Level 2: Numerics",
  "Level 0: Representations",
  "Level 3: Standard IR",
  "Level 4: ASM",
  "Other",
]


def render_demo_html(hierarchy: HierarchyMap, examples_json: str, tier_metadata_json: str) -> str:
  """
  Constructs the full HTML block for the switcheroo demo.
  """
  root_dir = Path(__file__).parents[3]
  dist_dir = root_dir / "dist"
  wheel_name = "ml_switcheroo-latest-py3-none-any.whl"

  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      wheel_name = latest.name

  priority_order = get_framework_priority_order()
  available_roots = set(hierarchy.keys())

  if "torch" in available_roots:
    def_source = "torch"
  else:
    def_source = priority_order[0] if priority_order else "source_placeholder"

  if "jax" in available_roots and def_source != "jax":
    def_target = "jax"
  else:
    candidates = [fw for fw in priority_order if fw != def_source]
    def_target = candidates[0] if candidates else def_source

  # 3. Generate HTML Blocks
  primary_opts = _render_primary_options(hierarchy)

  opts_src = primary_opts.replace(f'value="{def_source}"', f'value="{def_source}" selected')
  opts_tgt = primary_opts.replace(f'value="{def_target}"', f'value="{def_target}" selected')

  src_flavours_html = _render_flavour_dropdown("src", hierarchy, def_source)
  tgt_flavours_html = _render_flavour_dropdown("tgt", hierarchy, def_target)

  return f""" 
        <div id="switcheroo-wasm-root" class="switcheroo-material-card" data-wheel="{wheel_name}">
            <script>
                window.SWITCHEROO_PRELOADED_EXAMPLES = {examples_json}; 
                window.SWITCHEROO_FRAMEWORK_TIERS = {tier_metadata_json}; 
            </script>

            <div class="demo-header">
                <div>
                    <h3 style="margin:0">Live Transpiler Demo</h3>
                    <small style="color:#666">Client-side WebAssembly Engine</small>
                </div>
                <span id="engine-status" class="status-badge">Offline</span>
            </div>

            <!-- Splash Screen -->
            <div id="demo-splash" class="splash-screen">
                <p>Initialize the engine to translate code between frameworks locally.</p>
                <button id="btn-load-engine" class="md-btn md-btn-primary">
                    ⚡ Initialize Engine
                </button>
            </div>

            <div id="demo-interface" style="display:none;">
                <!-- Toolbar -->
                <div class="translate-toolbar">
                    <div class="select-wrapper">
                        <select id="select-src" class="material-select">{opts_src}</select>
                        <div id="src-flavour-region" style="background:transparent; margin-left:5px;">
                            {src_flavours_html} 
                        </div>
                    </div>
                    <button id="btn-swap" class="swap-icon-btn" title="Swap Languages">⇄</button>
                    <div class="select-wrapper">
                        <select id="select-tgt" class="material-select">{opts_tgt}</select>
                         <div id="tgt-flavour-region" style="background:transparent; margin-left:5px;">
                            {tgt_flavours_html} 
                        </div>
                    </div>
                </div>

                <!-- Examples -->
                <div class="example-toolbar">
                    <span class="label">Load Example:</span>
                    <select id="select-example" class="material-select-sm">
                        <option value="" disabled selected>-- Select a Pattern --</option>
                    </select>
                </div>

                <!-- Editors -->
                <div class="editor-grid">
                    <div class="editor-group source-group">
                        <textarea id="code-source" spellcheck="false" class="material-input" placeholder="Source code...">import torch
import torch.nn as nn

class Model(nn.Module): 
    def forward(self, x): 
        return torch.abs(x)</textarea>
                    </div>
                    <div class="editor-group target-group">
                        <textarea id="code-target" readonly class="material-input output-bg" placeholder="Translation..."></textarea>
                    </div>
                </div>

                <!-- Controls -->
                <div class="controls-bar">
                    <button type="button" id="btn-retro" class="icon-btn" title="Matrix Mode">🕶️</button>

                    <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                </div>

                <!-- Output Tabs -->
                <div class="output-tabs">
                    <input type="radio" name="wm-tabs" id="tab-trace" checked>
                    <label for="tab-trace" class="tab-label"><span class="tab-icon">⏱️</span> Time Travel</label>
                     <input type="radio" name="wm-tabs" id="tab-console">
                    <label for="tab-console" class="tab-label"><span class="tab-icon">💻</span> Logs</label>
                    <input type="radio" name="wm-tabs" id="tab-mermaid">
                    <label for="tab-mermaid" class="tab-label"><span class="tab-icon">🌳</span> Graph</label>
                    <input type="radio" name="wm-tabs" id="tab-render">
                    <label for="tab-render" id="label-tab-render" class="tab-label" style="display:none;"><span class="tab-icon">📐</span> Render</label>
                    <input type="radio" name="wm-tabs" id="tab-weights">
                    <label for="tab-weights" id="label-tab-weights" class="tab-label" style="display:none;"><span class="tab-icon">🏋️</span> Weight Script</label>
                    <!-- Content: Trace & Time Travel -->
                    <div class="tab-panel" id="panel-trace">
                        <div class="time-travel-bar">
                            <div class="tt-controls">
                                <button id="btn-tt-prev" class="icon-btn" title="Step Back" disabled>⏪</button>
                                <input type="range" id="tt-slider" min="0" max="0" value="0" class="tt-slider" disabled>
                                <button id="btn-tt-next" class="icon-btn" title="Step Forward" disabled>⏩</button>
                            </div>
                            <div class="tt-info">
                                 <span id="tt-phase-label" class="tt-phase">Waiting...</span>
                            </div>
                            <details class="tt-config">
                                <summary>Pipeline Config</summary>
                                <div class="tt-config-menu">
                                    <div class="tt-config-item">
                                        <label title="Enable Strict Mode (Fail on unknown symbols)">
                                            <input type="checkbox" id="chk-strict-mode"> Strict Mode
                                        </label>
                                    </div>
                                    <div class="tt-config-item">
                                        <label title="Enable Import Fixer logic (Pruning/Injection)">
                                            <input type="checkbox" id="chk-opt-imports" checked> Import Fixer
                                        </label>
                                    </div>
                                    <div class="tt-config-item">
                                        <label title="Enable Graph Fusion/Optimization (Experimental)">
                                            <input type="checkbox" id="chk-opt-graph"> Graph Opt
                                        </label>
                                    </div>
                                </div>
                            </details>
                        </div>
                        <div id="trace-visualizer" class="trace-container"></div>
                    </div>
                    <!-- Content: Console -->
                    <div class="tab-panel" id="panel-console">
                        <pre id="console-output">Waiting for engine...</pre>
                    </div>
                    <!-- Content: Mermaid -->
                    <div class="tab-panel" id="panel-mermaid">
                        <div id="ast-mermaid-target" class="mermaid" style="text-align:center; overflow-x:auto;">
                            <em style="color:#999">Run a translation to generate the AST graph.</em>
                        </div>
                    </div>
                    <!-- Content: Render (TikZ) -->
                    <div class="tab-panel" id="panel-render">
                        <div id="tikz-output-container" class="tikz-container">
                            <em style="color:#999">Select 'TikZ' or 'HTML' as target framework and run translation.</em>
                        </div>
                    </div>
                    <!-- Content: Weight Script -->
                    <div class="tab-panel" id="panel-weights">
                        <div class="weight-script-container">
                            <textarea id="code-weights" readonly class="material-input output-bg" placeholder="Migration script will appear here..."></textarea>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """


def _render_primary_options(hierarchy: HierarchyMap) -> str:
  """
  Renders the top-level <option> elements for root frameworks.
  Organizes frameworks into <optgroup> categories based on their semantic level.
  """
  # Organizes roots into buckets
  grouped: Dict[str, List[str]] = defaultdict(list)

  priorities = get_framework_priority_order()
  roots = list(hierarchy.keys())
  sorted_roots = sorted(roots, key=lambda x: priorities.index(x) if x in priorities else 999)

  for root in sorted_roots:
    group_name = FRAMEWORK_GROUPS.get(root, "Other")
    grouped[group_name].append(root)

  html_parts = []

  for group_name in GROUP_ORDER:
    if group_name not in grouped:
      continue

    members = grouped[group_name]
    if not members:
      continue

    html_parts.append(f'<optgroup label="{group_name}">')
    for root in members:
      adapter = get_adapter(root)
      label = getattr(adapter, "display_name", root.capitalize())
      html_parts.append(f'<option value="{root}">{label}</option>')
    html_parts.append("</optgroup>")

  return "\n".join(html_parts)


def _render_flavour_dropdown(side: str, hierarchy: HierarchyMap, active_root: str) -> str:
  """
  Renders the secondary dropdown for Framework Flavours.
  """
  children = hierarchy.get(active_root, [])

  style = "display:inline-block;" if children else "display:none;"
  style += " background:#f0f4c3;"

  opts = []
  if children:
    for i, child in enumerate(children):
      sel = "selected" if i == 0 else ""
      opts.append(f'<option value="{child["key"]}" {sel}>{child["label"]}</option>')
  else:
    opts.append('<option value="" disabled selected>No Flavours</option>')

  return f""" 
        <select id="{side}-flavour" class="material-select-sm" style="{style}">
            {"".join(opts)} 
        </select>
        """
