"""
Sphinx Extension for ML-Switcheroo WASM Demo (Level 2 Hierarchy Support + Tier Filtering).

This module renders the WASM interface HTML.
It introspects the Framework Registry to determine which frameworks
support hierarchical "Flavours" and what Tiers they support, sending this
metadata to the client-side JS for filtering invalid conversion pairs.

Features:
- **Hierarchy Detection**: Checks `inherits_from` metadata on adapters.
- **Tier Detection**: Checks `supported_tiers` on adapters.
- **Dynamic Priority**: Uses `config.get_framework_priority_order()` to determine
  default selections and example targets, removing hardcoded framework preferences.
- **Example Injection**: Preloads tiered examples from all registered adapters.
- **Retro Mode**: Adds UI toggle for CRT/Matrix visualization style.
- **AST Visualizer**: Creates DOM container for interactive Tree diagrams.
"""

import os
import shutil
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from docutils import nodes
from docutils.parsers.rst import Directive

from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.config import get_framework_priority_order

# Map of Parent Key -> List of {key, label} for children
HierarchyMap = Dict[str, List[Dict[str, str]]]


class SwitcherooDemo(Directive):
  """
  Sphinx Directive to embed the interactive WASM demo.

  Usage:
      .. switcheroo_demo::
  """

  has_content = True

  def run(self) -> List[nodes.raw]:
    """
    Main execution entry point for the directive.

    1. Finds the latest .whl file in dist/.
    2. Scans the registry for frameworks and examples.
    3. Renders the HTML template with dynamic dropdowns.

    Returns:
        List[nodes.raw]: A list containing the raw HTML node.
    """
    # 1. Locate Wheel
    root_dir = Path(__file__).parents[3]
    dist_dir = root_dir / "dist"
    wheel_name = "ml_switcheroo-latest-py3-none-any.whl"
    if dist_dir.exists():
      wheels = list(dist_dir.glob("*.whl"))
      if wheels:
        latest = sorted(wheels, key=os.path.getmtime)[-1]
        wheel_name = latest.name

    # 2. Build Hierarchy & Examples & Tier Metadata
    hierarchy, examples_json, tier_metadata_json = self._scan_registry()

    # 3. Generate HTML Blocks
    # Primary Options (Parents)
    primary_opts = self._render_primary_options(hierarchy)

    # Flavour DOM (Children) - rendered but hidden by default CSS logic in JS
    src_flavours_html = self._render_flavour_dropdown("src", hierarchy)
    tgt_flavours_html = self._render_flavour_dropdown("tgt", hierarchy)

    # 4. Determine Defaults dynamically from Config Priority
    priority_order = get_framework_priority_order()

    # Default Source: First priority framework (or fallback 'torch')
    def_source = priority_order[0] if priority_order else "torch"

    # Default Target: Second priority framework, or first if only one exists (or fallback 'jax')
    if len(priority_order) > 1:
      def_target = priority_order[1]
    elif priority_order:
      def_target = priority_order[0]
    else:
      def_target = "jax"

    opts_src = primary_opts.replace(f'value="{def_source}"', f'value="{def_source}" selected')
    opts_tgt = primary_opts.replace(f'value="{def_target}"', f'value="{def_target}" selected')

    html = f"""
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
                        <div id="src-flavour-region" style="display:none; margin-left:10px;">
                            {src_flavours_html}
                        </div>
                    </div>
                    <button id="btn-swap" class="swap-icon-btn" title="Swap Languages">⇄</button>
                    <div class="select-wrapper">
                        <select id="select-tgt" class="material-select">{opts_tgt}</select>
                         <div id="tgt-flavour-region" style="display:none; margin-left:10px;">
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
                    
                    <label class="md-switch" title="Strict Mode: Fail on unknown APIs.">
                        <input type="checkbox" id="chk-strict-mode">
                        <span class="md-switch-track"><span class="md-switch-thumb"></span></span>
                        <span class="md-switch-text">Strict Mode</span>
                    </label>

                    <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                </div>

                <!-- Three Tabs: Console, Mermaid, Timeline -->
                <div class="output-tabs">
                    <input type="radio" name="wm-tabs" id="tab-console">
                    <label for="tab-console" class="tab-label"><span class="tab-icon">💻</span> Console</label>

                    <input type="radio" name="wm-tabs" id="tab-mermaid">
                    <label for="tab-mermaid" class="tab-label"><span class="tab-icon">🌳</span> AST visualisation</label>

                    <input type="radio" name="wm-tabs" id="tab-trace" checked>
                    <label for="tab-trace" class="tab-label"><span class="tab-icon">⏱️</span> Timeline</label>

                    <div class="tab-panel" id="panel-console">
                        <pre id="console-output">Waiting for engine...</pre>
                    </div>

                    <div class="tab-panel" id="panel-mermaid">
                        <div class="ast-toolbar">
                            <button id="btn-ast-prev" class="md-btn md-btn-primary" style="padding:4px 10px; font-size:12px;">Before Pivot</button>
                            <button id="btn-ast-next" class="md-btn md-btn-primary" style="padding:4px 10px; font-size:12px; opacity:0.6;">After Pivot</button>
                        </div>
                        <div id="ast-mermaid-target" class="mermaid" style="text-align:center; overflow-x:auto;">
                            <em style="color:#999">Run a translation to generate the AST graph.</em>
                        </div>
                    </div>

                    <div class="tab-panel" id="panel-trace">
                        <div id="trace-visualizer" class="trace-container"></div>
                    </div>
                </div>

            </div>
        </div>
        """
    return [nodes.raw("", html, format="html")]

  def _scan_registry(self) -> Tuple[HierarchyMap, str, str]:
    """
    Scans registered adapters to build hierarchy, examples, and tier metadata.

    Returns:
        Tuple[HierarchyMap, str, str]:
            - The hierarchy dictionary.
            - JSON string of examples.
            - JSON string of framework tiers.
    """
    fws = available_frameworks()
    priorities = get_framework_priority_order()

    # 1. Build Node Map
    hierarchy: HierarchyMap = defaultdict(list)
    tier_metadata: Dict[str, List[str]] = {}

    # Track roots explicitly
    roots = set()

    for key in fws:
      adapter = get_adapter(key)
      if not adapter:
        continue

      label = getattr(adapter, "display_name", key.capitalize())
      parent = getattr(adapter, "inherits_from", None)

      # Extract Tiers
      tiers = []
      if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:
        tiers = [t.value for t in adapter.supported_tiers]
      else:
        # Fallback if property missing to prevent breakage
        tiers = ["array", "neural", "extras"]
      tier_metadata[key] = tiers

      if parent:
        # This is a child node (e.g. flax_nnx -> jax)
        hierarchy[parent].append({"key": key, "label": label})
      else:
        # This is a root node (e.g. torch, jax)
        roots.add(key)

    # 2. Convert to Render-Ready structures & Gather Examples
    examples = {}

    # Ensure we iterate roots sorted by priority provided by config
    sorted_roots = sorted(
      list(roots),
      key=lambda x: priorities.index(x) if x in priorities else 999,
    )

    final_hierarchy = {root: sorted(hierarchy.get(root, []), key=lambda x: x["label"]) for root in sorted_roots}

    # Collect Examples from Adapters
    for key in fws:
      adapter = get_adapter(key)
      if hasattr(adapter, "get_tiered_examples"):
        tiers = adapter.get_tiered_examples()
        parent_key = getattr(adapter, "inherits_from", None)

        for tier_name, code in tiers.items():
          # Unique ID for the example entry
          uid = f"{key}_{tier_name}"

          # Determine Source configuration
          if parent_key:
            src_fw = parent_key
            src_flavour = key
          else:
            src_fw = key
            src_flavour = None

          # Use tier_name to deduce required tier.
          req_tier = "extras"
          if "math" in tier_name:
            req_tier = "array"
          elif "neural" in tier_name:
            req_tier = "neural"

          # Simple cleanup for label
          clean_tier_name = tier_name.replace("tier", "")
          clean_label = (
            clean_tier_name.split("_")[-1].capitalize() if "_" in clean_tier_name else clean_tier_name.capitalize()
          )

          display_fw = getattr(adapter, "display_name", key.title())
          label = f"{display_fw}: {clean_label}"

          # Dynamic Target Heuristic: Next available in priority list
          tgt_fw = None
          tgt_flavour = None

          # Filter candidates: Must not be source, and not be parent of source
          candidates = [fw for fw in priorities if fw != src_fw and fw != parent_key and fw != src_flavour]

          # Pick the first viable candidate, searching forward from current src position if possible
          # This creates a logical rotation effect (A -> B, B -> C, C -> A)
          if candidates:
            try:
              curr_idx = priorities.index(src_fw)
              # Create round robin slice
              rotated = priorities[curr_idx + 1 :] + priorities[:curr_idx]
              for c in rotated:
                if c in candidates:
                  tgt_fw = c
                  break
            except ValueError:
              tgt_fw = candidates[0]

          # Ultimate fallback
          if not tgt_fw:
            tgt_fw = "target_placeholder"

          examples[uid] = {
            "label": label,
            "srcFw": src_fw,
            "srcFlavour": src_flavour,
            "tgtFw": tgt_fw,
            "tgtFlavour": tgt_flavour,
            "code": code,
            "requiredTier": req_tier,
          }

    return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)

  def _render_primary_options(self, hierarchy: HierarchyMap) -> str:
    """
    Renders the top-level <option> elements for root frameworks.
    """
    html = []
    for root in hierarchy.keys():
      adapter = get_adapter(root)
      label = getattr(adapter, "display_name", root.capitalize())
      html.append(f'<option value="{root}">{label}</option>')
    return "\n".join(html)

  def _render_flavour_dropdown(self, side: str, hierarchy: HierarchyMap) -> str:
    """
    Renders the secondary dropdown for Framework Flavours.
    It attempts to default to the first hierarchy parent found, resolving dynamic lookup.
    """
    # Dynamically find which framework has children to render defaults logic
    target_parent = "jax"  # Fallback
    if hierarchy:
      # Pick first available parent key
      target_parent = next(iter(hierarchy.keys()))

    children = hierarchy.get(target_parent, [])

    opts = []
    if children:
      # Pick first as default selected
      for i, child in enumerate(children):
        sel = "selected" if i == 0 else ""
        opts.append(f'<option value="{child["key"]}" {sel}>{child["label"]}</option>')
    else:
      # Fallback for empty state or tests
      opts.append('<option value="" disabled selected>No Flavours</option>')

    return f"""
        <select id="{side}-flavour" class="material-select-sm" style="background:#f0f4c3;">
            {"".join(opts)}
        </select>
        """


def setup(app: Any) -> Dict[str, Any]:
  """
  Sphinx Extension Setup Hook.
  """
  app.add_directive("switcheroo_demo", SwitcherooDemo)

  # CodeMirror
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")

  # Mermaid JS (Critical for AST Visualizer)
  app.add_js_file("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js")

  # Local Assets
  app.add_css_file("switcheroo_demo.css")
  app.add_css_file("trace_graph.css")
  app.add_js_file("trace_render.js")
  app.add_js_file("switcheroo_demo.js")

  app.connect("builder-inited", add_static_path)
  app.connect("build-finished", copy_wheel_and_reqs)

  return {"version": "0.9.8", "parallel_read_safe": True, "parallel_write_safe": True}


def add_static_path(app: Any) -> None:
  """Adds the extension's static directory to HTML build configuration."""
  static_path = Path(__file__).parent / "static"
  if static_path.exists() and hasattr(app, "config"):
    app.config.html_static_path.append(str(static_path.resolve()))


def copy_wheel_and_reqs(app: Any, exception: Optional[Exception]) -> None:
  """
  Post-build hook to copy the latest .whl file into _static for WASM usage.
  """
  if exception or not hasattr(app, "builder"):
    return
  here = Path(__file__).parent
  root_dir = here.parents[2]
  dist_dir = root_dir / "dist"
  static_dst = Path(app.builder.outdir) / "_static"
  static_dst.mkdir(exist_ok=True, parents=True)

  reqs_file = root_dir / "requirements.txt"
  if reqs_file.exists():
    shutil.copy2(reqs_file, static_dst / "requirements.txt")

  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      target_file = static_dst / latest.name
      if not target_file.exists() or target_file.stat().st_mtime < latest.stat().st_mtime:
        shutil.copy2(latest, target_file)
