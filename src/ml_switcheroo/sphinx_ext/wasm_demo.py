"""
Sphinx Extension for ML-Switcheroo WASM Demo (Level 2 Hierarchy Support).

This module renders the WASM interface HTML.
It now introspects the Framework Registry to determine which frameworks
support hierarchical "Flavours" (e.g. JAX -> Flax NNX, PaxML) and generates
secondary dropdown menus for them.

Features:
- **Hierarchy Detection**: Checks `inherits_from` metadata on adapters.
- **Dynamic HTML**: Generates nested `<select>` elements hidden/shown via JS.
- **Flavour Routing**: Associates `flax_nnx` with the parent `jax` key.
- **Example Injection**: Preloads tiered examples from all registered adapters.
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

    # 2. Build Hierarchy & Examples
    hierarchy, examples_json = self._scan_registry()

    # 3. Generate HTML Blocks
    # Primary Options (Parents)
    primary_opts = self._render_primary_options(hierarchy)

    # Flavour DOM (Children) - rendered but hidden by default CSS logic in JS
    src_flavours_html = self._render_flavour_dropdown("src", hierarchy)
    tgt_flavours_html = self._render_flavour_dropdown("tgt", hierarchy)

    # Pre-select JAX/Torch for demo defaults
    opts_src = primary_opts.replace('value="torch"', 'value="torch" selected')
    # Default target to JAX to show off the hierarchy
    opts_tgt = primary_opts.replace('value="jax"', 'value="jax" selected')

    html = f"""
        <div id="switcheroo-wasm-root" class="switcheroo-material-card" data-wheel="{wheel_name}">
            <script>
                window.SWITCHEROO_PRELOADED_EXAMPLES = {examples_json};
            </script>

            <div class="demo-header">
                <div>
                    <h3 style="margin:0">Live Transpiler Demo</h3>
                    <small style="color:#666">Client-side WebAssembly Engine</small>
                </div>
                <span id="engine-status" class="status-badge">Offline</span>
            </div>

            <div id="demo-splash" class="splash-screen">
                <p>Initialize the engine to translate code between frameworks locally.</p>
                <button id="btn-load-engine" class="md-btn md-btn-primary">
                    ⚡ Initialize Engine
                </button>
            </div>

            <div id="demo-interface" style="display:none;">
                <div class="translate-toolbar">
                    <!-- Source Column -->
                    <div class="select-wrapper">
                        <select id="select-src" class="material-select">
                            {opts_src}
                        </select>
                        <div id="src-flavour-region" style="display:none; margin-left:10px;">
                            {src_flavours_html}
                        </div>
                    </div>

                    <button id="btn-swap" class="swap-icon-btn" title="Swap Languages">⇄</button>

                    <!-- Target Column -->
                    <div class="select-wrapper">
                        <select id="select-tgt" class="material-select">
                            {opts_tgt}
                        </select>
                         <div id="tgt-flavour-region" style="display:none; margin-left:10px;">
                            {tgt_flavours_html}
                        </div>
                    </div>
                </div>

                <div class="example-toolbar">
                    <span class="label">Load Example:</span>
                    <select id="select-example" class="material-select-sm">
                        <option value="" disabled selected>-- Select a Pattern --</option>
                    </select>
                </div>

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

                <div class="controls-bar">
                    <label class="md-switch" title="Strict Mode: Fail on unknown APIs.">
                        <input type="checkbox" id="chk-strict-mode">
                        <span class="md-switch-track"><span class="md-switch-thumb"></span></span>
                        <span class="md-switch-text">Strict Mode</span>
                    </label>

                    <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                </div>

                <div id="trace-visualizer" class="trace-container" style="display:none;"></div>
            </div>

            <div class="console-group">
                <div class="editor-label">Engine Logs</div>
                <pre id="console-output">Waiting for engine...</pre>
            </div>
        </div>
        """
    return [nodes.raw("", html, format="html")]

  def _scan_registry(self) -> Tuple[HierarchyMap, str]:
    """
    Scans registered adapters to build hierarchy and examples.

    Returns:
        Tuple[HierarchyMap, str]: A tuple containing:
            - The hierarchy dictionary (Parent -> [Children]).
            - A JSON string of all discovered examples.
    """
    fws = available_frameworks()

    # 1. Build Node Map
    # Nodes that are parents (inherits_from=None) or implicit parents.
    # Structure: parent_key -> List of children definitions
    hierarchy: HierarchyMap = defaultdict(list)

    # Track roots explicitly
    roots = set()

    for key in fws:
      adapter = get_adapter(key)
      if not adapter:
        continue

      label = getattr(adapter, "display_name", key.capitalize())
      parent = getattr(adapter, "inherits_from", None)

      if parent:
        # This is a child node (e.g. flax_nnx -> jax)
        hierarchy[parent].append({"key": key, "label": label})
      else:
        # This is a root node (e.g. torch, jax)
        roots.add(key)

    # 2. Convert to Render-Ready structures & Gather Examples
    examples = {}

    # Ensure we iterate roots sorted by priority
    main_priority = ["torch", "jax", "tensorflow", "numpy"]
    sorted_roots = sorted(list(roots), key=lambda x: (main_priority.index(x) if x in main_priority else 99, x))

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
          # If this is a flavour (child), prompt user to select Parent as FW and Key as Flavour
          if parent_key:
            src_fw = parent_key
            src_flavour = key
          else:
            src_fw = key
            src_flavour = None

          # Simple cleanup for label (e.g. "tier2_neural" -> "Neural")
          # Split on underscores, take the last part and title case it.
          clean_tier_name = tier_name.replace("tier", "")
          clean_label = (
            clean_tier_name.split("_")[-1].capitalize() if "_" in clean_tier_name else clean_tier_name.capitalize()
          )

          display_fw = getattr(adapter, "display_name", key.title())
          label = f"{display_fw}: {clean_label}"

          # Target Heuristic
          # If source is JAX-based, target Torch. Otherwise target JAX.
          tgt_fw = "jax"
          if "jax" in src_fw:
            tgt_fw = "torch"

          examples[uid] = {
            "label": label,
            "srcFw": src_fw,
            "srcFlavour": src_flavour,
            "tgtFw": tgt_fw,
            "tgtFlavour": None,  # Default
            "code": code,
          }

    return final_hierarchy, json.dumps(examples)

  def _render_primary_options(self, hierarchy: HierarchyMap) -> str:
    """
    Renders the top-level <option> elements for root frameworks.

    Args:
        hierarchy (HierarchyMap): The framework hierarchy.

    Returns:
        str: HTML string of option elements.
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

    Currently hardcodes generation for 'jax' based on hierarchy.
    In a robust implementation, this would be generated dynamically by JS,
    but pre-rendering works for the known JAX/Flax split.

    Args:
        side (str): 'src' or 'tgt', for ID generation.
        hierarchy (HierarchyMap): The framework hierarchy.

    Returns:
        str: HTML string for the select element.
    """
    # We only support JAX hierarchy in UI for now
    children = hierarchy.get("jax", [])
    if not children:
      return ""

    # Default to Flax NNX if available, else first child
    # Flax NNX key is 'flax_nnx' per phase 3
    default_child = "flax_nnx"

    opts = []
    for child in children:
      sel = "selected" if child["key"] == default_child else ""
      opts.append(f'<option value="{child["key"]}" {sel}>{child["label"]}</option>')

    return f"""
        <select id="{side}-flavour" class="material-select-sm" style="background:#f0f4c3;">
            {"".join(opts)}
        </select>
        """


def setup(app: Any) -> Dict[str, Any]:
  """
  Sphinx Extension Setup Hook.

  Registers directives and static assets.
  """
  app.add_directive("switcheroo_demo", SwitcherooDemo)
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")
  app.add_css_file("switcheroo_demo.css")
  app.add_css_file("trace_graph.css")
  app.add_js_file("trace_render.js")
  app.add_js_file("switcheroo_demo.js")

  app.connect("builder-inited", add_static_path)
  app.connect("build-finished", copy_wheel_and_reqs)

  return {"version": "0.9.5", "parallel_read_safe": True, "parallel_write_safe": True}


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
