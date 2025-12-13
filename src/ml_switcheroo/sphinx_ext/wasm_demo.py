"""
Sphinx Extension for ML-Switcheroo WASM Demo.

This module provides a custom `switcheroo_demo` directive that renders an interactive
in-browser transpiler interface. It dynamically populates the framework selection
dropdowns by querying the `ml_switcheroo.frameworks` registry and reading metadata
from JSON Snapshots.

Features:
- Dynamically discovers registered frameworks for the UI.
- Generates HTML for the WASM interface (Source/Target selectors, Editor divs).
- Injects necessary JS/CSS assets into the Sphinx build.
- **Protocol-Driven Examples**: Extracts code snippets from `snapshots/*.json`.
- **Strict Mode Toggle**: UI element to enabling strict API validation.
"""

import os
import shutil
import json
import glob
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive

from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.semantics.paths import resolve_snapshots_dir


class SwitcherooDemo(Directive):
  has_content = True

  def run(self):
    # 1. Locate the Wheel (Best effort for render time, actual copy happens later)
    root_dir = Path(__file__).parents[3]
    dist_dir = root_dir / "dist"

    wheel_name = "ml_switcheroo-latest-py3-none-any.whl"
    if dist_dir.exists():
      wheels = list(dist_dir.glob("*.whl"))
      if wheels:
        latest = sorted(wheels, key=os.path.getmtime)[-1]
        wheel_name = latest.name

    # 2. Dynamically Generate Framework Options & Examples
    options_html, examples_json = self._generate_dynamic_data()

    # Create target variant (default JAX selected)
    opts_src = options_html
    opts_tgt = options_html.replace('value="jax"', 'value="jax" selected')
    # Ensure Torch is default source if present
    if 'value="torch"' in opts_src:
      opts_src = opts_src.replace(" selected", "")  # clear others
      opts_src = opts_src.replace('value="torch"', 'value="torch" selected')

    html = f"""
        <div id="switcheroo-wasm-root" class="switcheroo-material-card" data-wheel="{wheel_name}">
            <!-- Inject Data-Driven Examples -->
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
                    <div class="select-wrapper">
                        <select id="select-src" class="material-select">
                            {opts_src}
                        </select>
                    </div>

                    <button id="btn-swap" class="swap-icon-btn" title="Swap Languages">⇄</button>

                    <div class="select-wrapper">
                        <select id="select-tgt" class="material-select">
                            {opts_tgt}
                        </select>
                    </div>
                </div>

                <!-- Example Selector Toolbar -->
                <div class="example-toolbar">
                    <span class="label">Load Example:</span>
                    <select id="select-example" class="material-select-sm">
                        <option value="" disabled selected>-- Select a Pattern --</option>
                    </select>
                </div>

                <div class="editor-grid">
                    <div class="editor-group source-group">
                        <textarea id="code-source" spellcheck="false" class="material-input" placeholder="Paste source code here...">import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.abs(x)</textarea>
                    </div>
                    <div class="editor-group target-group">
                        <textarea id="code-target" readonly class="material-input output-bg" placeholder="Translation will appear here..."></textarea>
                    </div>
                </div>

                <div class="controls-bar">
                    <!-- Strict Mode Toggle -->
                    <label class="md-switch" title="Strict Mode: Fail if an API cannot be mapped instead of passing it through.">
                        <input type="checkbox" id="chk-strict-mode">
                        <span class="md-switch-track">
                            <span class="md-switch-thumb"></span>
                        </span>
                        <span class="md-switch-text">Strict Mode</span>
                    </label>

                    <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                </div>

                <!-- Trace Visualization Container -->
                <div id="trace-visualizer" class="trace-container" style="display:none;">
                    <div class="trace-row placeholder">
                        <div class="trace-content" style="text-align:center;color:#999;padding:20px;">
                            Trace events will appear here after conversion...
                        </div>
                    </div>
                </div>
            </div>

            <div class="console-group">
                <div class="editor-label">Engine Logs</div>
                <pre id="console-output">Waiting for engine...</pre>
            </div>
        </div>
        """
    node = nodes.raw("", html, format="html")
    return [node]

  def _generate_dynamic_data(self) -> tuple[str, str]:
    """
    Iterates available frameworks to generate:
    1. HTML <options> for dropdown (using Adapter for Name).
    2. JSON string of example code snippets (using Snapshots).
    """
    fws = available_frameworks()
    priority = ["torch", "jax"]
    sorted_fws = sorted(fws, key=lambda x: (priority.index(x) if x in priority else 99, x))

    parts = []
    examples = {}

    # Load Snapshots Metadata
    examples_map = self._load_examples_from_snapshots()

    for key in sorted_fws:
      adapter = get_adapter(key)
      if not adapter:
        continue

      # 1. Option Label
      label = getattr(adapter, "display_name", key.capitalize())
      parts.append(f'<option value="{key}">{label}</option>')

      # 2. Example Extraction (From JSON Metadata)
      code = examples_map.get(key)

      # Fallback (deprecated but safe)
      if not code and hasattr(adapter, "get_example_code"):
        try:
          code = adapter.get_example_code()
        except Exception:
          pass

      if code:
        examples[key] = {
          "label": f"Standard {label} Example",
          "srcFw": key,
          "tgtFw": "jax" if key != "jax" else "torch",
          "code": code,
        }

    return "\n".join(parts), json.dumps(examples)

  def _load_examples_from_snapshots(self) -> dict:
    """Helper to scan snapshot directory and extract 'demo_example' metadata."""
    snap_dir = resolve_snapshots_dir()
    if not snap_dir.exists():
      return {}

    results = {}
    # Find all JSONs
    for fpath in snap_dir.glob("*.json"):
      try:
        content = json.loads(fpath.read_text(encoding="utf-8"))

        # Identify Framework
        fw = content.get("__framework__")
        if not fw:
          # Fallback to filename: 'torch_v1.0.json' -> 'torch'
          fw = fpath.name.split("_")[0]

        # Identify Example
        ex = content.get("metadata", {}).get("demo_example")
        if fw and ex:
          # Overwrite if we find multiple; assume FS order or specific pattern isn't critical
          # unless versioning matters. Latest likely overwrites older.
          results[fw] = ex

      except Exception:
        continue

    return results


def setup(app):
  app.add_directive("switcheroo_demo", SwitcherooDemo)

  # External Deps
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")

  # Internal Assets
  app.add_css_file("switcheroo_demo.css")
  app.add_css_file("trace_graph.css")
  app.add_js_file("trace_render.js")
  app.add_js_file("switcheroo_demo.js")

  app.connect("builder-inited", add_static_path)
  app.connect("build-finished", copy_wheel_and_reqs)

  return {"version": "0.9.2", "parallel_read_safe": True, "parallel_write_safe": True}


def add_static_path(app):
  static_path = Path(__file__).parent / "static"
  if static_path.exists() and hasattr(app, "config"):
    app.config.html_static_path.append(str(static_path.resolve()))


def copy_wheel_and_reqs(app, exception):
  if exception or not hasattr(app, "builder"):
    return

  here = Path(__file__).parent
  root_dir = here.parents[2]
  dist_dir = root_dir / "dist"
  static_dst = Path(app.builder.outdir) / "_static"
  static_dst.mkdir(exist_ok=True, parents=True)

  # Req
  reqs_file = root_dir / "requirements.txt"
  if reqs_file.exists():
    shutil.copy2(reqs_file, static_dst / "requirements.txt")

  # Wheel
  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      target_file = static_dst / latest.name
      if not target_file.exists() or target_file.stat().st_mtime < latest.stat().st_mtime:
        shutil.copy2(latest, target_file)
        print(f"📦 [WASM] Copied {latest.name} to _static/")
