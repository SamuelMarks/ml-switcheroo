import os
import shutil
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util.fileutil import copy_asset


class SwitcherooDemo(Directive):
  has_content = True

  def run(self):
    # 1. Locate the Wheel
    root_dir = Path(__file__).parents[3]
    dist_dir = root_dir / "dist"

    wheel_name = ""
    if dist_dir.exists():
      wheels = list(dist_dir.glob("*.whl"))
      if wheels:
        latest = sorted(wheels, key=os.path.getmtime)[-1]
        wheel_name = latest.name

    # Framework Options
    opts = """ 
        <option value="torch" selected>PyTorch</option>
        <option value="jax">JAX / Flax</option>
        <option value="numpy">NumPy</option>
        <option value="tensorflow">TensorFlow</option>
        <option value="mlx">Apple MLX</option>
        """
    opts_tgt = opts.replace("selected", "").replace('value="jax"', 'value="jax" selected')

    html = f""" 
        <div id="switcheroo-wasm-root" class="switcheroo-material-card" data-wheel="{wheel_name}">
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
                            {opts} 
                        </select>
                    </div>

                    <button id="btn-swap" class="swap-icon-btn" title="Swap Languages">⇄</button>

                    <div class="select-wrapper">
                        <select id="select-tgt" class="material-select">
                            {opts_tgt} 
                        </select>
                    </div>
                </div>

                <!-- NEW: Example Selector Toolbar -->
                <div class="example-toolbar">
                    <span class="label">Load Example:</span>
                    <select id="select-example" class="material-select-sm">
                        <option value="" disabled selected>-- Select a Pattern --</option>
                        <!-- Populated by JS -->
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
                    <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                </div>
            </div>

            <!-- Moved Console Group OUTSIDE demo-interface so it can be shown standalone on error -->
            <div class="console-group">
                <div class="editor-label">Engine Logs</div>
                <pre id="console-output">Waiting for engine...</pre>
            </div>
        </div>
        """
    node = nodes.raw("", html, format="html")
    return [node]


def setup(app):
  app.add_directive("switcheroo_demo", SwitcherooDemo)

  # -- Feature 0: Syntax Highlighting dependencies --
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")

  app.add_css_file("switcheroo_demo.css")
  app.add_js_file("switcheroo_demo.js")
  app.connect("build-finished", copy_assets_and_wheel)
  return {"version": "0.5", "parallel_read_safe": True, "parallel_write_safe": True}


def copy_assets_and_wheel(app, exception):
  if exception:
    return
  here = Path(__file__).parent
  root_dir = here.parents[2]
  static_src = here / "static"
  static_dst = Path(app.builder.outdir) / "_static"
  dist_dir = root_dir / "dist"

  # 1. Copy JS/CSS
  for f in static_src.glob("*"):
    copy_asset(str(f), str(static_dst))

    # 2. Copy requirements.txt explicitly so JS can Fetch it
  # This fulfills "install dependencies from the txt files"
  reqs_file = root_dir / "requirements.txt"
  if reqs_file.exists():
    shutil.copy2(reqs_file, static_dst / "requirements.txt")
    print("📄 [WASM] Copied requirements.txt to _static/")

    # 3. Copy Wheel
  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      shutil.copy2(latest, static_dst / latest.name)
      print(f"📦 [WASM] Copied {latest.name} to _static/")
