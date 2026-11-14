"""HTML Rendering logic for the WASM demo.

This module provides utility functions to construct and render the HTML interface
and dropdown elements for the live ML Switcheroo WebAssembly transpiler demo.
"""

import os
from collections import defaultdict
from pathlib import Path

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
  "ir": "Level 3: Standard IR",
  "ml_switcheroo_ir": "Level 3: Standard IR",
  # Level 4 - Updated to include NVIDIA_SASS/RDNA
  "nvidia_sass": "Level 4: ASM",
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
  """Construct the full HTML block for the switcheroo demo.

  Args:
      hierarchy: A dictionary mapping root frameworks to their list of flavours.
      examples_json: A JSON string containing preloaded code examples.
      tier_metadata_json: A JSON string containing framework tier metadata.

  Returns:
      The fully rendered HTML block as a string.
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

  def_source = "torch" if "torch" in available_roots else priority_order[0] if priority_order else "source_placeholder"

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

                <!-- Main Interface -->
                <div id="main-interface-content" class="active">
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

                        <label class="on-device-field" title="Choose between 1:1 CST whitespace/comment preservation (default) and multi-pass compiler optimization">
                            <span>Mode:</span>
                            <select id="select-conversion-mode" class="material-select-sm">
                                <option value="preserve_trivia" selected>1:1 Preserve Whitespace &amp; Trivia (Default)</option>
                                <option value="optimized">Optimised Mode (Multi-Pass: Constant Folding, CSE, DCE, Fusion)</option>
                            </select>
                        </label>

                        <label id="opt-level-wrapper" class="on-device-field" style="display:none;" title="Multi-pass optimization level for ml-switcheroo-compiler PassManager">
                            <span>Opt Level:</span>
                            <select id="select-opt-level" class="material-select-sm">
                                <option value="O0">O0 (Canonicalization Only)</option>
                                <option value="O1">O1 (Constant Folding + DCE)</option>
                                <option value="O2" selected>O2 (Default: CSE + Fusion + Scheduling)</option>
                                <option value="O3">O3 (Aggressive: Mixed Precision + Tiling + Vectorization + SPMD)</option>
                            </select>
                        </label>

                        <label class="md-switch" title="Execute forward and backward passes on-device in-browser via zero-* and ml-switcheroo-compiler">
                            <input type="checkbox" id="chk-on-device-exec">
                            <div class="md-switch-track">
                                <div class="md-switch-thumb"></div>
                            </div>
                            <span class="md-switch-text">⚡ Run Fwd &amp; Bwd On-Device (<code>zero-*</code>)</span>
                        </label>

                        <label class="md-switch" title="Enable automatic distributed sharding annotations and optimizations (e.g. QKV fusions, jax.sharding)">
                            <input type="checkbox" id="chk-enable-sharding">
                            <div class="md-switch-track">
                                <div class="md-switch-thumb"></div>
                            </div>
                            <span class="md-switch-text">Sharding</span>
                        </label>

                        <button id="btn-convert" class="md-btn md-btn-accent" disabled>Run Translation</button>
                    </div>

                    <!-- On-Device Forward & Backward Execution Config Panel -->
                    <div id="on-device-config-bar" class="on-device-config-bar" style="display:none;">
                        <div class="on-device-config-row">
                            <label class="on-device-field">
                                <span>Execution Backend:</span>
                                <select id="select-exec-backend" class="material-select-sm">
                                    <optgroup label="Web Native (In-Browser)">
                                        <option value="webgpu" selected>WebGPU (Web Native — Default)</option>
                                        <option value="wasm_simd">WASM SIMD (128-bit WebAssembly)</option>
                                        <option value="webgl">WebGL2 (Fragment Shader)</option>
                                        <option value="pyodide">Pyodide (zero-* + Compiler Evaluator)</option>
                                        <option value="auto">Auto (WebGPU &rarr; WASM &rarr; Pyodide)</option>
                                        <option value="onnx">Edge ONNX</option>
                                        <option value="stablehlo">Edge StableHLO</option>
                                    </optgroup>
                                    <optgroup label="Hardware Accelerators (ml-switcheroo-compiler)">
                                        <option value="cuda">NVIDIA CUDA / PTX</option>
                                        <option value="rocm">AMD ROCm / HIP</option>
                                        <option value="metal">Apple Metal (MSL)</option>
                                        <option value="llvm_cpp">LLVM C++</option>
                                    </optgroup>
                                    <optgroup label="Array &amp; Framework Runtimes (ml-switcheroo-compiler)">
                                        <option value="jax">JAX</option>
                                        <option value="pytorch">PyTorch</option>
                                        <option value="mlx">Apple MLX</option>
                                        <option value="tensorflow">TensorFlow</option>
                                        <option value="keras">Keras 3</option>
                                        <option value="numpy">NumPy</option>
                                        <option value="cupy">CuPy</option>
                                        <option value="dask">Dask Array</option>
                                        <option value="numba">Numba JIT</option>
                                        <option value="dpnp">Intel DPNP</option>
                                        <option value="sparse">PyData Sparse (COO)</option>
                                        <option value="awkward">Awkward Array</option>
                                        <option value="pyarrow_compute">PyArrow Compute</option>
                                        <option value="pure_python">Pure Python</option>
                                    </optgroup>
                                </select>
                            </label>
                            <label class="on-device-field">
                                <span>Input Preset:</span>
                                <select id="select-input-preset" class="material-select-sm">
                                    <option value="" selected>Auto-infer from AST</option>
                                    <option value="2, 16">MLP Batch: (2, 16)</option>
                                    <option value="2, 3, 8, 8">CNN NCHW: (2, 3, 8, 8)</option>
                                    <option value="2, 8, 32">Sequence Tokens: (2, 8, 32)</option>
                                </select>
                            </label>
                            <label class="on-device-field">
                                <span>Input Shape:</span>
                                <input type="text" id="input-tensor-shape" class="material-select-sm" placeholder="Auto (e.g. 2, 16)" value="">
                            </label>
                            <label class="on-device-field">
                                <span>DType:</span>
                                <select id="select-input-dtype" class="material-select-sm">
                                    <option value="float32" selected>float32</option>
                                    <option value="float16">float16</option>
                                    <option value="int32">int32</option>
                                </select>
                            </label>
                            <label class="on-device-field">
                                <span>Seed:</span>
                                <input type="number" id="input-random-seed" class="material-select-sm" value="42" style="width:64px;">
                            </label>
                            <label class="on-device-field">
                                <span>Loss Reduction:</span>
                                <select id="select-loss-reduction" class="material-select-sm">
                                    <option value="sum" selected>sum</option>
                                    <option value="mean">mean</option>
                                    <option value="l2">l2</option>
                                </select>
                            </label>
                        </div>
                        <div class="on-device-config-row on-device-toggles">
                            <label title="Inject inline forward shape and backward gradient L2 norm comments into target code">
                                <input type="checkbox" id="chk-inject-annotations" checked> Inline Shape/Grad Comments
                            </label>
                            <label title="Inject executable runtime shape assertions (e.g. chex.assert_shape) into target code">
                                <input type="checkbox" id="chk-inject-assertions"> Emit Shape Assertions
                            </label>
                            <label title="Synthesize explicit target-framework backward/VJP function from the autodiff LogicalGraph">
                                <input type="checkbox" id="chk-emit-bwd-code"> Emit Target Backward/VJP Code
                            </label>
                        </div>
                    </div>

                    <!-- Output Tabs -->
                    <div class="output-tabs">
                        <input type="radio" name="wm-tabs" id="tab-trace" checked>
                        <label for="tab-trace" class="tab-label"><span class="tab-icon">⏱️</span> Time Travel</label>
                        <input type="radio" name="wm-tabs" id="tab-runtime">
                        <label for="tab-runtime" id="label-tab-runtime" class="tab-label"><span class="tab-icon">🔬</span> Runtime &amp; Gradients</label>
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
                        <!-- Content: Runtime & Gradients -->
                        <div class="tab-panel" id="panel-runtime">
                            <div id="runtime-summary-badges" class="runtime-summary-badges">
                                <em style="color:#999">Enable "⚡ Run Fwd &amp; Bwd On-Device" and click Run Translation to inspect forward shapes and backward gradients.</em>
                            </div>
                            <div id="runtime-impasse-card" class="runtime-impasse-card" style="display:none;"></div>
                            <div id="runtime-telemetry-table" class="runtime-telemetry-table"></div>
                            <div id="runtime-param-grads-table" class="runtime-param-grads-table"></div>
                            <div id="runtime-ir-inspector" class="runtime-ir-inspector" style="display:none;">
                                <div class="runtime-ir-inspector-header">
                                    <h4>Shared <code>ml-switcheroo-ir</code> &amp; Autodiff Graph Inspector</h4>
                                    <select id="select-ir-inspector-view" class="material-select-sm">
                                        <option value="forward_ir_python" selected>Forward LogicalGraph (ml-switcheroo-ir Python)</option>
                                        <option value="backward_ir_python">Backward VJP LogicalGraph (ml-switcheroo-ir Python)</option>
                                        <option value="synthesized_backward_code">Synthesized Target Backward/VJP Code</option>
                                        <option value="backend_kernel">Emitted Backend Kernel (WebGPU / WASM / CUDA / Metal / ...)</option>
                                        <option value="backward_mermaid">Backward Autodiff Mermaid DAG</option>
                                    </select>
                                </div>
                                <pre id="runtime-ir-inspector-code" class="runtime-ir-inspector-code"></pre>
                            </div>
                        </div>
                        <!-- Content: Console -->
                        <div class="tab-panel" id="panel-console">
                            <pre id="console-output">Waiting for engine...</pre>
                        </div>
                        <!-- Content: Mermaid -->
                        <div class="tab-panel" id="panel-mermaid">
                            <div id="ast-mermaid-target" style="text-align:center; overflow-x:auto;">
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
        </div>
        """


def _render_primary_options(hierarchy: HierarchyMap) -> str:
  """Render the top-level <option> elements for root frameworks.

  Organizes frameworks into <optgroup> categories based on their semantic level.

  Args:
      hierarchy: A dictionary mapping root frameworks to their list of flavours.

  Returns:
      An HTML string containing grouped option elements.
  """
  # Organizes roots into buckets
  grouped: dict[str, list[str]] = defaultdict(list)

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

    html_parts.append(f'<optgroup label="{group_name}">')
    for root in members:
      adapter = get_adapter(root)
      label = adapter.display_name if adapter is not None else root.capitalize()
      html_parts.append(f'<option value="{root}">{label}</option>')
    html_parts.append("</optgroup>")

  return "\n".join(html_parts)


def _render_flavour_dropdown(side: str, hierarchy: HierarchyMap, active_root: str) -> str:
  """Render the secondary dropdown for Framework Flavours.

  Renders options for all frameworks, using data-parent to allow JS to filter.

  Args:
      side: The interface side, typically "src" (source) or "tgt" (target).
      hierarchy: A dictionary mapping root frameworks to their list of flavours.
      active_root: The active framework root whose flavours are initially visible.

  Returns:
      An HTML string containing the secondary dropdown element.
  """
  children = hierarchy.get(active_root, [])

  style = "display:inline-block;" if children else "display:none;"
  style += " background:#f0f4c3;"

  opts = []

  has_any_children = any(len(v) > 0 for v in hierarchy.values())
  if not has_any_children:
    opts.append('<option value="" disabled selected data-parent="">No Flavours</option>')
  else:
    for root, root_children in hierarchy.items():
      if root_children:
        for i, child in enumerate(root_children):
          sel = "selected" if root == active_root and i == 0 else ""
          hidden = "display:none;" if root != active_root else ""
          opts.append(
            f'<option value="{child["key"]}" data-parent="{root}" {sel} style="{hidden}">{child["label"]}</option>'
          )

  return f"""
        <select id="{side}-flavour" class="material-select-sm" style="{style}">
            {"".join(opts)}
        </select>
        """
