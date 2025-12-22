"""
Sphinx Extension for ML-Switcheroo WASM Demo.

This module provides the ``switcheroo_demo`` directive, which embeds a client-side
transpiler interface into the documentation. It configures the WebAssembly environment
to load the package directly from the "latest" GitHub Release, ensuring the WASM
demo always pulls the most recent build artifacts.

Features:
- **Remote Loading**: Links to the ``latest`` .whl asset on GitHub Releases.
- **Dynamic Registry**: Introspects installed adapters to populate UI dropdowns.
- **Tier Filtering**: Injects metadata enabling the frontend to filter incompatible conversions.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docutils import nodes
from docutils.parsers.rst import Directive

from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo import __version__  # Used to compute wheel filename

# Type alias for structure: { "parent_fw": [{"key": "child_fw", "label": "Display Name"}] }
HierarchyMap = Dict[str, List[Dict[str, str]]]

class SwitcherooDemo(Directive):
    """
    Sphinx Directive to embed the interactive WASM demo.

    Generates the HTML shell required by ``switcheroo_demo.js``, including
    data attributes for the remote wheel URL and injected JSON payloads
    for examples and compatibility logic.

    Usage:
        .. switcheroo_demo::
    """

    has_content = True

    def run(self) -> List[nodes.raw]:
        """
        Main execution entry point for the directive.

        It performs the following steps:
        1. Constants definition (GitHub Repo/User).
        2. Construction of the remote "latest" release URL using the current package version.
        3. Introspection of the Python framework registry.
        4. Generation of the HTML block with populated options.

        Returns:
            List[nodes.raw]: A list containing the raw HTML node to be rendered.
        """
        # 1. Configuration
        github_user = "SamuelMarks"
        github_repo = "ml-switcheroo"

        # 2. Construct wheel filename dynamically from package version
        # e.g., ml_switcheroo-0.0.1-py3-none-any.whl
        wheel_filename = f"ml_switcheroo-{__version__}-py3-none-any.whl"

        # 3. Construct URL pointing to the 'latest' tag
        remote_url = (
            f"https://github.com/{github_user}/{github_repo}/releases/download/latest/{wheel_filename}"
        )

        # 4. Build Hierarchy & Examples & Tier Metadata
        hierarchy, examples_json, tier_metadata_json = self._scan_registry()

        # 5. Generate HTML Blocks
        primary_opts = self._render_primary_options(hierarchy)
        src_flavours_html = self._render_flavour_dropdown("src", hierarchy)
        tgt_flavours_html = self._render_flavour_dropdown("tgt", hierarchy)

        opts_src = primary_opts.replace('value="torch"', 'value="torch" selected')
        opts_tgt = primary_opts.replace('value="jax"', 'value="jax" selected')

        html = f"""
        <div id="switcheroo-wasm-root" 
             class="switcheroo-material-card" 
             data-wheel-url="{remote_url}">
             
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

    def _scan_registry(self) -> Tuple[HierarchyMap, str, str]:
        """
        Scans registered adapters to build hierarchy, examples, and tier metadata.
        Returns JSON strings for frontend injection.
        """
        fws = available_frameworks()
        hierarchy: HierarchyMap = defaultdict(list)
        tier_metadata: Dict[str, List[str]] = {}
        roots = set()

        for key in fws:
            adapter = get_adapter(key)
            if not adapter:
                continue

            label = getattr(adapter, "display_name", key.capitalize())
            parent = getattr(adapter, "inherits_from", None)

            tiers = []
            if hasattr(adapter, "supported_tiers") and adapter.supported_tiers:
                tiers = [t.value for t in adapter.supported_tiers]
            else:
                tiers = ["array", "neural", "extras"]
            tier_metadata[key] = tiers

            if parent:
                hierarchy[parent].append({"key": key, "label": label})
            else:
                roots.add(key)

        examples = {}
        main_priority = ["torch", "jax", "tensorflow", "numpy"]
        sorted_roots = sorted(list(roots), key=lambda x: (main_priority.index(x) if x in main_priority else 99, x))

        final_hierarchy = {
            root: sorted(hierarchy.get(root, []), key=lambda x: x["label"])
            for root in sorted_roots
        }

        for key in fws:
            adapter = get_adapter(key)
            if hasattr(adapter, "get_tiered_examples"):
                tiers_dict = adapter.get_tiered_examples()
                parent_key = getattr(adapter, "inherits_from", None)

                for tier_name, code in tiers_dict.items():
                    uid = f"{key}_{tier_name}"

                    if parent_key:
                        src_fw = parent_key
                        src_flavour = key
                    else:
                        src_fw = key
                        src_flavour = None

                    req_tier = "extras"
                    if "math" in tier_name: req_tier = "array"
                    elif "neural" in tier_name: req_tier = "neural"

                    clean_tier_name = tier_name.replace("tier", "")
                    clean_label = (
                        clean_tier_name.split("_")[-1].capitalize()
                        if "_" in clean_tier_name
                        else clean_tier_name.capitalize()
                    )

                    display_fw = getattr(adapter, "display_name", key.title())
                    label = f"{display_fw}: {clean_label}"

                    tgt_fw = "jax"
                    if "jax" in src_fw:
                        tgt_fw = "torch"

                    examples[uid] = {
                        "label": label,
                        "srcFw": src_fw,
                        "srcFlavour": src_flavour,
                        "tgtFw": tgt_fw,
                        "tgtFlavour": None,
                        "code": code,
                        "requiredTier": req_tier,
                    }

        return final_hierarchy, json.dumps(examples), json.dumps(tier_metadata)

    def _render_primary_options(self, hierarchy: HierarchyMap) -> str:
        html = []
        for root in hierarchy.keys():
            adapter = get_adapter(root)
            label = getattr(adapter, "display_name", root.capitalize())
            html.append(f'<option value="{root}">{label}</option>')
        return "\n".join(html)

    def _render_flavour_dropdown(self, side: str, hierarchy: HierarchyMap) -> str:
        children = hierarchy.get("jax", [])
        if not children:
            return ""

        default_child = "flax_nnx"
        opts = []
        for child in children:
            sel = "selected" if child["key"] == default_child else ""
            opts.append(
                f'<option value="{child["key"]}" {sel}>{child["label"]}</option>'
            )

        return f"""
        <select id="{side}-flavour" class="material-select-sm" style="background:#f0f4c3;">
            {"".join(opts)}
        </select>
        """

def setup(app: Any) -> Dict[str, Any]:
    """Sphinx Setup Hook."""
    app.add_directive("switcheroo_demo", SwitcherooDemo)
    app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
    app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
    app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")
    app.add_css_file("switcheroo_demo.css")
    app.add_css_file("trace_graph.css")
    app.add_js_file("trace_render.js")
    app.add_js_file("switcheroo_demo.js")

    app.connect("builder-inited", add_static_path)
    app.connect("build-finished", copy_deps_only)

    return {"version": "0.9.8", "parallel_read_safe": True, "parallel_write_safe": True}

def add_static_path(app: Any) -> None:
    static_path = Path(__file__).parent / "static"
    if static_path.exists() and hasattr(app, "config"):
        app.config.html_static_path.append(str(static_path.resolve()))

def copy_deps_only(app: Any, exception: Optional[Exception]) -> None:
    """Copies requirements.txt to _static for Micropip."""
    if exception or not hasattr(app, "builder") or app.builder.format != "html":
        return

    here = Path(__file__).parent
    root_dir = here.parents[2]
    static_dst = Path(app.builder.outdir) / "_static"
    static_dst.mkdir(exist_ok=True, parents=True)

    reqs_file = root_dir / "requirements.txt"
    if reqs_file.exists():
        shutil.copy2(reqs_file, static_dst / "requirements.txt")
