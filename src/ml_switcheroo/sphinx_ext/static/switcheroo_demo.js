/**
 * @fileoverview Client-side controller for the ml-switcheroo WASM & On-Device interactive demo.
 *
 * Capabilities:
 * - Pyodide Engine Management (WASM) + `wheels.json` (`zero-*` & `ml-switcheroo-compiler` installation).
 * - On-Device Forward & Backward Pass execution (Pyodide `zero-*`, WebGPU WGSL, and WASM SIMD).
 * - CodeMirror Editor sync, Trace Graph Visualization, Weight Script Generation, TikZ/HTML rendering, and Time Travel.
 */

// Global State
let pyodide = null;
let srcEditor = null;
let tgtEditor = null;
let weightEditor = null;
let EXAMPLES = {};
let FW_TIERS = {};
let tikzLoaded = false;
let timeTravel = null;
let latestRuntimeTrace = null;

const DEFAULT_TIKZJAX_URL = "https://tikzjax.com/v1/tikzjax.js";

/**
 * Python bridge script executed inside Pyodide to run `ASTEngine`, on-device
 * forward/backward tracing, and weight script generation.
 */
const PYTHON_BRIDGE = `
import json
import traceback
import pathlib
import sys
from pydantic import BaseModel
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import (
    ConversionMode,
    ExecutionBackend,
    LossReduction,
    OptimizationLevel,
    RuntimeConfig,
    parse_input_shapes_string,
)
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console

class BridgePayload(BaseModel):
    source_code: str = ""
    src_fw: str = "torch"
    tgt_fw: str = "jax"
    src_flavour: str | None = None
    tgt_flavour: str | None = None
    conversion_mode: str = "preserve_trivia"
    opt_level: str = "O2"
    strict_mode: bool = False
    enable_sharding: bool = False
    enable_graph_opt: bool = False
    enable_import_fixer: bool = True
    enable_on_device: bool = False
    exec_backend: str = "webgpu"
    input_shape: str = ""
    input_dtype: str = "float32"
    random_seed: int = 42
    loss_reduction: str = "sum"
    inject_annotations: bool = True
    inject_assertions: bool = False
    emit_bwd_code: bool = False
    observed_shapes: dict[str, list[int]] | None = None

process_log = Console(record=True, force_terminal=False, width=120)
set_console(process_log)

response = {}

try:
    semantics = SemanticsManager()
    payload = BridgePayload.model_validate_json(js_bridge_payload_json)

    src_flavour = payload.src_flavour if payload.src_flavour else None
    tgt_flavour = payload.tgt_flavour if payload.tgt_flavour else None

    real_source = src_flavour if src_flavour else payload.src_fw
    real_target = tgt_flavour if tgt_flavour else payload.tgt_fw

    raw_mode = payload.conversion_mode or 'preserve_trivia'
    raw_opt_level = payload.opt_level or 'O2'
    opt_graph = (raw_mode == 'optimized') or payload.enable_graph_opt

    config = RuntimeConfig(
        source_framework=payload.src_fw,
        target_framework=payload.tgt_fw,
        source_flavour=src_flavour,
        target_flavour=tgt_flavour,
        strict_mode=payload.strict_mode,
        mode=ConversionMode(raw_mode),
        opt_level=OptimizationLevel(raw_opt_level),
        enable_graph_optimization=opt_graph,
        enable_import_fixer=payload.enable_import_fixer,
        enable_sharding=payload.enable_sharding,
        enable_on_device_execution=payload.enable_on_device,
        execution_backend=ExecutionBackend(payload.exec_backend or 'webgpu'),
        input_shapes=parse_input_shapes_string(payload.input_shape),
        input_dtype=payload.input_dtype or 'float32',
        random_seed=payload.random_seed,
        loss_reduction=LossReduction(payload.loss_reduction or 'sum'),
        inject_runtime_annotations=payload.inject_annotations,
        inject_shape_assertions=payload.inject_assertions,
        emit_backward_code=payload.emit_bwd_code,
        observed_runtime_shapes=payload.observed_shapes,
    )

    print(f"Running AST Engine ({real_source} -> {real_target})...")
    engine = ASTEngine(semantics=semantics, config=config)
    result = engine.run(payload.source_code)

    if result.success:
        print("Transpilation successful.")
    else:
        print("Transpilation failed with errors.")

    weight_script_code = ""
    ir_targets = {'mlir', 'stablehlo', 'tikz', 'latex_dsl', 'html'}

    if result.success and real_target not in ir_targets:
        try:
            f_src = pathlib.Path("/tmp/src_model.py")
            f_src.write_text(payload.source_code, encoding="utf-8")
            f_out = pathlib.Path("/tmp/migration_script.py")

            from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator

            wgen = WeightScriptGenerator(semantics, config)
            success = wgen.generate(f_src, f_out)

            if success and f_out.exists():
                weight_script_code = f_out.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Weight Gen Error: {e}")

    response = {
        "code": result.code,
        "logs": process_log.export_text(),
        "is_success": result.success,
        "errors": result.errors,
        "trace_events": result.trace_events,
        "weight_script": weight_script_code,
        "runtime_trace": result.runtime_trace.model_dump() if result.runtime_trace else None,
    }
except Exception as e:
    err_str = traceback.format_exc()
    response = {
        "code": "",
        "logs": f"{process_log.export_text()}
CRITICAL PYTHON ERROR: {str(e)}
{err_str}",
        "is_success": False,
        "errors": [str(e)],
        "trace_events": [],
        "weight_script": "",
        "runtime_trace": None,
    }

json_output = json.dumps(response)
`;

/**
 * Resolves a relative `_static/` asset filename against Sphinx `URL_ROOT`.
 *
 * @param {string} filename - Asset filename inside `_static/`.
 * @returns {string} Resolved URL path to the static asset.
 */
function getStaticPath(filename) {
    if (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) {
        const root = window.DOCUMENTATION_OPTIONS.URL_ROOT;
        const safeRoot = root.endsWith('/') ? root : root + '/';
        return `${safeRoot}_static/${filename}`;
    }
    return `_static/${filename}`;
}

/**
 * Initializes the Pyodide WASM environment and installs `ml_switcheroo`,
 * `ml_switcheroo_compiler`, and bundled `zero-*` wheels from `_static/wheels.json`.
 *
 * @returns {Promise<void>} Resolves when engine initialization finishes.
 */
async function initEngine() {
    const rootEl = document.getElementById("switcheroo-wasm-root");
    const statusEl = document.getElementById("engine-status");
    const btnLoad = document.getElementById("btn-load-engine");
    const splashEl = document.getElementById("demo-splash");
    const interfaceEl = document.getElementById("demo-interface");
    const wheelName = (rootEl && rootEl.dataset.wheel) ? rootEl.dataset.wheel : "ml_switcheroo-0.0.6-py3-none-any.whl";

    statusEl.innerText = "Downloading...";
    statusEl.className = "status-badge status-loading";
    btnLoad.disabled = true;

    try {
        if (!window.loadPyodide) await loadScript("https://cdn.jsdelivr.net/pyodide/v314.0.3/full/pyodide.js");
        if (!pyodide) pyodide = await loadPyodide();

        if (!tikzLoaded) {
            let urlToLoad = DEFAULT_TIKZJAX_URL;
            if (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) {
                const root = window.DOCUMENTATION_OPTIONS.URL_ROOT;
                const safeRoot = root.endsWith('/') ? root : root + '/';
                urlToLoad = `${safeRoot}_static/tikzjax/tikzjax.js`;
            } else if (window.TIKZJAX_URL) {
                urlToLoad = window.TIKZJAX_URL;
            }

            try {
                await loadScript(urlToLoad);
                tikzLoaded = true;
            } catch (_tiErr) {
                try {
                    await loadScript(DEFAULT_TIKZJAX_URL);
                    tikzLoaded = true;
                } catch (_cdnErr) {
                    tikzLoaded = false;
                }
            }
        }

        const isInstalled = pyodide.runPython(`
import importlib.util
importlib.util.find_spec("ml_switcheroo") is not None
        `);

        if (!isInstalled) {
            statusEl.innerText = "Fetching Requirements...";
            await pyodide.loadPackage("micropip");
            const micropip = pyodide.pyimport("micropip");

            try {
                const reqRes = await fetch(getStaticPath("requirements.txt"));
                if (reqRes.ok) {
                    const reqText = await reqRes.text();
                    const reqs = reqText.split('
')
                        .map(l => l.trim())
                        .filter(l => l && !l.startsWith('#') && !l.includes('git+'))
                        .map(l => {
                            if (l.includes(' @ ')) {
                                const parts = l.split(' @ ');
                                const spec = parts[1].trim().split(';')[0].trim();
                                if (!spec.startsWith('http')) {
                                    const url = new URL(getStaticPath(spec), window.location.href).href;
                                    return `${parts[0]} @ ${url}`;
                                } else if (spec.includes('github.com') && spec.includes('.whl')) {
                                    const filename = spec.split('/').pop().split('?')[0];
                                    const url = new URL(getStaticPath(filename), window.location.href).href;
                                    return `${parts[0]} @ ${url}`;
                                }
                            }
                            return l;
                        });
                    await micropip.install("numpy");
                    for (const req of reqs) {
                        try {
                            if (req.includes(".whl") || req.includes(" @ ")) {
                                await micropip.install(req, false, false);
                            } else {
                                await micropip.install(req);
                            }
                        } catch (_reqErr) {
                            try {
                                await micropip.install(req, false, false);
                            } catch (fallbackErr) {
                                console.warn("Failed to install requirement:", req, fallbackErr);
                            }
                        }
                    }
                } else {
                    await micropip.install(["numpy", "pydantic", "rich", "libcst"]);
                }
            } catch (_e) {
                await micropip.install(["numpy", "pydantic", "rich", "libcst"]);
            }

            // Install bundled zero-* and compiler wheels from wheels.json if present
            try {
                const wheelsRes = await fetch(getStaticPath("wheels.json"));
                if (wheelsRes.ok) {
                    const manifest = await wheelsRes.json();
                    const wheelList = Array.isArray(manifest.wheels) ? manifest.wheels : [];
                    statusEl.innerText = "Installing zero-* Stack...";
                    for (const whlFile of wheelList) {
                        const whlUrl = new URL(getStaticPath(whlFile), window.location.href).href;
                        try {
                            await micropip.install(whlUrl, false, false);
                        } catch (_wErr) {
                            console.warn("Optional wheel install skipped:", whlFile);
                        }
                    }
                }
            } catch (_manifestErr) {
                // wheels.json is optional when running minimal docs build
            }

            statusEl.innerText = "Installing Engine...";
            const wheelUrl = new URL(getStaticPath(wheelName), window.location.href).href;
            try {
                await micropip.install(wheelUrl, false, false);
            } catch (_instErr) {
                await micropip.install(wheelUrl);
            }
        }

        splashEl.style.display = "none";
        interfaceEl.style.display = "block";
        initEditors();

        if (window.SWITCHEROO_PRELOADED_EXAMPLES) EXAMPLES = window.SWITCHEROO_PRELOADED_EXAMPLES;
        if (window.SWITCHEROO_FRAMEWORK_TIERS) FW_TIERS = window.SWITCHEROO_FRAMEWORK_TIERS;

        initExampleSelector();
        initFrameworkListeners();
        initOnDeviceControls();
        updateRenderTabVisibility();

        if (window.TimeTravelController) {
            timeTravel = new window.TimeTravelController({
                onUpdateCode: (code) => {
                    if (tgtEditor) {
                        tgtEditor.setValue(code);
                        let mode = "python";
                        if (document.getElementById("select-tgt").value === 'html') mode = "htmlmixed";
                        tgtEditor.setOption("mode", mode);
                    }
                },
                onUpdateGraph: (mermaidStr) => renderMermaid(mermaidStr)
            });
        }

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";
        document.getElementById("btn-convert").disabled = false;

        if (typeof mermaid !== "undefined") {
            mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });
        }

    } catch (err) {
        console.error(err);
        statusEl.innerText = "Load Failed";
        statusEl.className = "status-badge status-error";
        btnLoad.disabled = false;
    }
}

/**
 * Initializes CodeMirror instances for Source, Target, and Weights.
 *
 * @returns {void}
 */
function initEditors() {
    if (srcEditor) {
        srcEditor.refresh();
        if (tgtEditor) tgtEditor.refresh();
        if (weightEditor) weightEditor.refresh();
        return;
    }
    const opts = { mode: "python", lineNumbers: true, viewportMargin: Infinity, theme: "default" };
    srcEditor = CodeMirror.fromTextArea(document.getElementById("code-source"), { ...opts, readOnly: false });
    tgtEditor = CodeMirror.fromTextArea(document.getElementById("code-target"), { ...opts, readOnly: true });

    const weightEl = document.getElementById("code-weights");
    if (weightEl) {
        weightEditor = CodeMirror.fromTextArea(weightEl, { ...opts, readOnly: true });
    }
}

/**
 * Populates the Example dropdown from the loaded registry.
 *
 * @returns {void}
 */
function initExampleSelector() {
    const sel = document.getElementById("select-example");
    if (!sel) return;
    sel.innerHTML = '<option value="" disabled>-- Select a Pattern --</option>';

    const sortedKeys = Object.keys(EXAMPLES).sort();
    for (const key of sortedKeys) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = EXAMPLES[key].label;
        sel.appendChild(opt);
    }
    sel.onchange = (e) => loadExample(e.target.value);

    if (sortedKeys.length > 0) {
       const preferred = sortedKeys.find(k => k.includes("torch") && k.includes("neural"));
       const defKey = preferred || sortedKeys[0];
       sel.value = defKey;
       loadExample(defKey);
    }
}

/**
 * Attaches event listeners for the On-Device execution toggle, Conversion Mode selector,
 * Optimization Level selector, input preset selector, and IR Inspector view selector.
 *
 * @returns {void}
 */
function initOnDeviceControls() {
    const chkOnDevice = document.getElementById("chk-on-device-exec");
    const configBar = document.getElementById("on-device-config-bar");
    const presetSel = document.getElementById("select-input-preset");
    const shapeInput = document.getElementById("input-tensor-shape");
    const modeSel = document.getElementById("select-conversion-mode");
    const optWrapper = document.getElementById("opt-level-wrapper");
    const chkOptGraph = document.getElementById("chk-opt-graph");
    const inspectorSel = document.getElementById("select-ir-inspector-view");

    if (chkOnDevice && configBar) {
        chkOnDevice.addEventListener("change", () => {
            configBar.style.display = chkOnDevice.checked ? "block" : "none";
        });
    }
    if (presetSel && shapeInput) {
        presetSel.addEventListener("change", () => {
            shapeInput.value = presetSel.value;
        });
    }
    if (modeSel && optWrapper) {
        modeSel.addEventListener("change", () => {
            const isOpt = modeSel.value === "optimized";
            optWrapper.style.display = isOpt ? "inline-flex" : "none";
            if (chkOptGraph) chkOptGraph.checked = isOpt;
        });
    }
    if (chkOptGraph && modeSel && optWrapper) {
        chkOptGraph.addEventListener("change", () => {
            modeSel.value = chkOptGraph.checked ? "optimized" : "preserve_trivia";
            optWrapper.style.display = chkOptGraph.checked ? "inline-flex" : "none";
        });
    }
    if (inspectorSel) {
        inspectorSel.addEventListener("change", () => {
            updateIrInspectorView(latestRuntimeTrace);
        });
    }
}

/**
 * Attaches event listeners for framework dropdowns and tab switching.
 *
 * @returns {void}
 */
function initFrameworkListeners() {
    const handler = (type) => {
        const sel = document.getElementById(`select-${type}`);
        const region = document.getElementById(`${type}-flavour-region`);
        const flavourSel = document.getElementById(`${type}-flavour`);

        if (region && flavourSel) {
            const rootValue = sel.value;
            let hasVisibleFlavours = false;
            let firstVisibleFlavour = null;

            for (let i = 0; i < flavourSel.options.length; i++) {
                const opt = flavourSel.options[i];
                if (opt.getAttribute('data-parent') === rootValue) {
                    opt.style.display = '';
                    hasVisibleFlavours = true;
                    if (!firstVisibleFlavour) firstVisibleFlavour = opt;
                } else if (opt.getAttribute('data-parent')) {
                    opt.style.display = 'none';
                }
            }

            if (hasVisibleFlavours) {
                region.style.display = 'inline-block';
                flavourSel.style.display = 'inline-block';

                if (flavourSel.options[flavourSel.selectedIndex].style.display === 'none') {
                     flavourSel.value = firstVisibleFlavour.value;
                }
            } else {
                region.style.display = 'none';
                flavourSel.style.display = 'none';
            }
        }
        updateRenderTabVisibility();
    };

    document.getElementById("select-src").addEventListener("change", () => handler('src'));
    document.getElementById("select-tgt").addEventListener("change", () => handler('tgt'));
    updateRenderTabVisibility();

    document.querySelectorAll('input[name="wm-tabs"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.id === 'tab-weights' && weightEditor) {
                setTimeout(() => weightEditor.refresh(), 20);
            }
        });
    });
}

/**
 * Toggles visibility of the visual Render tab based on selected frameworks.
 *
 * @returns {void}
 */
function updateRenderTabVisibility() {
    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;

    const isVisual = (srcFw === 'tikz' || tgtFw === 'tikz' || srcFw === 'html' || tgtFw === 'html');
    const tabLabel = document.getElementById("label-tab-render");
    const tabInput = document.getElementById("tab-render");

    if (tabLabel) {
        if (isVisual) {
            tabLabel.style.display = "inline-flex";
        } else {
            tabLabel.style.display = "none";
            if (tabInput && tabInput.checked) {
                document.getElementById("tab-trace").checked = true;
            }
        }
    }
}

/**
 * Loads a preconfigured code example into the editor and selects its frameworks.
 *
 * @param {string} key - Example key in `EXAMPLES`.
 * @returns {void}
 */
function loadExample(key) {
    const ex = EXAMPLES[key];
    if (!ex) return;
    srcEditor.setValue(ex.code);
    tgtEditor.setValue("");

    if (weightEditor) weightEditor.setValue("");

    setSelectValue(document.getElementById("select-src"), ex.srcFw);
    const tgtSelect = document.getElementById("select-tgt");
    for (let opt of tgtSelect.options) opt.disabled = false;
    setSelectValue(tgtSelect, ex.tgtFw);

    document.getElementById("select-src").dispatchEvent(new Event('change'));
    document.getElementById("select-tgt").dispatchEvent(new Event('change'));

    if (ex.srcFlavour) setSelectValue(document.getElementById("src-flavour"), ex.srcFlavour);
    if (ex.tgtFlavour) setSelectValue(document.getElementById("tgt-flavour"), ex.tgtFlavour);

    if (typeof ex.enableOnDevice === "boolean") {
        const chkOnDevice = document.getElementById("chk-on-device-exec");
        const configBar = document.getElementById("on-device-config-bar");
        if (chkOnDevice) {
            chkOnDevice.checked = ex.enableOnDevice;
            if (configBar) configBar.style.display = ex.enableOnDevice ? "block" : "none";
        }
    }
    if (ex.inputShape) {
        const shapeInput = document.getElementById("input-tensor-shape");
        if (shapeInput) shapeInput.value = ex.inputShape;
    }
}

/**
 * Sets the selected option of a `<select>` element by value.
 *
 * @param {HTMLSelectElement|null} el - Select element to update.
 * @param {string} val - Option value to select.
 * @returns {void}
 */
function setSelectValue(el, val) {
    if (!el) return;
    for (let i = 0; i < el.options.length; i++) {
        if (el.options[i].value === val) {
            el.selectedIndex = i;
            break;
        }
    }
}

/**
 * Swaps the source and target frameworks, flavours, and editor contents.
 *
 * @returns {void}
 */
function swapContext() {
    const s = document.getElementById("select-src");
    const t = document.getElementById("select-tgt");

    const sf = document.getElementById("src-flavour");
    const tf = document.getElementById("tgt-flavour");
    const sfValue = sf ? sf.value : null;
    const tfValue = tf ? tf.value : null;

    [s.value, t.value] = [t.value, s.value];

    s.dispatchEvent(new Event("change"));
    t.dispatchEvent(new Event("change"));

    if (sf && tf) {
        sf.value = tfValue;
        tf.value = sfValue;
        sf.dispatchEvent(new Event("change"));
        tf.dispatchEvent(new Event("change"));
    }

    const tmp = srcEditor.getValue();
    srcEditor.setValue(tgtEditor.getValue());
    tgtEditor.setValue(tmp);
}

/**
 * Highlights or unhighlights a source editor line when hovering over a trace event.
 *
 * @param {{lineno?: number}} event - Trace event metadata containing `lineno`.
 * @param {boolean} hover - True when mouse enters, false when mouse leaves.
 * @returns {void}
 */
function updateLineHighlight(event, hover) {
    if (!srcEditor || !event.lineno) return;
    const ln = event.lineno - 1;
    const method = hover ? "addLineClass" : "removeLineClass";
    srcEditor[method](ln, "background", "cm-trace-highlight");
}

/**
 * Toggles the Retro CRT / Matrix visual theme on `document.body`.
 *
 * @returns {void}
 */
function toggleRetroMode() {
    document.body.classList.toggle("crt-mode");
}

/**
 * Dynamically loads an external JavaScript file via a `<script>` tag.
 *
 * @param {string} src - Script URL to load.
 * @returns {Promise<Event>} Promise resolving on script load.
 */
function loadScript(src) {
    return new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = src;
        s.async = true;
        s.onload = res;
        s.onerror = rej;
        document.head.appendChild(s);
    });
}

/**
 * Renders a Mermaid graph into the AST visualization tab.
 *
 * @param {string} graphDef - Mermaid graph definition string.
 * @returns {Promise<void>} Resolves when SVG is rendered.
 */
async function renderMermaid(graphDef) {
    const element = document.getElementById('ast-mermaid-target');
    if (!element) return;

    if (!graphDef || graphDef.trim() === '') {
        element.innerHTML = '<em style="color:#999">No graph data available.</em>';
        return;
    }

    element.innerHTML = '<div style="padding:20px; color:#666">Rendered Graph</div>';

    try {
        if (typeof mermaid === "undefined") {
            element.innerText = "Mermaid library not loaded.";
            return;
        }

        const uniqueId = `mermaid-graph-${Date.now()}`;
        const result = await mermaid.render(uniqueId, graphDef);
        element.innerHTML = result.svg;

    } catch (e) {
        console.error("Mermaid Render Fail", e);
        element.innerHTML = `<div style="color:red; padding:10px; border:1px solid red;">
            <strong>Graph Render Error:</strong><br/>
            ${e.message || String(e)}
        </div>`;
    }
}

/**
 * Renders HTML DSL output inside an isolated iframe in the Render tab.
 *
 * @param {string} htmlCode - Generated HTML DSL string.
 * @returns {void}
 */
function renderHtmlDSL(htmlCode) {
    const container = document.getElementById("tikz-output-container");
    if (!container) return;

    container.innerHTML = '<iframe style="width:100%; height:100%; min-height:400px; border:none;"></iframe>';
    const iframe = container.querySelector("iframe");
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    doc.open();
    doc.write(htmlCode);
    doc.close();
}

/**
 * Renders TikZ code inside the Render tab for TikZJax processing.
 *
 * @param {string} tikzCode - Generated TikZ LaTeX string.
 * @returns {void}
 */
function renderTikZ(tikzCode) {
    const container = document.getElementById("tikz-output-container");
    if (!container) return;
    container.innerHTML = `<script type="text/tikz">${tikzCode}</script>`;
}

/**
 * Executes emitted WebGPU WGSL or WASM SIMD forward and backward kernels in-browser
 * and captures hardware-observed tensor shapes.
 *
 * @param {Object|null} runtimeTrace - `RuntimeTraceReport` object from Python.
 * @param {string} preferredBackend - Selected backend ('auto', 'pyodide', 'webgpu', 'wasm_simd').
 * @returns {Promise<Object<string, Array<number>>>} Captured hardware shape dictionary.
 */
async function executeHardwareForwardBackward(runtimeTrace, preferredBackend) {
    if (!runtimeTrace || !runtimeTrace.forward_kernel_code) {
        return {};
    }
    const inMeta = runtimeTrace.ir_inputs_meta || {};
    const outMeta = runtimeTrace.ir_outputs_meta || {};

    if (preferredBackend === "wasm_simd" && window.runWasmCompute) {
        const fwdRes = await window.runWasmCompute(runtimeTrace.forward_kernel_code, inMeta, outMeta);
        if (runtimeTrace.backward_kernel_code && window.runWasmBackward) {
            await window.runWasmBackward(runtimeTrace.backward_kernel_code, inMeta, outMeta);
        }
        return window.captureWasmTensorShapes ? window.captureWasmTensorShapes(fwdRes) : {};
    }

    if ((preferredBackend === "webgpu" || preferredBackend === "auto") && window.runWebGPUCompute) {
        const fwdRes = await window.runWebGPUCompute(runtimeTrace.forward_kernel_code, inMeta, outMeta);
        if (runtimeTrace.backward_kernel_code && window.runWebGPUBackward) {
            await window.runWebGPUBackward(runtimeTrace.backward_kernel_code, inMeta, outMeta);
        }
        return window.captureRuntimeShapes ? window.captureRuntimeShapes(fwdRes) : {};
    }

    return {};
}

/**
 * Updates the Shared `ml-switcheroo-ir` & Autodiff Graph Inspector panel based on the selected view.
 *
 * @param {Object|null} runtimeTrace - `RuntimeTraceReport` payload from `ConversionResult`.
 * @returns {void}
 */
function updateIrInspectorView(runtimeTrace) {
    const inspectorBox = document.getElementById("runtime-ir-inspector");
    const viewSel = document.getElementById("select-ir-inspector-view");
    const codePre = document.getElementById("runtime-ir-inspector-code");
    if (!inspectorBox || !viewSel || !codePre) return;

    if (!runtimeTrace || !runtimeTrace.executed) {
        inspectorBox.style.display = "none";
        codePre.textContent = "";
        return;
    }

    inspectorBox.style.display = "block";
    const view = viewSel.value;
    if (view === "forward_ir_python") {
        codePre.textContent = runtimeTrace.forward_ir_python || "# No forward LogicalGraph captured.";
    } else if (view === "backward_ir_python") {
        codePre.textContent = runtimeTrace.backward_ir_python || "# No backward VJP LogicalGraph captured.";
    } else if (view === "synthesized_backward_code") {
        codePre.textContent = runtimeTrace.synthesized_backward_code || "# No target backward/VJP code synthesized.";
    } else if (view === "backend_kernel") {
        const fwdK = runtimeTrace.forward_kernel_code || "";
        const bwdK = runtimeTrace.backward_kernel_code || "";
        codePre.textContent = `// === Forward Kernel (${runtimeTrace.backend_used}) ===\n${fwdK}\n\n// === Backward VJP Kernel (${runtimeTrace.backend_used}) ===\n${bwdK}`;
    } else if (view === "backward_mermaid") {
        codePre.textContent = runtimeTrace.backward_mermaid || "graph TD;";
    }
}

/**
 * Renders the On-Device Forward & Backward Pass telemetry into `#panel-runtime`.
 *
 * @param {Object|null} runtimeTrace - `RuntimeTraceReport` payload from `ConversionResult`.
 * @returns {void}
 */
function renderRuntimeTelemetry(runtimeTrace) {
    latestRuntimeTrace = runtimeTrace;
    const badgesEl = document.getElementById("runtime-summary-badges");
    const impasseEl = document.getElementById("runtime-impasse-card");
    const tableEl = document.getElementById("runtime-telemetry-table");
    const paramEl = document.getElementById("runtime-param-grads-table");

    if (!badgesEl || !impasseEl || !tableEl || !paramEl) return;

    if (!runtimeTrace || !runtimeTrace.executed) {
        const errMsg = (runtimeTrace && runtimeTrace.error)
            ? `On-Device Execution Error: ${runtimeTrace.error}`
            : 'Enable "⚡ Run Fwd & Bwd On-Device" and click Run Translation to inspect forward shapes and backward gradients.';
        badgesEl.innerHTML = `<em style="color:#999">${errMsg}</em>`;
        impasseEl.style.display = "none";
        tableEl.innerHTML = "";
        paramEl.innerHTML = "";
        updateIrInspectorView(null);
        return;
    }

    const inShapesStr = (runtimeTrace.input_shapes || []).map(s => `(${s.join(", ")})`).join("; ");
    const outShapeStr = `(${(runtimeTrace.output_shape || []).join(", ")})`;
    const lossStr = runtimeTrace.loss_value !== null && runtimeTrace.loss_value !== undefined
        ? Number(runtimeTrace.loss_value).toFixed(4)
        : "N/A";
    const gradNormStr = runtimeTrace.total_grad_norm !== null && runtimeTrace.total_grad_norm !== undefined
        ? Number(runtimeTrace.total_grad_norm).toExponential(3)
        : "N/A";
    const modeLabel = runtimeTrace.conversion_mode === "optimized"
        ? `Optimised (${runtimeTrace.opt_level || "O2"})`
        : "1:1 Preserve Trivia";
    const passesCount = Array.isArray(runtimeTrace.executed_passes) ? runtimeTrace.executed_passes.length : 0;
    const memBytes = Number(runtimeTrace.estimated_memory_bytes || 0);
    const commBytes = Number(runtimeTrace.estimated_comm_bytes || 0);
    const finiteBadge = runtimeTrace.finite_check_passed
        ? '<span class="rt-badge rt-badge-ok">✓ All Finite (zero_chex)</span>'
        : '<span class="rt-badge rt-badge-warn">⚠ NaN/Inf Detected</span>';

    badgesEl.innerHTML = `
        <span class="rt-badge"><strong>Mode:</strong> ${modeLabel} (${passesCount} passes)</span>
        <span class="rt-badge"><strong>Backend:</strong> ${runtimeTrace.backend_used}</span>
        <span class="rt-badge"><strong>Input Shape:</strong> <code>${inShapesStr}</code></span>
        <span class="rt-badge"><strong>Fwd Output Shape:</strong> <code>${outShapeStr}</code></span>
        <span class="rt-badge"><strong>Scalar Loss:</strong> <code>${lossStr}</code></span>
        <span class="rt-badge"><strong>Total Grad L2 Norm:</strong> <code>${gradNormStr}</code></span>
        <span class="rt-badge"><strong>Est. Memory:</strong> <code>${memBytes} B</code></span>
        <span class="rt-badge"><strong>Collective Comm:</strong> <code>${commBytes} B</code></span>
        ${finiteBadge}
    `;

    updateIrInspectorView(runtimeTrace);

    // Static Analysis Limits Resolved Card
    const impasseItems = [];
    const ctors = runtimeTrace.resolved_constructors || {};
    for (const lname of Object.keys(ctors)) {
        const c = ctors[lname];
        impasseItems.push(
            `<li><strong>Lazy Constructor Concretized:</strong> <code>self.${c.layer_name}</code> (<code>${c.op_type}</code>) &rarr; inferred <code>in_features=${c.inferred_in_features}</code></li>`
        );
    }
    const nodes = Object.values(runtimeTrace.node_telemetry || {});
    for (const n of nodes) {
        if (n.is_dynamic) {
            impasseItems.push(
                `<li><strong>Data-Dependent Shape Resolved:</strong> <code>${n.var_name}</code> (line ${n.lineno || "-"}) &rarr; concrete shape <code>(${(n.output_shape || []).join(", ")})</code> (${n.dynamic_reason || "dynamic"})</li>`
            );
        }
    }
    for (const br of (runtimeTrace.resolved_branches || [])) {
        impasseItems.push(
            `<li><strong>Value-Dependent Branch Resolved:</strong> line ${br.lineno} (<code>${br.condition_src}</code>) &rarr; taken=<code>${br.taken_branch}</code></li>`
        );
    }
    const loops = runtimeTrace.loop_trip_counts || {};
    for (const ln of Object.keys(loops)) {
        impasseItems.push(
            `<li><strong>Dynamic Loop Trip Count:</strong> line ${ln} &rarr; <code>${loops[ln]}</code> iteration(s)</li>`
        );
    }

    if (impasseItems.length > 0) {
        impasseEl.style.display = "block";
        impasseEl.innerHTML = `
            <div class="impasse-title">🔓 Static Analysis Limits Resolved On-Device</div>
            <ul class="impasse-list">${impasseItems.join("")}</ul>
        `;
    } else {
        impasseEl.style.display = "none";
    }

    // Forward & Backward Node Telemetry Table
    if (nodes.length > 0) {
        const rowsHtml = nodes.map(n => {
            const inStr = (n.input_shapes || []).map(s => `(${s.join(", ")})`).join(", ") || "-";
            const outStr = `(${(n.output_shape || []).join(", ")})`;
            const dynBadge = n.is_dynamic
                ? `<span class="badge-dynamic" title="${n.dynamic_reason || ''}">Dynamic</span>`
                : '<span class="badge-static">Static</span>';
            const gShape = n.grad_shape ? `(${n.grad_shape.join(", ")})` : "-";
            const gNorm = (n.grad_norm !== null && n.grad_norm !== undefined)
                ? Number(n.grad_norm).toExponential(3)
                : "-";
            return `<tr>
                <td>L${n.lineno || "-"}: <code>${n.var_name}</code></td>
                <td><code>${n.op_type}</code></td>
                <td><code>${inStr}</code></td>
                <td><code>${outStr}</code> (${n.dtype})</td>
                <td>${dynBadge}</td>
                <td><code>${gShape}</code></td>
                <td><code>${gNorm}</code></td>
            </tr>`;
        }).join("");

        tableEl.innerHTML = `
            <h4>Forward &amp; Backward Activation Telemetry</h4>
            <table class="rt-table">
                <thead>
                    <tr>
                        <th>Line / Var</th>
                        <th>Operation</th>
                        <th>Input Shapes</th>
                        <th>Forward Output Shape</th>
                        <th>Resolution</th>
                        <th>Backward Grad Shape</th>
                        <th>Grad L2 Norm</th>
                    </tr>
                </thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    } else {
        tableEl.innerHTML = "";
    }

    // Parameter Gradients Table
    const paramGrads = runtimeTrace.parameter_gradients || {};
    const paramNorms = runtimeTrace.parameter_grad_norms || {};
    const pKeys = Object.keys(paramGrads);
    if (pKeys.length > 0) {
        const pRows = pKeys.map(pk => {
            const pShape = `(${(paramGrads[pk] || []).join(", ")})`;
            const pNorm = paramNorms[pk] !== undefined ? Number(paramNorms[pk]).toExponential(3) : "-";
            return `<tr>
                <td><code>${pk}</code></td>
                <td><code>${pShape}</code></td>
                <td><code>${pNorm}</code></td>
            </tr>`;
        }).join("");
        paramEl.innerHTML = `
            <h4>Parameter &amp; Input Gradients (Reverse-Mode Autodiff)</h4>
            <table class="rt-table">
                <thead>
                    <tr>
                        <th>Parameter / Tensor</th>
                        <th>Gradient Shape</th>
                        <th>Gradient L2 Norm</th>
                    </tr>
                </thead>
                <tbody>${pRows}</tbody>
            </table>
        `;
    } else {
        paramEl.innerHTML = "";
    }
}

/**
 * Executes the main translation and optional on-device forward/backward pass in Pyodide.
 *
 * @returns {Promise<void>} Resolves when transpilation and rendering complete.
 */
async function runTranspilation() {
    if (!pyodide || !srcEditor) return;
    const consoleEl = document.getElementById("console-output");
    const btn = document.getElementById("btn-convert");
    const srcCode = srcEditor.getValue();

    if (!srcCode.trim()) {
        consoleEl.innerText = "Source code is empty.";
        return;
    }

    btn.disabled = true;
    btn.innerText = "Running...";
    consoleEl.innerText = "Processing...";

    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;
    let srcFlav = "", tgtFlav = "";
    const srcReg = document.getElementById("src-flavour-region");
    const tgtReg = document.getElementById("tgt-flavour-region");
    if (srcReg && srcReg.style.display !== 'none') srcFlav = document.getElementById("src-flavour").value;
    if (tgtReg && tgtReg.style.display !== 'none') tgtFlav = document.getElementById("tgt-flavour").value;

    const conversionMode = document.getElementById("select-conversion-mode")?.value || "preserve_trivia";
    const optLevel = document.getElementById("select-opt-level")?.value || "O2";
    const optStrictMode = !!document.getElementById("chk-strict-mode").checked;
    const optSharding = !!document.getElementById("chk-enable-sharding").checked;
    const optGraph = (conversionMode === "optimized") || !!document.getElementById("chk-opt-graph").checked;
    const optImports = !!document.getElementById("chk-opt-imports").checked;

    const chkOnDevice = document.getElementById("chk-on-device-exec");
    const optOnDevice = chkOnDevice ? !!chkOnDevice.checked : false;
    const execBackend = document.getElementById("select-exec-backend")?.value || "webgpu";
    const inputShape = document.getElementById("input-tensor-shape")?.value || "";
    const inputDtype = document.getElementById("select-input-dtype")?.value || "float32";
    const randomSeed = document.getElementById("input-random-seed")?.value || "42";
    const lossReduction = document.getElementById("select-loss-reduction")?.value || "sum";
    const injectAnnotations = document.getElementById("chk-inject-annotations")
        ? !!document.getElementById("chk-inject-annotations").checked
        : true;
    const injectAssertions = document.getElementById("chk-inject-assertions")
        ? !!document.getElementById("chk-inject-assertions").checked
        : false;
    const emitBwdCode = document.getElementById("chk-emit-bwd-code")
        ? !!document.getElementById("chk-emit-bwd-code").checked
        : false;

    try {
        const bridgePayload = {
            source_code: srcCode,
            src_fw: srcFw,
            tgt_fw: tgtFw,
            src_flavour: srcFlav || null,
            tgt_flavour: tgtFlav || null,
            conversion_mode: conversionMode,
            opt_level: optLevel,
            strict_mode: !!optStrictMode,
            enable_sharding: !!optSharding,
            enable_graph_opt: !!optGraph,
            enable_import_fixer: !!optImports,
            enable_on_device: !!optOnDevice,
            exec_backend: execBackend,
            input_shape: inputShape,
            input_dtype: inputDtype,
            random_seed: parseInt(randomSeed, 10) || 42,
            loss_reduction: lossReduction,
            inject_annotations: !!injectAnnotations,
            inject_assertions: !!injectAssertions,
            emit_bwd_code: !!emitBwdCode,
            observed_shapes: null,
        };
        pyodide.globals.set("js_bridge_payload_json", JSON.stringify(bridgePayload));

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const jsonRaw = pyodide.globals.get("json_output");
        const result = JSON.parse(jsonRaw);

        if (optOnDevice && result.runtime_trace) {
            const hwShapes = await executeHardwareForwardBackward(result.runtime_trace, execBackend);
            if (Object.keys(hwShapes).length > 0) {
                bridgePayload.observed_shapes = hwShapes;
                pyodide.globals.set("js_bridge_payload_json", JSON.stringify(bridgePayload));
            }
            renderRuntimeTelemetry(result.runtime_trace);
            const tabRuntime = document.getElementById("tab-runtime");
            if (tabRuntime && result.runtime_trace.executed) {
                tabRuntime.checked = true;
            }
        } else {
            renderRuntimeTelemetry(null);
        }

        tgtEditor.setValue(result.code);

        if (weightEditor) {
            if (result.weight_script) {
                weightEditor.setValue(result.weight_script);
            } else {
                weightEditor.setValue("# No weight script generated.");
            }
        }

        if (tgtFw === 'html') {
             tgtEditor.setOption("mode", "htmlmixed");
        } else {
             tgtEditor.setOption("mode", "python");
        }

        setTimeout(() => {
            tgtEditor.refresh();
            if (weightEditor) weightEditor.refresh();
        }, 50);

        const weightLabel = document.getElementById("label-tab-weights");
        if (weightLabel) {
            if (result.weight_script && result.weight_script.length > 20) {
                weightLabel.style.color = "#2196f3";
                weightLabel.style.display = "inline-flex";
            } else {
                weightLabel.style.display = "none";
                if (document.getElementById("tab-weights") && document.getElementById("tab-weights").checked) {
                    document.getElementById("tab-trace").checked = true;
                }
            }
        }

        if (result.is_success) {
             consoleEl.innerText = `✅ Success!

--- Engine Logs ---
${result.logs}`;

             if (tgtFw === 'tikz' || srcFw === 'tikz') {
                 document.getElementById("tab-render").checked = true;
                 setTimeout(() => renderTikZ(result.code), 150);
             } else if (tgtFw === 'html' || srcFw === 'html') {
                 document.getElementById("tab-render").checked = true;
                 renderHtmlDSL(result.code);
             }

        } else {
             consoleEl.innerText = `❌ Error

${result.logs}`;
        }

        if (result.trace_events) {
             if (window.TraceGraph) {
                 new window.TraceGraph('trace-visualizer', updateLineHighlight).render(result.trace_events);
             }
             if (timeTravel) {
                 timeTravel.loadEvents(result.trace_events);
             }
        }
    } catch (err) {
        console.error("Transpilation JS Error", err);
        consoleEl.innerText += `
❌ JS Runtime Error:
${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Translation";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("btn-load-engine").addEventListener("click", initEngine);
    document.getElementById("btn-convert").addEventListener("click", runTranspilation);
    document.getElementById("btn-swap").addEventListener("click", swapContext);
    document.getElementById("btn-retro").addEventListener("click", toggleRetroMode);
});
