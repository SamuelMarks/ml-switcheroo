/*
 * switcheroo_demo.js
 * Client-side controller for the WASM interactive demo.
 *
 * Capabilities:
 * - Pyodide Engine Management (WASM).
 * - CodeMirror Editor sync.
 * - Trace Graph Visualization.
 * - Weight Script Generation.
 * - TikZ/LaTeX Rendering Integration (TikZJax).
 * - Mermaid AST Visualization.
 */

// Global State
let pyodide = null;
let srcEditor = null;
let tgtEditor = null;
let weightEditor = null;
let EXAMPLES = {};
let FW_TIERS = {};
let tikzLoaded = false;

// Default CDN fallback in case local assets are missing
const DEFAULT_TIKZJAX_URL = "https://tikzjax.com/v1/tikzjax.js";

// Trace Data State
let currentAstGraphs = { pre: "", post: "" };

/**
 * Python Script to be executed inside the Pyodide environment.
 * It imports the ml_switcheroo package, configures the ASTEngine,
 * runs the conversion, and generates weight scripts if applicable.
 */
const PYTHON_BRIDGE = `
import json
import traceback
import pathlib
import sys
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console

# Setup Output Capture
process_log = Console(record=True, force_terminal=False, width=120)
set_console(process_log)

response = {}

try:
    # Initialize Semantics Singleton if needed to avoid re-parsing JSONs
    if 'GLOBAL_SEMANTICS' not in globals():
        GLOBAL_SEMANTICS = SemanticsManager()
    
    # Resolve flavors from JS globals
    src_flavour = js_src_flavour if 'js_src_flavour' in globals() and js_src_flavour else None
    tgt_flavour = js_tgt_flavour if 'js_tgt_flavour' in globals() and js_tgt_flavour else None

    # Determine execution config
    real_source = src_flavour if src_flavour else js_src_fw
    real_target = tgt_flavour if tgt_flavour else js_tgt_fw
    
    config = RuntimeConfig(
        source_framework=js_src_fw,
        target_framework=js_tgt_fw,
        source_flavour=src_flavour,
        target_flavour=tgt_flavour,
        strict_mode=js_strict_mode
    )
    
    # 1. Run Transpilation
    print(f"Running AST Engine ({real_source} -> {real_target})...")
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config)
    result = engine.run(js_source_code)
    
    if result.success:
        print("Transpilation successful.")
    else:
        print("Transpilation failed with errors.")

    # 2. Run Weight Script Generator (If architecture detected)
    weight_script_code = ""
    # We always attempt weight gen if successful, but fallback to empty string if no layers found
    if result.success:
        try:
            # Virtual FS in Pyodide
            f_src = pathlib.Path("/tmp/src_model.py")
            f_src.write_text(js_source_code, encoding="utf-8")
            f_out = pathlib.Path("/tmp/migration_script.py")
            
            from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
            
            wgen = WeightScriptGenerator(GLOBAL_SEMANTICS, config)
            success = wgen.generate(f_src, f_out)
            
            if success and f_out.exists():
                weight_script_code = f_out.read_text(encoding="utf-8")
            else:
                pass 
                # Silent fail typical for non-architecture code
        except Exception as e:
            print(f"Weight Gen Error: {e}")

    # Build Response using Pydantic dumps where available or manual dict
    response = {
        "code": result.code,
        "logs": process_log.export_text(),
        "is_success": result.success,
        "errors": result.errors,
        "trace_events": result.trace_events,
        "weight_script": weight_script_code
    }
except Exception as e:
    err_str = traceback.format_exc()
    response = {
        "code": "",
        "logs": f"{process_log.export_text()}\\nCRITICAL PYTHON ERROR: {str(e)}\\n{err_str}",
        "is_success": False,
        "errors": [str(e)],
        "trace_events": [],
        "weight_script": ""
    }

json_output = json.dumps(response)
`;

/**
 * Initializes the Python environment (Pyodide) and installs dependencies.
 * Triggered by the "Initialize Engine" button.
 */
async function initEngine() {
    const rootEl = document.getElementById("switcheroo-wasm-root");
    const statusEl = document.getElementById("engine-status");
    const btnLoad = document.getElementById("btn-load-engine");
    const splashEl = document.getElementById("demo-splash");
    const interfaceEl = document.getElementById("demo-interface");
    const wheelName = (rootEl && rootEl.dataset.wheel) ? rootEl.dataset.wheel : "ml_switcheroo-0.0.1-py3-none-any.whl";

    statusEl.innerText = "Downloading...";
    statusEl.className = "status-badge status-loading";
    btnLoad.disabled = true;

    try {
        // 1. Load Pyodide
        if (!window.loadPyodide) await loadScript("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
        if (!pyodide) pyodide = await loadPyodide();

        // 2. Load TikZJax (Lazy with Smart Path Resolution)
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
            } catch (tiErr) {
                try {
                    await loadScript(DEFAULT_TIKZJAX_URL);
                    tikzLoaded = true;
                } catch (cdnErr) {
                    tikzLoaded = false;
                }
            }
        }

        // 3. Check for Engine Install
        const isInstalled = pyodide.runPython(`
import importlib.util
importlib.util.find_spec("ml_switcheroo") is not None
        `);

        if (!isInstalled) {
            statusEl.innerText = "Fetching Requirements...";
            await pyodide.loadPackage("micropip");
            const micropip = pyodide.pyimport("micropip");

            // Attempt to load local requirements or fallback to minimal set
            try {
                const reqRes = await fetch("_static/requirements.txt");
                if (reqRes.ok) {
                    const reqText = await reqRes.text();
                    const reqs = reqText.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
                    await micropip.install("numpy");
                    if (reqs.length > 0) await micropip.install(reqs);
                } else {
                    await micropip.install(["numpy", "pydantic", "rich", "libcst"]);
                }
            } catch (e) {
                await micropip.install(["numpy", "pydantic", "rich", "libcst"]);
            }

            statusEl.innerText = "Installing Engine...";
            await micropip.install(`_static/${wheelName}`);
        }

        // 4. UI Transition
        splashEl.style.display = "none";
        interfaceEl.style.display = "block";
        initEditors();

        if (window.SWITCHEROO_PRELOADED_EXAMPLES) EXAMPLES = window.SWITCHEROO_PRELOADED_EXAMPLES;
        if (window.SWITCHEROO_FRAMEWORK_TIERS) FW_TIERS = window.SWITCHEROO_FRAMEWORK_TIERS;

        initExampleSelector();
        initFrameworkListeners();
        updateRenderTabVisibility();

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";
        document.getElementById("btn-convert").disabled = false;

        // Initialize Mermaid config if not already done
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

    // Initialize Weight Script Editor
    const weightEl = document.getElementById("code-weights");
    if (weightEl) {
        weightEditor = CodeMirror.fromTextArea(weightEl, { ...opts, readOnly: true });
    }
}

/**
 * Populates the Example dropdown from the loaded registry.
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

    // Auto-select a neural example if possible for better first impression
    if (sortedKeys.length > 0) {
       const preferred = sortedKeys.find(k => k.includes("torch") && k.includes("neural"));
       const defKey = preferred || sortedKeys[0];
       sel.value = defKey;
       loadExample(defKey);
    }
}

/**
 * Attaches event listeners for framework dropdowns logic handling.
 */
function initFrameworkListeners() {
    const handler = (type) => {
        const sel = document.getElementById(`select-${type}`);
        const region = document.getElementById(`${type}-flavour-region`);

        if (region) {
             const hasFlavours = ['jax', 'flax'].includes(sel.value);
             region.style.display = hasFlavours ? 'inline-block' : 'none';
        }
        updateRenderTabVisibility();
    };

    document.getElementById("select-src").addEventListener("change", () => handler('src'));
    document.getElementById("select-tgt").addEventListener("change", () => handler('tgt'));
    updateRenderTabVisibility();

    // -- Tab Switching Listener for Refreshing Editors --
    document.querySelectorAll('input[name="wm-tabs"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.id === 'tab-weights' && weightEditor) {
                // Short timeout to allow CSS display transition so CodeMirror calculates size correctly
                setTimeout(() => weightEditor.refresh(), 20);
            }
        });
    });
}

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
}

function setSelectValue(el, val) {
    if (!el) return;
    for (let i=0; i<el.options.length; i++) {
        if (el.options[i].value === val) {
            el.selectedIndex = i;
            break;
        }
    }
}

function swapContext() {
    const s = document.getElementById("select-src");
    const t = document.getElementById("select-tgt");
    [s.value, t.value] = [t.value, s.value];

    s.dispatchEvent(new Event("change"));
    t.dispatchEvent(new Event("change"));

    const tmp = srcEditor.getValue();
    srcEditor.setValue(tgtEditor.getValue());
    tgtEditor.setValue(tmp);
}

/**
 * Highlights a line in the source editor when hovering over trace components.
 */
function updateLineHighlight(event, hover) {
    if (!srcEditor || !event.lineno) return;
    const ln = event.lineno - 1;
    const method = hover ? "addLineClass" : "removeLineClass";
    srcEditor[method](ln, "background", "cm-trace-highlight");
}

function toggleRetroMode() {
    document.body.classList.toggle("crt-mode");
}

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
 * Handles Mermaid v10+ async rendering properly.
 *
 * @param {string} graphDef - The Mermaid graph definition string (e.g. "graph TD...").
 */
async function renderMermaid(graphDef) {
    const element = document.getElementById('ast-mermaid-target');
    if (!element) return;

    if (!graphDef || graphDef.trim() === '') {
        element.innerHTML = '<em style="color:#999">No graph data available.</em>';
        return;
    }

    // Set loading state
    element.innerHTML = '<div style="padding:20px; color:#666">Generating Graph...</div>';

    try {
        if (typeof mermaid === "undefined") {
            element.innerText = "Mermaid library not loaded.";
            return;
        }

        // Generate unique ID for SVG isolation to prevent conflicts
        const uniqueId = `mermaid-graph-${Date.now()}`;

        // Attempt Render
        // API: mermaid.render(id, text) -> { svg: string } in v10
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
 * Executes the main translation logic in Pyodide.
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

    // Gather Inputs
    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;
    let srcFlav = "", tgtFlav = "";
    const srcReg = document.getElementById("src-flavour-region");
    const tgtReg = document.getElementById("tgt-flavour-region");
    if(srcReg && srcReg.style.display !== 'none') srcFlav = document.getElementById("src-flavour").value;
    if(tgtReg && tgtReg.style.display !== 'none') tgtFlav = document.getElementById("tgt-flavour").value;

    try {
        pyodide.globals.set("js_source_code", srcCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);
        pyodide.globals.set("js_src_flavour", srcFlav);
        pyodide.globals.set("js_tgt_flavour", tgtFlav);
        pyodide.globals.set("js_strict_mode", !!document.getElementById("chk-strict-mode").checked);

        // Execute Python
        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const jsonRaw = pyodide.globals.get("json_output");
        const result = JSON.parse(jsonRaw);

        tgtEditor.setValue(result.code);

        // Update Weight Script editor
        if (weightEditor) {
            if (result.weight_script) {
                weightEditor.setValue(result.weight_script);
            } else {
                weightEditor.setValue("# No weight script generated.");
            }
        }

        // Update syntax highlighting based on target framework
        if (tgtFw === 'html') {
             tgtEditor.setOption("mode", "htmlmixed");
        } else {
             tgtEditor.setOption("mode", "python");
        }

        // Force refresh to ensure highlighting applies
        setTimeout(() => {
            tgtEditor.refresh();
            if (weightEditor) weightEditor.refresh();
        }, 50);

        // Manage Weight Tab Visibility
        const weightLabel = document.getElementById("label-tab-weights");
        if (weightLabel) {
            // Reduced threshold to 20 chars to catch short valid scripts or errors
            if (result.weight_script && result.weight_script.length > 20) {
                weightLabel.style.color = "#2196f3"; // Visual cue
                weightLabel.style.display = "inline-flex"; // Show tab
            } else {
                weightLabel.style.display = "none"; // Hide tab
                // If user was on the tab, switch them back to trace
                if (document.getElementById("tab-weights") && document.getElementById("tab-weights").checked) {
                    document.getElementById("tab-trace").checked = true;
                }
            }
        }

        if (result.is_success) {
             consoleEl.innerText = `✅ Success!\n\n--- Engine Logs ---\n${result.logs}`;

             if (tgtFw === 'tikz' || srcFw === 'tikz') {
                 document.getElementById("tab-render").checked = true;
                 setTimeout(() => renderTikZ(result.code), 150);
             } else if (tgtFw === 'html' || srcFw === 'html') {
                 document.getElementById("tab-render").checked = true;
                 renderHtmlDSL(result.code);
             }

        } else {
             consoleEl.innerText = `❌ Error\n\n${result.logs}`;
        }

        // Trace Handling (Viz)
        if (result.trace_events) {
             if (window.TraceGraph) {
                 new TraceGraph('trace-visualizer', updateLineHighlight).render(result.trace_events);
             }
             const snapshots = result.trace_events.filter(e => e.type === "ast_snapshot");
             currentAstGraphs.pre = "";
             currentAstGraphs.post = "";

             snapshots.forEach(s => {
                 if (s.description.includes("Before")) currentAstGraphs.pre = s.metadata.mermaid;
                 if (s.description.includes("After")) currentAstGraphs.post = s.metadata.mermaid;
             });

             // Use the fixed renderMermaid logic
             await renderMermaid(currentAstGraphs.pre || currentAstGraphs.post);
        }
    } catch (err) {
        console.error("Transpilation JS Error", err);
        consoleEl.innerText += `\n❌ JS Runtime Error:\n${err}`;
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

    document.getElementById("btn-ast-prev")?.addEventListener("click", (e) => {
        renderMermaid(currentAstGraphs.pre);
    });

    document.getElementById("btn-ast-next")?.addEventListener("click", (e) => {
        renderMermaid(currentAstGraphs.post);
    });
});
