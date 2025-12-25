/*
 * switcheroo_demo.js
 * Client-side controller with Debugging Enabled.
 */

// Global State
let pyodide = null;
let srcEditor = null;
let tgtEditor = null;
let EXAMPLES = {};
let FW_TIERS = {};

// Trace Data State
let currentAstGraphs = { pre: "", post: "" };

// Debug Configuration
const DEBUG_MODE = true;
function debugLog(msg, data=null) {
    if (DEBUG_MODE) {
        if (data) console.log(`[Switcheroo] ${msg}`, data);
        else console.log(`[Switcheroo] ${msg}`);
    }
}

// Python Bridge Script
// Updates: Now prints diagnostic info to the output logs
const PYTHON_BRIDGE = `
import json
import traceback
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console, log_info

# Setup Output Capture
process_log = Console(record=True, force_terminal=False, width=80) 
set_console(process_log) 

response = {} 

try: 
    print(">>> Python Bridge Started") 
    
    # Initialize Semantics Singleton if needed
    if 'GLOBAL_SEMANTICS' not in globals(): 
        print(">>> Initializing SemanticsManager...") 
        GLOBAL_SEMANTICS = SemanticsManager() 
    
    # Resolve frameworks
    real_source = js_src_flavour if js_src_flavour else js_src_fw
    real_target = js_tgt_flavour if js_tgt_flavour else js_tgt_fw
    
    print(f">>> Config: {real_source} -> {real_target} (Strict: {js_strict_mode})") 
    
    config = RuntimeConfig( 
        source_framework=real_source, 
        target_framework=real_target, 
        strict_mode=js_strict_mode
    ) 
    
    print(">>> Launching ASTEngine...") 
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config) 
    result = engine.run(js_source_code) 
    print(f">>> Conversion Finished. Success: {result.success}") 
    print(f">>> Trace Events Captured: {len(result.trace_events)}") 
    
    # Check specifically for snapshots
    snap_count = sum(1 for e in result.trace_events if e['type'] == 'ast_snapshot') 
    print(f">>> Visualizer Snapshots Found: {snap_count}") 

    response = { 
        "code": result.code, 
        "logs": process_log.export_text(), 
        "is_success": result.success, 
        "errors": result.errors, 
        "trace_events": result.trace_events
    } 
except Exception as e: 
    err_str = traceback.format_exc() 
    print(f"!!! CRITICAL BRIDGE ERROR: {e}") 
    print(err_str) 
    
    response = { 
        "code": "", 
        "logs": f"{process_log.export_text()}\\nCRITICAL PYTHON ERROR: {str(e)}\\n{err_str}", 
        "is_success": False, 
        "errors": [str(e)], 
        "trace_events": [] 
    } 

json_output = json.dumps(response) 
`;

async function initEngine() {
    const rootEl = document.getElementById("switcheroo-wasm-root");
    const statusEl = document.getElementById("engine-status");
    const btnLoad = document.getElementById("btn-load-engine");
    const splashEl = document.getElementById("demo-splash");
    const interfaceEl = document.getElementById("demo-interface");
    const wheelName = rootEl.dataset.wheel;

    statusEl.innerText = "Downloading...";
    statusEl.className = "status-badge status-loading";
    btnLoad.disabled = true;

    try {
        if (!window.loadPyodide) await loadScript("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
        if (!pyodide) pyodide = await loadPyodide();

        const isInstalled = pyodide.runPython(`
import importlib.util
importlib.util.find_spec("ml_switcheroo") is not None
        `);

        if (!isInstalled) {
            statusEl.innerText = "Fetching Requirements...";
            await pyodide.loadPackage("micropip");
            const micropip = pyodide.pyimport("micropip");
            const reqRes = await fetch("_static/requirements.txt");
            const reqText = await reqRes.text();
            const reqs = reqText.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
            await micropip.install("numpy");
            await micropip.install(reqs);
            statusEl.innerText = "Installing Engine...";
            await micropip.install(`_static/${wheelName}`);
        }

        splashEl.style.display = "none";
        interfaceEl.style.display = "block";
        initEditors();

        if (window.SWITCHEROO_PRELOADED_EXAMPLES) EXAMPLES = window.SWITCHEROO_PRELOADED_EXAMPLES;
        if (window.SWITCHEROO_FRAMEWORK_TIERS) FW_TIERS = window.SWITCHEROO_FRAMEWORK_TIERS;

        initExampleSelector();
        initFlavourListeners();

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";
        document.getElementById("btn-convert").disabled = false;

    } catch (err) {
        console.error(err);
        statusEl.innerText = "Load Failed";
        statusEl.className = "status-badge status-error";
    }
}

function initEditors() {
    if (srcEditor) {
        srcEditor.refresh();
        if (tgtEditor) tgtEditor.refresh();
        return;
    }
    const opts = { mode: "python", lineNumbers: true, viewportMargin: Infinity, theme: "default" };
    srcEditor = CodeMirror.fromTextArea(document.getElementById("code-source"), { ...opts, readOnly: false });
    tgtEditor = CodeMirror.fromTextArea(document.getElementById("code-target"), { ...opts, readOnly: true });
}

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

    // Set default example to Flax NNX if available, otherwise first alphabetical
    const preferred = "flax_nnx_tier2_neural";
    if (Object.prototype.hasOwnProperty.call(EXAMPLES, preferred)) {
        sel.value = preferred;
        loadExample(preferred);
    } else if (sortedKeys.length > 0) {
        sel.value = sortedKeys[0];
        loadExample(sel.value);
    }
}

function initFlavourListeners() {
    const handler = (type) => {
        const sel = document.getElementById(`select-${type}`);
        const region = document.getElementById(`${type}-flavour-region`);
        region.style.display = sel.value === 'jax' ? 'inline-block' : 'none';
    };
    document.getElementById("select-src").addEventListener("change", () => handler('src'));
    document.getElementById("select-tgt").addEventListener("change", () => handler('tgt'));
    handler('src'); handler('tgt');
}

function loadExample(key) {
    const ex = EXAMPLES[key];
    if (!ex) return;
    srcEditor.setValue(ex.code);
    tgtEditor.setValue("");
    setSelectValue(document.getElementById("select-src"), ex.srcFw);
    filterTargetOptions(ex.requiredTier);
    setSelectValue(document.getElementById("select-tgt"), ex.tgtFw);
    // Trigger flavour logic
    document.getElementById("select-src").dispatchEvent(new Event('change'));
    document.getElementById("select-tgt").dispatchEvent(new Event('change'));
}

function filterTargetOptions(tier) {
    // Simplified logic for brevity in debug file
    const tgt = document.getElementById("select-tgt");
    for (let opt of tgt.options) opt.disabled = false;
}

function setSelectValue(el, val) {
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
        s.onload = res;
        s.onerror = rej;
        document.head.appendChild(s);
    });
}

// -----------------------------------------------------------------------------
// Debugged Transpilation Logic
// -----------------------------------------------------------------------------

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

    // Flavours
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

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const jsonRaw = pyodide.globals.get("json_output");

        const result = JSON.parse(jsonRaw);

        tgtEditor.setValue(result.code);

        if (result.is_success) {
             consoleEl.innerText = `✅ Success!\n\n--- Engine Logs ---\n${result.logs}`;
        } else {
             consoleEl.innerText = `❌ Error\n\n${result.logs}`;
        }

        // --- Visualizer Logic ---
        if (result.trace_events) {
             // 1. Render List
             if (window.TraceGraph) {
                 new TraceGraph('trace-visualizer', updateLineHighlight).render(result.trace_events);
             }

             // 2. Extract AST Graphs
             // Look for 'ast_snapshot' type strings (case sensitive match to Python Enum)
             const snapshots = result.trace_events.filter(e => e.type === "ast_snapshot");

             currentAstGraphs.pre = "";
             currentAstGraphs.post = "";

             snapshots.forEach(s => {
                 if (s.description.includes("Before")) currentAstGraphs.pre = s.metadata.mermaid;
                 if (s.description.includes("After")) currentAstGraphs.post = s.metadata.mermaid;
             });

             // Refresh View
             renderMermaid(currentAstGraphs.pre || currentAstGraphs.post);
        } else {
            debugLog("No trace events array in result.");
        }
    } catch (err) {
        console.error("Transpilation JS Error", err);
        consoleEl.innerText += `\n❌ JS Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Translation";
    }
}

async function renderMermaid(graphDefinition) {
    const targetEl = document.getElementById("ast-mermaid-target");

    if (!graphDefinition) {
        targetEl.innerHTML = "<em style='color:#999'>No AST graph data found in trace results.</em>";
        return;
    }

    if (typeof mermaid === "undefined") {
        targetEl.innerHTML = "<b style='color:red'>Mermaid Library Not Loaded</b>";
        return;
    }

    try {
        // Unique ID for SVG
        const id = 'ast-svg-' + Date.now();

        // mermaid.render returns {svg} in v10+
        const result = await mermaid.render(id, graphDefinition);

        // Handle API difference between v9 and v10
        // v10 returns object. v9 returns string callback or string.
        const svgContent = (typeof result === 'string') ? result : result.svg;
        targetEl.innerHTML = svgContent;
    } catch (e) {
        console.error("Mermaid Render Exception:", e);
        targetEl.innerHTML = `<div style='color:red; text-align:left; font-family:monospace;'>Mermaid Syntax Error:<br/>${e.message}</div>`;
    }
}

// Init Listeners
document.addEventListener("DOMContentLoaded", () => {
    // ... Buttons ...
    document.getElementById("btn-load-engine").addEventListener("click", initEngine);
    document.getElementById("btn-convert").addEventListener("click", runTranspilation);
    document.getElementById("btn-swap").addEventListener("click", swapContext);
    document.getElementById("btn-retro").addEventListener("click", toggleRetroMode);

    // AST Sub-Buttons (Before/After)
    document.getElementById("btn-ast-prev")?.addEventListener("click", (e) => {
        renderMermaid(currentAstGraphs.pre);
    });

    document.getElementById("btn-ast-next")?.addEventListener("click", (e) => {
        renderMermaid(currentAstGraphs.post);
    });

    // Config Mermaid
    if (typeof mermaid !== "undefined") {
        mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
    }
});
