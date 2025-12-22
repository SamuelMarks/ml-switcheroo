/**
 * @file switcheroo_demo.js
 * @description Client-side logic for the ML-Switcheroo WebAssembly Demo in Sphinx documentation.
 * Handles Pyodide initialization, CodeMirror editor state, and the UI interaction for running
 * transpilation purely in the browser.
 *
 * Capabilities:
 * - Loads Pyodide and installs wheels dynamically.
 * - Manages Hierarchical Framework Selection (Flavour Dropdowns).
 * - Executes the AST Engine via a Python Bridge.
 * - Renders Trace Graphs for debugging.
 * - **NEW**: Filters target options based on Source Tier compatibility.
 */

/**
 * Global Pyodide instance.
 * @type {any}
 */
let pyodide = null;

/**
 * CodeMirror instance for the Source editor.
 * @type {any}
 */
let srcEditor = null;

/**
 * CodeMirror instance for the Target editor.
 * @type {any}
 */
let tgtEditor = null;

/**
 * Dictionary of pre-loaded examples.
 * This is populated/merged by 'window.SWITCHEROO_PRELOADED_EXAMPLES' injected by the Sphinx extension.
 * @type {Object.<string, {label: string, srcFw: string, tgtFw: string, code: string, requiredTier?: string}>}
 */
let EXAMPLES = {
    "torch_nn": {
        "label": "PyTorch -> JAX (Flax NNX)",
        "srcFw": "torch",
        "tgtFw": "jax",
        "tgtFlavour": "flax_nnx",
        "requiredTier": "neural",
        "code": `import torch
import torch.nn as nn

class Model(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.linear = nn.Linear(10, 10) 

    def forward(self, x): 
        return self.linear(x)`
    }
};

/**
 * Tier Capability Map injected by Python.
 * Example: { "torch": ["neural", "array"], "numpy": ["array"] }
 * @type {Object.<string, Array<string>>}
 */
let FW_TIERS = {};

// Python Bridge Script (unchanged logic)
const PYTHON_BRIDGE = `
import json
import traceback
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console

process_log = Console(record=True, force_terminal=False, width=80) 
set_console(process_log) 

response = {} 

try: 
    if 'GLOBAL_SEMANTICS' not in globals(): 
        GLOBAL_SEMANTICS = SemanticsManager() 
    
    real_source = js_src_flavour if js_src_flavour else js_src_fw
    real_target = js_tgt_flavour if js_tgt_flavour else js_tgt_fw
    
    config = RuntimeConfig( 
        source_framework=real_source, 
        target_framework=real_target, 
        strict_mode=js_strict_mode
    ) 
    
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config) 
    result = engine.run(js_source_code) 

    response = { 
        "code": result.code, 
        "logs": process_log.export_text(), 
        "is_success": result.success, 
        "errors": result.errors, 
        "trace_events": result.trace_events
    } 
except Exception as e: 
    response = { 
        "code": "", 
        "logs": f"{process_log.export_text()}\\nCRITICAL ERROR: {str(e)}\\n{traceback.format_exc()}", 
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
    const logBox = document.getElementById("console-output");
    const wheelName = rootEl.dataset.wheel;

    statusEl.innerText = "Downloading...";
    statusEl.className = "status-badge status-loading";
    btnLoad.disabled = true;
    btnLoad.innerText = "Loading Pyodide...";

    try {
        if (!window.loadPyodide) {
            await loadScript("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
        }
        if (!pyodide) {
            pyodide = await loadPyodide();
        }

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
            const reqs = reqText.split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#'));

            statusEl.innerText = `Installing Dependencies...`;
            await micropip.install("numpy");
            await micropip.install(reqs);

            statusEl.innerText = "Installing Engine...";
            const wheelUrl = `_static/${wheelName}`;
            await micropip.install(wheelUrl);
        }

        // Reveal Interface
        splashEl.style.display = "none";
        interfaceEl.style.display = "block";

        initEditors();

        if (window.SWITCHEROO_PRELOADED_EXAMPLES) {
            EXAMPLES = window.SWITCHEROO_PRELOADED_EXAMPLES;
        }

        // Load Tier Metadata
        if (window.SWITCHEROO_FRAMEWORK_TIERS) {
            FW_TIERS = window.SWITCHEROO_FRAMEWORK_TIERS;
            console.log("[WASM] Loaded Framework Tiers:", FW_TIERS);
        }

        initExampleSelector();
        initFlavourListeners();

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";

        document.getElementById("btn-convert").disabled = false;
        logBox.innerText += "\nEngine initialized successfully.";

    } catch (err) {
        console.error(err);
        splashEl.style.display = "none";
        interfaceEl.style.display = "none";
        statusEl.innerText = "Load Failed";
        statusEl.className = "status-badge status-error";
        logBox.innerText = `❌ WASM Initialization Error:\n\n${err}\n`;
    }
}

function initEditors() {
    if (srcEditor) {
        srcEditor.refresh();
        if (tgtEditor) tgtEditor.refresh();
        return;
    }
    const commonOpts = { mode: "python", lineNumbers: true, viewportMargin: Infinity, theme: "default" };
    srcEditor = CodeMirror.fromTextArea(document.getElementById("code-source"), { ...commonOpts, readOnly: false });
    tgtEditor = CodeMirror.fromTextArea(document.getElementById("code-target"), { ...commonOpts, readOnly: true });
}

function initExampleSelector() {
    const sel = document.getElementById("select-example");
    if (!sel) return;

    sel.innerHTML = '<option value="" disabled>-- Select a Pattern --</option>';
    const sortedKeys = Object.keys(EXAMPLES).sort();
    let firstValid = null;

    for (const key of sortedKeys) {
        if (!firstValid) firstValid = key;
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = EXAMPLES[key].label;
        sel.appendChild(opt);
    }

    if (firstValid) {
        sel.value = firstValid;
        loadExample(firstValid);
    }

    sel.onchange = (e) => loadExample(e.target.value);
}

function initFlavourListeners() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

    const handler = (type) => {
        const sel = type === 'src' ? srcSel : tgtSel;
        const region = document.getElementById(`${type}-flavour-region`);
        if (sel.value === 'jax') {
            region.style.display = 'inline-block';
        } else {
            region.style.display = 'none';
        }
    };
    srcSel.addEventListener("change", () => handler('src'));
    tgtSel.addEventListener("change", () => handler('tgt'));
    handler('src');
    handler('tgt');
}

function loadExample(key) {
    const details = EXAMPLES[key];
    if (!details) return;

    if (srcEditor) srcEditor.setValue(details.code);
    if (tgtEditor) tgtEditor.setValue("");

    const srcEl = document.getElementById("select-src");
    const tgtEl = document.getElementById("select-tgt");

    if (srcEl && details.srcFw) {
        setSelectValue(srcEl, details.srcFw);
        srcEl.dispatchEvent(new Event('change'));
    }

    // Store the required tier on the DOM for validation
    srcEl.dataset.requiredTier = details.requiredTier || "array";

    // Filter targets based on this new requirement BEFORE setting target
    filterTargetOptions(details.requiredTier);

    if (tgtEl && details.tgtFw) {
        setSelectValue(tgtEl, details.tgtFw);
        tgtEl.dispatchEvent(new Event('change'));
    }

    // Handle Flavour updates...
    const srcFlavourEl = document.getElementById("src-flavour");
    const tgtFlavourEl = document.getElementById("tgt-flavour");
    if (srcFlavourEl && details.srcFlavour) setSelectValue(srcFlavourEl, details.srcFlavour);
    if (tgtFlavourEl && details.tgtFlavour) setSelectValue(tgtFlavourEl, details.tgtFlavour);

    const cons = document.getElementById("console-output");
    if (cons) cons.innerText = `Loaded example: ${details.label}\nRequirement: ${details.requiredTier}`;
}

/**
 * Disables target frameworks that do not support the required tier.
 * Used to preventing mapping High Level (Neural) -> Low Level (NumPy).
 * @param {string} reqTier - The required tier ('neural', 'array', 'extras').
 */
function filterTargetOptions(reqTier) {
    const tgtSel = document.getElementById("select-tgt");
    if (!tgtSel || !reqTier) return;

    // If metadata not loaded yet, skip
    if (Object.keys(FW_TIERS).length === 0) return;

    let firstValid = null;

    for (let i = 0; i < tgtSel.options.length; i++) {
        const opt = tgtSel.options[i];
        const fwKey = opt.value;
        const supports = FW_TIERS[fwKey] || ["array"]; // Default conservative

        if (supports.includes(reqTier)) {
            opt.disabled = false;
            if (!firstValid) firstValid = fwKey;
        } else {
            opt.disabled = true;
        }
    }

    // If current selection is invalid, switch to first valid
    const current = tgtSel.value;
    const currentSupports = FW_TIERS[current] || ["array"];
    if (!currentSupports.includes(reqTier) && firstValid) {
        tgtSel.value = firstValid;
        tgtSel.dispatchEvent(new Event('change'));
    }
}

function setSelectValue(selectEl, value) {
    let found = false;
    for (let i = 0; i < selectEl.options.length; i++) {
        if (selectEl.options[i].value === value && !selectEl.options[i].disabled) {
            selectEl.selectedIndex = i;
            found = true;
            break;
        }
    }
}

function swapContext() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

    // Check if swap is valid logic?
    // If we swap, we might be moving code that requires 'Neural' into a 'NumPy' target.
    // Generally allowed unless we re-validate content.

    const tmpFw = srcSel.value;
    srcSel.value = tgtSel.value;
    tgtSel.value = tmpFw;

    srcSel.dispatchEvent(new Event("change"));
    tgtSel.dispatchEvent(new Event("change"));

    if (srcEditor && tgtEditor) {
        const srcCode = srcEditor.getValue();
        const tgtCode = tgtEditor.getValue();
        srcEditor.setValue(tgtCode);
        tgtEditor.setValue(srcCode);
    }
}

async function runTranspilation() {
    if (!pyodide || !srcEditor) return;
    const consoleEl = document.getElementById("console-output");
    const btn = document.getElementById("btn-convert");
    const srcCode = srcEditor.getValue();

    if (!srcCode.trim()) {
        consoleEl.innerText = "Source code is empty.";
        return;
    }

    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;

    // Hierarchical Inputs extraction
    let srcFlavour = "";
    let tgtFlavour = "";
    const srcRegion = document.getElementById("src-flavour-region");
    const tgtRegion = document.getElementById("tgt-flavour-region");

    if (srcRegion && srcRegion.style.display !== "none") srcFlavour = document.getElementById("src-flavour").value;
    if (tgtRegion && tgtRegion.style.display !== "none") tgtFlavour = document.getElementById("tgt-flavour").value;

    // Final Compatibility Check
    // If user somehow bypassed UI or swapped contexts invalidly
    // We check: does configured Tgt Flavour (or Tgt FW) support the code's tier?
    // We define this loosely based on 'reqTier' from last loaded example, or default to array.

    const reqTier = document.getElementById("select-src").dataset.requiredTier || "array";

    // Need to resolve effective target tier support.
    // If flavour is used (e.g. flax_nnx), check its tiers. If generic 'jax', check its tiers.
    const effectiveTgt = tgtFlavour || tgtFw;
    const supported = FW_TIERS[effectiveTgt] || ["array", "neural", "extras"]; // Default permissive if missing logic

    if (!supported.includes(reqTier)) {
        consoleEl.innerText = `⚠️  Warning: Converting ${reqTier.toUpperCase()} code to ${effectiveTgt} which only supports [${supported.join(", ")}].\nResult may contain escape hatches.`;
    } else {
        consoleEl.innerText = `Translating...`;
    }

    btn.disabled = true;
    btn.innerText = "Running...";

    try {
        pyodide.globals.set("js_source_code", srcCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);
        pyodide.globals.set("js_src_flavour", srcFlavour);
        pyodide.globals.set("js_tgt_flavour", tgtFlavour);
        pyodide.globals.set("js_strict_mode", !!document.getElementById("chk-strict-mode").checked);

        await pyodide.runPythonAsync(PYTHON_BRIDGE);
        const result = JSON.parse(pyodide.globals.get("json_output"));
        tgtEditor.setValue(result.code);

        if (result.is_success) {
             consoleEl.innerText += "\n✅ Success!";
        } else {
             consoleEl.innerText = result.logs;
        }

        if (result.trace_events && window.TraceGraph) {
             new TraceGraph('trace-visualizer').render(result.trace_events);
        }
    } catch (err) {
        consoleEl.innerText = `❌ Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "🔄🦘 Run Translation";
    }
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const btnLoad = document.getElementById("btn-load-engine");
    if (btnLoad) btnLoad.addEventListener("click", initEngine);
    const btnConvert = document.getElementById("btn-convert");
    if (btnConvert) btnConvert.addEventListener("click", runTranspilation);
    const btnSwap = document.getElementById("btn-swap");
    if (btnSwap) btnSwap.addEventListener("click", swapContext);
});
