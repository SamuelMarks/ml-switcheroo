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
 * Keys are unique IDs, values are objects containing code and configuration.
 * @type {Object.<string, {label: string, srcFw: string, tgtFw: string, code: string, srcFlavour?: string, tgtFlavour?: string}>}
 */
let EXAMPLES = {
    "torch_nn": {
        "label": "PyTorch -> JAX (Flax NNX)",
        "srcFw": "torch",
        "tgtFw": "jax",
        "tgtFlavour": "flax_nnx",
        "code": `import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)`
    },
    "jax_nnx": {
        "label": "JAX (Flax NNX) -> PyTorch",
        "srcFw": "jax",
        "srcFlavour": "flax_nnx",
        "tgtFw": "torch",
        "code": `from flax import nnx
import jax.numpy as jnp

class Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(10, 10, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)`
    }
};

/**
 * Python script to execute inside Pyodide.
 * It serves as the interface between the JS UI and the Python ASTEngine.
 * Steps:
 * 1. Initializes SemanticsManager (Cached).
 * 2. Reads configuration from JS global variables.
 * 3. Runs transformation.
 * 4. Serializes output (Code, Logs, Trace) to JSON.
 * @const {string}
 */
const PYTHON_BRIDGE = `
import json
import traceback
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console

# Configure Log Capture
process_log = Console(record=True, force_terminal=False, width=80) 
set_console(process_log) 

response = {} 

try: 
    # 1. Load Inputs
    if 'GLOBAL_SEMANTICS' not in globals(): 
        # Pre-load semantics once to speed up subsequent runs
        GLOBAL_SEMANTICS = SemanticsManager() 
    
    # 2. Determine Effective Target Framework
    # If the user selected a "Flavour" (e.g. 'flax_nnx'), that becomes the 
    # structural target for the Rewriter, overriding the generic 'jax'. 
    # Level 0/1 logic (arrays/optax) is handled by inheritance in the adapter. 
    
    real_source = js_src_flavour if js_src_flavour else js_src_fw
    real_target = js_tgt_flavour if js_tgt_flavour else js_tgt_fw
    
    # Updated: Passed 'js_strict_mode' from JS context
    config = RuntimeConfig( 
        source_framework=real_source, 
        target_framework=real_target, 
        strict_mode=js_strict_mode
    ) 
    
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config) 
    result = engine.run(js_source_code) 

    # Serialize Result including Trace Data
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

/**
 * Initializes the Python VM (Pyodide), installs requirements, and prepares the UI.
 * This is triggered by the "Initialize Engine" button.
 *
 * @returns {Promise<void>}
 */
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
        // 1. Load Pyodide
        if (!window.loadPyodide) {
            await loadScript("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
        }

        // 2. Initialize VM if not ready
        if (!pyodide) {
            pyodide = await loadPyodide();
        }

        // 3. Guard: Check if package already loaded (for idempotency)
        const isInstalled = pyodide.runPython(`
import importlib.util
importlib.util.find_spec("ml_switcheroo") is not None
        `);

        if (!isInstalled) {
            statusEl.innerText = "Fetching Requirements...";
            await pyodide.loadPackage("micropip");
            const micropip = pyodide.pyimport("micropip");

            // FETCH AND PARSE requirements.txt
            const reqRes = await fetch("_static/requirements.txt");
            if (!reqRes.ok) throw new Error("Could not fetch requirements.txt");

            const reqText = await reqRes.text();
            const reqs = reqText.split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#'));

            statusEl.innerText = `Installing ${reqs.length} Dependencies...`;
            console.log("[WASM] Installing requirements:", reqs);

            await micropip.install("numpy");
            await micropip.install(reqs);

            // INSTALL WHEEL
            statusEl.innerText = "Installing Engine...";
            const wheelUrl = `_static/${wheelName}`;
            await micropip.install(wheelUrl);
        }

        // 4. Reveal Interface
        splashEl.style.display = "none";
        interfaceEl.style.display = "block";

        initEditors();

        // Merge Dynamic Examples
        if (window.SWITCHEROO_PRELOADED_EXAMPLES) {
            console.log("[WASM] Merging protocol-driven examples.");
            EXAMPLES = window.SWITCHEROO_PRELOADED_EXAMPLES;
        }

        initExampleSelector();
        initFlavourListeners(); // Bind new hierarchy listeners

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

/**
 * Initializes CodeMirror instances for Source and Target text areas.
 * @returns {void}
 */
function initEditors() {
    if (srcEditor) {
        srcEditor.refresh();
        if (tgtEditor) tgtEditor.refresh();
        return;
    }

    const commonOpts = {
        mode: "python",
        lineNumbers: true,
        viewportMargin: Infinity,
        theme: "default"
    };

    srcEditor = CodeMirror.fromTextArea(document.getElementById("code-source"), {
        ...commonOpts,
        readOnly: false
    });

    tgtEditor = CodeMirror.fromTextArea(document.getElementById("code-target"), {
        ...commonOpts,
        readOnly: true
    });
}

/**
 * Populates the example selection dropdown from the EXAMPLES object.
 * Automatically loads the first available example.
 * @returns {void}
 */
function initExampleSelector() {
    const sel = document.getElementById("select-example");
    if (!sel) return;

    sel.innerHTML = '<option value="" disabled>-- Select a Pattern --</option>';

    // Populate
    const sortedKeys = Object.keys(EXAMPLES).sort();
    let firstValid = null;

    for (const key of sortedKeys) {
        if (!firstValid) firstValid = key;
        const details = EXAMPLES[key];
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = details.label;
        sel.appendChild(opt);
    }

    // Default Selection
    if (firstValid) {
        sel.value = firstValid;
        loadExample(firstValid);
    } else {
        sel.querySelector('option[value=""]').selected = true;
    }

    sel.onchange = (e) => {
        loadExample(e.target.value);
    };
}

/**
 * Initializes listeners to show/hide Flavour dropdowns based on main framework selection.
 * E.g., showing 'Flax NNX' options only when 'JAX' is selected.
 * @returns {void}
 */
function initFlavourListeners() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

    const handler = (type) => {
        const sel = type === 'src' ? srcSel : tgtSel;
        const region = document.getElementById(`${type}-flavour-region`);

        // Logic: If selected framework has flavours defined in the DOM (e.g. key 'jax'), show them.
        if (sel.value === 'jax') {
            region.style.display = 'inline-block';
        } else {
            region.style.display = 'none';
        }
    };

    srcSel.addEventListener("change", () => handler('src'));
    tgtSel.addEventListener("change", () => handler('tgt'));

    // Initial triggering
    handler('src');
    handler('tgt');
}

/**
 * Loads a specific example configuration into the UI.
 * Updates CodeMirror, Framework Selectors, and Flavour Selectors.
 * @param {string} key - The example ID key.
 * @returns {void}
 */
function loadExample(key) {
    const details = EXAMPLES[key];
    if (!details) return;

    if (srcEditor) srcEditor.setValue(details.code);
    if (tgtEditor) tgtEditor.setValue("");

    // Update Main Frameworks
    const srcEl = document.getElementById("select-src");
    const tgtEl = document.getElementById("select-tgt");

    if (srcEl && details.srcFw) {
        setSelectValue(srcEl, details.srcFw);
        // Trigger event to update flavour visibility
        srcEl.dispatchEvent(new Event('change'));
    }

    if (tgtEl && details.tgtFw) {
        setSelectValue(tgtEl, details.tgtFw);
        tgtEl.dispatchEvent(new Event('change'));
    }

    // Update Flavours if provided
    const srcFlavourEl = document.getElementById("src-flavour");
    const tgtFlavourEl = document.getElementById("tgt-flavour");

    if (srcFlavourEl && details.srcFlavour) {
        setSelectValue(srcFlavourEl, details.srcFlavour);
    }

    if (tgtFlavourEl && details.tgtFlavour) {
        setSelectValue(tgtFlavourEl, details.tgtFlavour);
    }

    const cons = document.getElementById("console-output");
    if (cons) cons.innerText = `Loaded example: ${details.label}`;
}

/**
 * Helper to safely set a Select element's value.
 * Logs a warning if the option doesn't exist.
 * @param {HTMLSelectElement} selectEl - The select DOM element.
 * @param {string} value - The value to select.
 * @returns {void}
 */
function setSelectValue(selectEl, value) {
    let found = false;
    for (let i = 0; i < selectEl.options.length; i++) {
        if (selectEl.options[i].value === value) {
            selectEl.selectedIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        console.warn(`[WASM] Warning: Option '${value}' not found in dropdown.`);
    }
}

/**
 * Swaps Source and Target configurations (Values + Flavours + Code).
 * @returns {void}
 */
function swapContext() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");
    const srcFlavour = document.getElementById("src-flavour");
    const tgtFlavour = document.getElementById("tgt-flavour");

    // Swap Main
    const tmpFw = srcSel.value;
    srcSel.value = tgtSel.value;
    tgtSel.value = tmpFw;

    // Swap Flavours if applicable
    const tmpFlavour = srcFlavour.value;
    srcFlavour.value = tgtFlavour.value;
    tgtFlavour.value = tmpFlavour;

    // Trigger visibility update
    srcSel.dispatchEvent(new Event("change"));
    tgtSel.dispatchEvent(new Event("change"));

    if (!srcEditor || !tgtEditor) return;

    const srcCode = srcEditor.getValue();
    const tgtCode = tgtEditor.getValue();

    srcEditor.setValue(tgtCode);
    tgtEditor.setValue(srcCode);

    document.getElementById("console-output").innerText = "Context swapped.";
}

/**
 * Executes the Transpilation Pipeline via Pyodide.
 * @returns {Promise<void>}
 */
async function runTranspilation() {
    if (!pyodide || !srcEditor) return;

    const sourceCode = srcEditor.getValue();
    if (!sourceCode.trim()) {
        document.getElementById("console-output").innerText = "Source code is empty.";
        return;
    }

    const consoleEl = document.getElementById("console-output");
    const btn = document.getElementById("btn-convert");

    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;

    // Hierarchical Inputs
    const srcRegion = document.getElementById("src-flavour-region");
    const tgtRegion = document.getElementById("tgt-flavour-region");

    // Only send flavour if the region is actually visible (meaning hierarchy is active)
    let srcFlavour = "";
    let tgtFlavour = "";

    if (srcRegion && srcRegion.style.display !== "none") {
        srcFlavour = document.getElementById("src-flavour").value;
    }

    if (tgtRegion && tgtRegion.style.display !== "none") {
        tgtFlavour = document.getElementById("tgt-flavour").value;
    }

    const strictMode = !!document.getElementById("chk-strict-mode").checked;

    btn.disabled = true;
    btn.innerText = "Running...";
    consoleEl.innerText = `Translating ${srcFw}${srcFlavour ? '(' + srcFlavour + ')' : ''} -> ${tgtFw}${tgtFlavour ? '(' + tgtFlavour + ')' : ''}...`;

    try {
        pyodide.globals.set("js_source_code", sourceCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);

        // Pass flavours (empty string if standard)
        pyodide.globals.set("js_src_flavour", srcFlavour);
        pyodide.globals.set("js_tgt_flavour", tgtFlavour);

        pyodide.globals.set("js_strict_mode", strictMode);

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const rawJson = pyodide.globals.get("json_output");
        const result = JSON.parse(rawJson);

        tgtEditor.setValue(result.code);

        let logs = result.logs;
        if (!result.is_success) {
            logs += "\n[System] Errors detected.";
        }
        if (result.errors && result.errors.length > 0) {
            logs += `\n\n[Issues]:\n${result.errors.join('\n')}`;
        }

        consoleEl.innerText = logs;

        if (result.trace_events && window.TraceGraph) {
            const vis = new TraceGraph('trace-visualizer');
            vis.render(result.trace_events);
        }

    } catch (err) {
        consoleEl.innerText = `❌ Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "🔄🦘 Run Translation";
    }
}

/**
 * Loads an external JavaScript file asynchronously.
 * @param {string} src - The URL of the script to load.
 * @returns {Promise<void>}
 */
function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// Global Event Listeners
document.addEventListener("DOMContentLoaded", () => {
    const btnLoad = document.getElementById("btn-load-engine");
    if (btnLoad) btnLoad.addEventListener("click", initEngine);

    const btnConvert = document.getElementById("btn-convert");
    if (btnConvert) btnConvert.addEventListener("click", runTranspilation);

    const btnSwap = document.getElementById("btn-swap");
    if (btnSwap) btnSwap.addEventListener("click", swapContext);
});
