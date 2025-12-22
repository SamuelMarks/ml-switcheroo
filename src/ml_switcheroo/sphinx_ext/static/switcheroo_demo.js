/**
 * @file switcheroo_demo.js
 * @description Client-side logic for the ML-Switcheroo WebAssembly Demo in Sphinx documentation.
 * Handles Pyodide initialization, CodeMirror editor state, and the UI interaction for running
 * transpilation purely in the browser via a Python Wheel loaded from GitHub Releases.
 *
 * Capabilities:
 * - Loads Pyodide and installs wheels dynamically from a remote URL ("latest").
 * - Manages Hierarchical Framework Selection (Flavour Dropdowns).
 * - Executes the AST Engine via a Python Bridge.
 * - Renders Trace Graphs for debugging.
 * - Filters target options based on Source Tier compatibility.
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
 * @type {Object.<string, {label: string, srcFw: string, tgtFw: string, srcFlavour?: string, tgtFlavour?: string, code: string, requiredTier?: string}>}
 */
let EXAMPLES = {
    "torch_nn": {
        "label": "PyTorch -> JAX (Flax NNX)",
        "srcFw": "torch",
        "tgtFw": "jax",
        "tgtFlavour": "flax_nnx",
        "requiredTier": "neural",
        "code": `import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.abs(x)`
    }
};

/**
 * Tier Capability Map injected by Python.
 * Keys are framework strings (e.g. 'torch'), values are arrays of supported tiers (['neural', 'array']).
 * @type {Object.<string, Array<string>>}
 */
let FW_TIERS = {};

/**
 * Python script string executed inside Pyodide to invoke the transpiler.
 * Handles stdout capturing, exception formatting, and JSON response creation.
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

/**
 * Initializes the Pyodide runtime and installs core logic.
 * Called when the "Initialize Engine" button is clicked.
 *
 * 1. Loads pyodide.js from CDN.
 * 2. Initializes the WASM runtime.
 * 3. Installs `micropip` and fetches `requirements.txt`.
 * 4. Installs the project wheel from the remote URL defined in data attributes.
 * 5. Visualizes the main interface upon success.
 */
async function initEngine() {
    const rootEl = document.getElementById("switcheroo-wasm-root");
    const statusEl = document.getElementById("engine-status");
    const btnLoad = document.getElementById("btn-load-engine");
    const splashEl = document.getElementById("demo-splash");
    const interfaceEl = document.getElementById("demo-interface");
    const logBox = document.getElementById("console-output");

    // Remote URL retrieved from Python injection via Jinja template logic in Sphinx extension
    const wheelUrl = rootEl.dataset.wheelUrl;

    // UI Updates
    statusEl.innerText = "Downloading...";
    statusEl.className = "status-badge status-loading";
    btnLoad.disabled = true;
    btnLoad.innerText = "Loading Pyodide...";

    try {
        // 1. Load Pyodide Core
        if (!window.loadPyodide) {
            await loadScript("https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js");
        }
        if (!pyodide) {
            pyodide = await loadPyodide();
        }

        // 2. Check if already installed
        const isInstalled = pyodide.runPython(`
import importlib.util
importlib.util.find_spec("ml_switcheroo") is not None
        `);

        // 3. Install Package
        if (!isInstalled) {
            statusEl.innerText = "Fetching Requirements...";
            await pyodide.loadPackage("micropip");
            const micropip = pyodide.pyimport("micropip");

            // Requirements (copied locally to _static during doc build)
            const reqRes = await fetch("_static/requirements.txt");
            if (reqRes.ok) {
                const reqText = await reqRes.text();
                const reqs = reqText.split('\n')
                    .map(line => line.trim())
                    .filter(line => line && !line.startsWith('#'));

                statusEl.innerText = `Installing Dependencies...`;
                await micropip.install("numpy");
                // Allow error tolerance for complex deps not in Pyodide
                try {
                    await micropip.install(reqs);
                } catch(e) {
                    console.warn("Some requirements failed to install, proceeding:", e);
                }
            }

            statusEl.innerText = "Installing Engine (Remote)...";

            console.log(`[WASM] Downloading wheel from: ${wheelUrl}`);
            await micropip.install(wheelUrl);
        }

        // 4. Reveal Interface
        splashEl.style.display = "none";
        interfaceEl.style.display = "block";

        initEditors();

        // Load injected globals from Python
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
        logBox.innerText = `❌ WASM Initialization Error:\n\n${err}\n\nCheck console for details.`;
    }
}

/**
 * Initializes CodeMirror editors for source and target areas.
 * Checks for existing instances to avoid double initialization.
 */
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

/**
 * Populates the example selector dropdown based on the global EXAMPLES object.
 * Triggers the load of the first available example if possible.
 */
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

/**
 * Sets up event listeners to show/hide flavour dropdowns.
 * For example, if JAX is selected, shows the Flax NNX sub-option.
 */
function initFlavourListeners() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

    const handler = (type) => {
        const sel = type === 'src' ? srcSel : tgtSel;
        const region = document.getElementById(`${type}-flavour-region`);
        // Logic currently only shows flavours for JAX (hardcoded for demo simplicity)
        if (sel.value === 'jax') {
            region.style.display = 'inline-block';
        } else {
            region.style.display = 'none';
        }
    };
    srcSel.addEventListener("change", () => handler('src'));
    tgtSel.addEventListener("change", () => handler('tgt'));

    // Initial trigger
    handler('src');
    handler('tgt');
}

/**
 * Loads a specific example logic into the editor and updates dropdown states.
 * Also triggers logic to filter valid targets based on source tier.
 *
 * @param {string} key - The example ID key (e.g. 'torch_nn').
 */
function loadExample(key) {
    const details = EXAMPLES[key];
    if (!details) return;

    if (srcEditor) srcEditor.setValue(details.code);
    if (tgtEditor) tgtEditor.setValue("");

    const srcEl = document.getElementById("select-src");
    const tgtEl = document.getElementById("select-tgt");

    if (srcEl && details.srcFw) {
        setSelectValue(srcEl, details.srcFw);
        // Dispatch change to trigger flavour visibility logic
        srcEl.dispatchEvent(new Event('change'));
    }

    // Store the required tier on the DOM for validation during execution/selection
    srcEl.dataset.requiredTier = details.requiredTier || "array";

    // Filter targets based on this new requirement BEFORE setting target
    filterTargetOptions(details.requiredTier);

    if (tgtEl && details.tgtFw) {
        setSelectValue(tgtEl, details.tgtFw);
        tgtEl.dispatchEvent(new Event('change'));
    }

    // Handle Flavour updates if defined in the example
    const srcFlavourEl = document.getElementById("src-flavour");
    const tgtFlavourEl = document.getElementById("tgt-flavour");
    if (srcFlavourEl && details.srcFlavour) setSelectValue(srcFlavourEl, details.srcFlavour);
    if (tgtFlavourEl && details.tgtFlavour) setSelectValue(tgtFlavourEl, details.tgtFlavour);

    const cons = document.getElementById("console-output");
    if (cons) cons.innerText = `Loaded example: ${details.label}\nRequirement: ${details.requiredTier}`;
}

/**
 * Disables target framework options that do not support the required tier.
 * Used to preventing mapping High Level (Neural) code to Low Level (NumPy) targets.
 *
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
        const supports = FW_TIERS[fwKey] || ["array"]; // Default to conservative if unknown

        if (supports.includes(reqTier)) {
            opt.disabled = false;
            if (!firstValid) firstValid = fwKey;
        } else {
            opt.disabled = true;
        }
    }

    // If current selection became invalid, switch to first valid option
    const current = tgtSel.value;
    const currentSupports = FW_TIERS[current] || ["array"];
    if (!currentSupports.includes(reqTier) && firstValid) {
        tgtSel.value = firstValid;
        tgtSel.dispatchEvent(new Event('change'));
    }
}

/**
 * Helper to select an option in a dropdown by value.
 * @param {HTMLSelectElement} selectEl - The select element.
 * @param {string} value - The value string to select.
 */
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

/**
 * Swaps Source and Target frameworks (and code content via temp variable).
 * Triggered by the swap button.
 */
function swapContext() {
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

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

/**
 * Triggers the Python Transpilation Engine via Pyodide.
 * Gathers inputs, sets globals, runs the bridge script, and processes output.
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
    const reqTier = document.getElementById("select-src").dataset.requiredTier || "array";
    const effectiveTgt = tgtFlavour || tgtFw;
    const supported = FW_TIERS[effectiveTgt] || ["array", "neural", "extras"];

    if (!supported.includes(reqTier)) {
        consoleEl.innerText = `⚠️  Warning: Converting ${reqTier.toUpperCase()} code to ${effectiveTgt} which only supports [${supported.join(", ")}].\nResult may contain escape hatches.`;
    } else {
        consoleEl.innerText = `Translating...`;
    }

    btn.disabled = true;
    btn.innerText = "Running...";

    try {
        // Pass variables to Python scope
        pyodide.globals.set("js_source_code", srcCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);
        pyodide.globals.set("js_src_flavour", srcFlavour);
        pyodide.globals.set("js_tgt_flavour", tgtFlavour);
        pyodide.globals.set("js_strict_mode", !!document.getElementById("chk-strict-mode").checked);

        // Execute Bridge
        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        // Retrieve JSON response
        const result = JSON.parse(pyodide.globals.get("json_output"));
        tgtEditor.setValue(result.code);

        if (result.is_success) {
             consoleEl.innerText += "\n✅ Success!";
        } else {
             consoleEl.innerText = result.logs;
        }

        // Render Traces if visualiser class is available
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

/**
 * Helper to dynamically load external JS scripts relative to the page.
 * @param {string} src - The script URL.
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

// Initialization Hook
document.addEventListener("DOMContentLoaded", () => {
    const btnLoad = document.getElementById("btn-load-engine");
    if (btnLoad) btnLoad.addEventListener("click", initEngine);
    const btnConvert = document.getElementById("btn-convert");
    if (btnConvert) btnConvert.addEventListener("click", runTranspilation);
    const btnSwap = document.getElementById("btn-swap");
    if (btnSwap) btnSwap.addEventListener("click", swapContext);
});
