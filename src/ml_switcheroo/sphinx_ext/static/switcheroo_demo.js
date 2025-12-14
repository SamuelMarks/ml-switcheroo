let pyodide = null;
let srcEditor = null;
let tgtEditor = null;

// --- Feature 1 & 2: Examples Configuration ---
// Define robust defaults here so the dropdown never renders empty
// even if the Python build pipeline fails to inject dynamic examples.
let EXAMPLES = {
    "torch": {
        "label": "Standard PyTorch Example",
        "srcFw": "torch",
        "tgtFw": "jax",
        "code": `import torch
import torch.nn as nn

class Model(nn.Module): 
    def forward(self, x): 
        return torch.abs(x)`
    },
    "jax": {
        "label": "Standard JAX Example",
        "srcFw": "jax",
        "tgtFw": "torch",
        "code": `import jax.numpy as jnp

def compute(x):
    return jnp.abs(x)`
    }
};

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
        GLOBAL_SEMANTICS = SemanticsManager() 
    
    # Updated: Passed 'js_strict_mode' from JS context
    config = RuntimeConfig(source_framework=js_src_fw, target_framework=js_tgt_fw, strict_mode=js_strict_mode) 
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

        // 3. Guard: Check if package already loaded (for idompotency)
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

        // 4. Reveal Interface (MUST be visible before Init Editor)
        splashEl.style.display = "none";
        interfaceEl.style.display = "block";

        // --- Feature 0: Init CodeMirror ---
        // Initialize editors now that elements are visible in DOM
        initEditors();

        // --- Merge Dynamic Examples ---
        // If Python injected extra examples, merge them.
        // If not, our hardcoded defaults in EXAMPLES ensure the UI isn't broken.
        if (window.SWITCHEROO_PRELOADED_EXAMPLES) {
            console.log("[WASM] Merging protocol-driven examples.");
            // Merge logic: Dynamic overwrites default if keys collide, but we keep defaults
            EXAMPLES = { ...EXAMPLES, ...window.SWITCHEROO_PRELOADED_EXAMPLES };
        }

        initExampleSelector();

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";

        // Ensure buttons reactivated
        document.getElementById("btn-convert").disabled = false;

        // Append status
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
        // If re-initializing or unhiding, refresh layout
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

function initExampleSelector() {
    const sel = document.getElementById("select-example");
    if (!sel) return;

    // Reset Dropdown completely to prevent stale states
    sel.innerHTML = '<option value="" disabled>-- Select a Pattern --</option>';

    // Determine Logic for Default Selection
    // Priority:
    // 1. "torch" (Generic Python-injected matches key defined in TorchAdapter)
    let targetKey = "torch";

    // Fallback: If torch not found, pick first avaliable
    if (!EXAMPLES[targetKey]) {
        const keys = Object.keys(EXAMPLES);
        if (keys.length > 0) targetKey = keys[0];
        else targetKey = null;
    }

    let defaultFound = false;

    // Populate
    for (const [key, details] of Object.entries(EXAMPLES)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = details.label;
        sel.appendChild(opt);

        // Mark selected property
        if (key === targetKey) {
            opt.selected = true;
            defaultFound = true;
        }
    }

    // Trigger Load if we found a valid target
    if (defaultFound && targetKey) {
        // We load the example into the editor, ensuring editor state matches dropdown
        loadExample(targetKey);
    } else {
        // Fallback selection of placeholder
        sel.querySelector('option[value=""]').selected = true;
    }

    // Re-attach listener (clearing innerHTML removes old listeners on options)
    // We attach onchange only if not already attached?
    // Ideally we should just use onclick, but change is standard for Select.
    sel.onchange = (e) => {
        loadExample(e.target.value);
    };
}

function loadExample(key) {
    const details = EXAMPLES[key];
    if (!details) return;

    if (srcEditor) srcEditor.setValue(details.code);
    if (tgtEditor) tgtEditor.setValue(""); // clear old Output

    // Update Dropdowns if IDs exist
    const srcEl = document.getElementById("select-src");
    const tgtEl = document.getElementById("select-tgt");

    if (srcEl && details.srcFw) srcEl.value = details.srcFw;
    if (tgtEl && details.tgtFw) tgtEl.value = details.tgtFw;

    const cons = document.getElementById("console-output");
    if(cons) cons.innerText = `Loaded example: ${details.label}`;
}

function swapContext() {
    // HTML elements
    const srcSel = document.getElementById("select-src");
    const tgtSel = document.getElementById("select-tgt");

    // Swap Dropdowns
    const tmpFw = srcSel.value;
    srcSel.value = tgtSel.value;
    tgtSel.value = tmpFw;

    // Swap Code via Editors
    if (!srcEditor || !tgtEditor) return;

    const srcCode = srcEditor.getValue();
    const tgtCode = tgtEditor.getValue();

    srcEditor.setValue(tgtCode);
    tgtEditor.setValue(srcCode);

    document.getElementById("console-output").innerText = "Context swapped. Ready to translate.";
}

async function runTranspilation() {
    if (!pyodide || !srcEditor) return;

    // Read from CodeMirror
    const sourceCode = srcEditor.getValue();
    if (!sourceCode.trim()) {
        document.getElementById("console-output").innerText = "Source code is empty.";
        return;
    }

    const consoleEl = document.getElementById("console-output");
    const btn = document.getElementById("btn-convert");

    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;

    // Strict Mode: Check the toggle state
    const strictMode = !!document.getElementById("chk-strict-mode").checked;

    btn.disabled = true;
    btn.innerText = "Running...";
    consoleEl.innerText = `Translating ${srcFw} -> ${tgtFw} (Strict: ${strictMode})...`;

    try {
        pyodide.globals.set("js_source_code", sourceCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);
        // Pass strict mode flag to Python
        pyodide.globals.set("js_strict_mode", strictMode);

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const rawJson = pyodide.globals.get("json_output");
        const result = JSON.parse(rawJson);

        // Write to CodeMirror
        tgtEditor.setValue(result.code);

        let logs = result.logs;
        if(!result.is_success) {
            logs += "\n[System] Errors detected during ast conversion.";
        }

        // Append lifecycle or engine errors if present, even if success flag is true (warnings)
        if (result.errors && result.errors.length > 0) {
             logs += `\n\n[Warning/Error Logic]:\n${result.errors.join('\n')}`;
        }

        consoleEl.innerText = logs;

        // --- Feature 05: Visualizer Integration ---
        if (result.trace_events && window.TraceGraph) {
            console.log("[WASM] Rendering Trace Graph...");
            const vis = new TraceGraph('trace-visualizer');
            vis.render(result.trace_events);
        }

    } catch (err) {
        consoleEl.innerText = `❌ Python Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "🔄🦘Run Translation";
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
    if(btnLoad) btnLoad.addEventListener("click", initEngine);

    const btnConvert = document.getElementById("btn-convert");
    if(btnConvert) btnConvert.addEventListener("click", runTranspilation);

    const btnSwap = document.getElementById("btn-swap");
    if(btnSwap) btnSwap.addEventListener("click", swapContext);
});
