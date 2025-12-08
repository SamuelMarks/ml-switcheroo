let pyodide = null; 
let srcEditor = null;
let tgtEditor = null;

// --- Feature 1: Transformations Examples ---
const EXAMPLES = {
    "math_ops": {
        label: "Math Operations (Torch ↔ JAX)",
        srcFw: "torch",
        tgtFw: "jax",
        code: `import torch\n\ndef compute_loss(prediction, target):\n    """Calculates Mean Absolute Error."""\n    diff = torch.abs(prediction - target)\n    loss = torch.mean(diff)\n    return loss`
    },
    "neural_net": {
        label: "Neural Network (Torch ↔ NNX)",
        srcFw: "torch",
        tgtFw: "jax",
        code: `import torch.nn as nn\n\nclass SimplePerceptron(nn.Module):\n    """Basic Single-Layer Perceptron."""\n    def __init__(self, in_features, out_features):\n        super().__init__() \n        self.layer = nn.Linear(in_features, out_features)\n\n    def forward(self, x):\n        return self.layer(x)`
    },
    "array_dims": {
        label: "Dimension Swapping (Permute ↔ Transpose)",
        srcFw: "torch",
        tgtFw: "jax",
        code: `import torch\n\ndef transpose_batch(batch):\n    """Swaps dimensions (B, C, H, W) -> (B, H, W, C)."""\n    return torch.permute(batch, (0, 2, 3, 1))`
    },
    "inplace": {
        label: "In-Place Unrolling (add_ ↔ add)",
        srcFw: "torch",
        tgtFw: "jax",
        code: `def update_step(x, delta):\n    # In-place assignment ensures functional compatibility\n    # x.add_(delta) -> x = x + delta\n    x = x.add_(delta)\n    return x`
    },
    "rng_state": {
        label: "RNG Injection (Dropout)",
        srcFw: "torch",
        tgtFw: "jax",
        code: `import torch\nimport torch.nn.functional as F\n\ndef noisy_identity(x):\n    # Dropout requires explicit RNG keys in JAX\n    return F.dropout(x, p=0.5, training=True)`
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
    
    config = RuntimeConfig(source_framework=js_src_fw, target_framework=js_tgt_fw, strict_mode=False) 
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config) 
    result = engine.run(js_source_code) 

    response = { 
        "code": result.code, 
        "logs": process_log.export_text(), 
        "is_success": result.success, 
        "errors": result.errors 
    } 
except Exception as e: 
    response = { 
        "code": "", 
        "logs": f"{process_log.export_text()}\\nCRITICAL ERROR: {str(e)}\\n{traceback.format_exc()}", 
        "is_success": False, 
        "errors": [str(e)] 
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
        initExampleSelector();

        statusEl.innerText = "Ready";
        statusEl.className = "status-badge status-ready";

        // Ensure buttons reactivated
        document.getElementById("btn-convert").disabled = false;

        // Append status, preserving any log from example loader
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

    // Avoid double population
    if (sel.options.length > 1) return;

    // Clear default HTML placeholder
    sel.innerHTML = "";

    const DEFAULT_KEY = "neural_net";

    // Populate
    for (const [key, details] of Object.entries(EXAMPLES)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = details.label;
        if (key === DEFAULT_KEY) opt.selected = true;
        sel.appendChild(opt);
    }

    // Listener
    sel.addEventListener("change", (e) => {
        loadExample(e.target.value);
    });

    // Trigger loading defaults immediately
    loadExample(DEFAULT_KEY);
}

function loadExample(key) {
    const details = EXAMPLES[key];
    if (!details) return;

    if (srcEditor) srcEditor.setValue(details.code);
    if (tgtEditor) tgtEditor.setValue(""); // clear old Output

    document.getElementById("select-src").value = details.srcFw;
    document.getElementById("select-tgt").value = details.tgtFw;

    document.getElementById("console-output").innerText = `Loaded example: ${details.label}`;
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

    btn.disabled = true;
    btn.innerText = "Running...";
    consoleEl.innerText = `Translating ${srcFw} -> ${tgtFw}...`;

    try {
        pyodide.globals.set("js_source_code", sourceCode);
        pyodide.globals.set("js_src_fw", srcFw);
        pyodide.globals.set("js_tgt_fw", tgtFw);

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const rawJson = pyodide.globals.get("json_output");
        const result = JSON.parse(rawJson);

        // Write to CodeMirror
        tgtEditor.setValue(result.code);
        consoleEl.innerText = result.logs;

        if(!result.is_success) {
            consoleEl.innerText += "\n[System] Errors detected.";
        }

    } catch (err) {
        consoleEl.innerText = `❌ Python Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Translation";
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
