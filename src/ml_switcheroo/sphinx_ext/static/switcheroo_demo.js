/* src/ml_switcheroo/sphinx_ext/static/switcheroo_demo.js */

/*
 * switcheroo_demo.js
 * Client-side controller with Debugging Enabled.
 *
 * Capabilities:
 * - Pyodide Engine Management (WASM).
 * - CodeMirror Editor sync.
 * - Trace Graph Rendering.
 * - Semantic Knowledge Base Interaction (Validation/Hints).
 * - TikZ/LaTeX Rendering Integration (TikZJax).
 * - Mermaid for AST visualisation
 */

// Global State
let pyodide = null;
let srcEditor = null;
let tgtEditor = null;
let EXAMPLES = {};
let FW_TIERS = {};
let tikzLoaded = false;

// Default CDN fallback in case local assets are missing
const DEFAULT_TIKZJAX_URL = "https://tikzjax.com/v1/tikzjax.js";

// Trace Data State
let currentAstGraphs = { pre: "", post: "" };

// Debug Configuration
const DEBUG_MODE = true;
function debugLog(msg, data = null) {
    if (DEBUG_MODE) {
        if (data) console.log(`[Switcheroo] ${msg}`, data);
        else console.log(`[Switcheroo] ${msg}`);
    }
}

// Python Bridge Script
const PYTHON_BRIDGE = `
import json
import traceback
from rich.console import Console
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import set_console, log_info

# Setup Output Capture
process_log = Console(record=True, force_terminal=False, width=120) 
set_console(process_log) 

response = {} 

try: 
    print(">>> Python Bridge Started") 
    
    # Initialize Semantics Singleton if needed
    if 'GLOBAL_SEMANTICS' not in globals(): 
        print(">>> Initializing SemanticsManager...") 
        GLOBAL_SEMANTICS = SemanticsManager() 
    
    # Resolve flavors
    src_flavour = js_src_flavour if 'js_src_flavour' in globals() and js_src_flavour else None
    tgt_flavour = js_tgt_flavour if 'js_tgt_flavour' in globals() and js_tgt_flavour else None

    real_source = src_flavour if src_flavour else js_src_fw
    real_target = tgt_flavour if tgt_flavour else js_tgt_fw
    
    print(f">>> Config: {real_source} -> {real_target} (Strict: {js_strict_mode})") 
    
    config = RuntimeConfig( 
        source_framework=js_src_fw, 
        target_framework=js_tgt_fw, 
        source_flavour=src_flavour, 
        target_flavour=tgt_flavour, 
        strict_mode=js_strict_mode
    ) 
    
    print(">>> Launching ASTEngine...") 
    engine = ASTEngine(semantics=GLOBAL_SEMANTICS, config=config) 
    result = engine.run(js_source_code) 
    print(f">>> Conversion Finished. Success: {result.success}") 
    
    # Check trace statistics
    snap_count = sum(1 for e in result.trace_events if e['type'] == 'ast_snapshot') 
    print(f">>> Trace: {len(result.trace_events)} events, {snap_count} snapshots.") 

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

            // Try to construct local path using Sphinx DOCUMENTATION_OPTIONS
            if (window.DOCUMENTATION_OPTIONS && window.DOCUMENTATION_OPTIONS.URL_ROOT) {
                const root = window.DOCUMENTATION_OPTIONS.URL_ROOT;
                // Normalize root (ensure it ends with /)
                const safeRoot = root.endsWith('/') ? root : root + '/';
                const localPath = `${safeRoot}_static/tikzjax/tikzjax.js`;
                debugLog(`Attempting local TikZJax load: ${localPath}`);
                urlToLoad = localPath;
            } else if (window.TIKZJAX_URL) {
                // Fallback to what was injected in __init__.py (might be incorrect for subpages)
                urlToLoad = window.TIKZJAX_URL;
            }

            try {
                await loadScript(urlToLoad);
                tikzLoaded = true;
                debugLog("TikZJax loaded.");
            } catch (tiErr) {
                console.warn(`Local TikZ load failed (${urlToLoad}). Fallback to CDN.`);
                try {
                    await loadScript(DEFAULT_TIKZJAX_URL);
                    tikzLoaded = true;
                    debugLog("TikZJax loaded from CDN.");
                } catch (cdnErr) {
                    console.error("TikZJax CDN failed:", cdnErr);
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
                console.warn("Reqs fetch failed, using fallback deps.");
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

    } catch (err) {
        console.error(err);
        statusEl.innerText = "Load Failed";
        statusEl.className = "status-badge status-error";
        btnLoad.disabled = false;
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

    if (sortedKeys.length > 0) {
       const preferred = sortedKeys.find(k => k.includes("torch") && k.includes("neural"));
       const defKey = preferred || sortedKeys[0];
       sel.value = defKey;
       loadExample(defKey);
    }
}

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
}

function updateRenderTabVisibility() {
    const srcFw = document.getElementById("select-src").value;
    const tgtFw = document.getElementById("select-tgt").value;

    // Enable render tab if either side is TikZ
    const isTikzInvolved = (srcFw === 'tikz' || tgtFw === 'tikz');

    const tabLabel = document.getElementById("label-tab-render");
    const tabInput = document.getElementById("tab-render");

    if (tabLabel) {
        if (isTikzInvolved) {
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

        await pyodide.runPythonAsync(PYTHON_BRIDGE);

        const jsonRaw = pyodide.globals.get("json_output");
        const result = JSON.parse(jsonRaw);

        tgtEditor.setValue(result.code);

        if (result.is_success) {
             consoleEl.innerText = `✅ Success!\n\n--- Engine Logs ---\n${result.logs}`;

             if (tgtFw === 'tikz' || srcFw === 'tikz') {
                 document.getElementById("tab-render").checked = true;
                 // Give DOM render beat + TikZJax async search time
                 setTimeout(() => renderTikZ(result.code), 150);
             }
        } else {
             consoleEl.innerText = `❌ Error\n\n${result.logs}`;
        }

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

             renderMermaid(currentAstGraphs.pre || currentAstGraphs.post);
        }
    } catch (err) {
        console.error("Transpilation JS Error", err);
        consoleEl.innerText += `\n❌ JS Runtime Error:\n${err}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Run Translation";
    }
}

/**
 * Helper: Polls for window.tikzjax namespace presence.
 */
function waitForTikZ(timeout = 10000) {
    return new Promise(resolve => {
        const start = Date.now();
        const check = () => {
            // Check existence, not specific method, to be robust
            if (window.tikzjax) {
                resolve(true);
            } else if (Date.now() - start > timeout) {
                resolve(false);
            } else {
                requestAnimationFrame(check);
            }
        };
        check();
    });
}

/**
 * Injects TikZ Code into the DOM to trigger TikZJax processing.
 * @param {string} tikzCode - The LaTeX string generated by the engine.
 */
async function renderTikZ(tikzCode) {
    const container = document.getElementById("tikz-output-container");
    if (!container) return;

    // Clear previous results/loaders
    container.innerHTML = "";

    // Show loading state while waiting for WASM
    const loaderId = "tikz-loading-indicator";
    container.innerHTML = `<div id="${loaderId}" style='color:#666; padding:20px;'>Rendering TikZ Diagram... (WASM via <span id="tikz-method">...</span>)</div>`;
    container.style.display = "block";

    if (!tikzLoaded) {
        container.innerHTML = "<div style='color:orange;'>⚠️ TikZJax library not loaded. Check network or console.</div>";
        return;
    }

    debugLog("Waiting for TikZJax readiness...");

    // FIX: Increased timeout to 5000ms. WASM compilation can be slow on first load.
    const ready = await waitForTikZ(5000);

    if (!ready) {
         const methInfo = document.getElementById("tikz-method");
         if (methInfo) methInfo.innerText = "Event Trigger";
         debugLog("TikZJax global not ready. Fallback to DOMContentLoaded event.");
    } else {
         debugLog("TikZJax global detected.");
    }

    debugLog("Injecting TikZ Script...");

    // Clear loader
    container.innerHTML = "";

    // Create script tag expected by tikzjax
    const scriptEl = document.createElement("script");
    scriptEl.type = "text/tikz";
    scriptEl.dataset.showConsole = "true";
    scriptEl.textContent = tikzCode;

    container.appendChild(scriptEl);

    // Trigger processing logic
    if (window.tikzjax && typeof window.tikzjax.process === 'function') {
        try {
            debugLog("Calling tikzjax.process(container)...");
            await window.tikzjax.process(container);
            // Check visibility
            setTimeout(() => {
                const svg = container.querySelector("svg");
                if (svg) {
                    svg.style.display = "block";
                    svg.style.margin = "0 auto";
                }
            }, 100);
        } catch (e) {
            console.error("TikZJax API error:", e);
            container.innerHTML += `<div style="color:red; margin-top:10px;">Render Error: ${e.message}</div>`;
        }
    } else {
        debugLog("API not found. Triggering DOMContentLoaded event to force scan.");
        // Fallback for libraries relying on load events
        document.dispatchEvent(new Event("DOMContentLoaded"));
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
        const id = 'ast-svg-' + Date.now();
        const result = await mermaid.render(id, graphDefinition);
        const svgContent = (typeof result === 'string') ? result : result.svg;
        targetEl.innerHTML = svgContent;
    } catch (e) {
        console.error("Mermaid Render Exception:", e);
        targetEl.innerHTML = `<div style='color:red; text-align:left; font-family:monospace;'>Mermaid Syntax Error:<br/>${e.message}</div>`;
    }
}

// Init Listeners
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

    if (typeof mermaid !== "undefined") {
        mermaid.initialize({ startOnLoad: false, theme: 'neutral' });
    }
});
