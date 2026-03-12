(function() {
    let pyodide = null;
    let cmSource = null;
    let cmTarget = null;
    let baseTranslatedCode = "";
    
    document.addEventListener('DOMContentLoaded', () => {
        const root = document.getElementById('switcheroo-wasm-v2-root');
        if (!root) return;

        const btnLoad = document.getElementById('btn-v2-load-engine');
        const splash = document.getElementById('v2-splash');
        const loader = document.getElementById('v2-loading-indicator');
        const interfaceDiv = document.getElementById('v2-interface');
        const statusBadge = document.getElementById('v2-engine-status');
        const errorBanner = document.getElementById('v2-error-banner');
        const errorMsg = document.getElementById('v2-error-message');
        
        const selSrc = document.getElementById('v2-select-src');
        const selTgt = document.getElementById('v2-select-tgt');
        const selTemplate = document.getElementById('v2-select-template');
        const selStrategy = document.getElementById('v2-strategy-select');
        const btnTranslate = document.getElementById('btn-v2-translate-now');
        const targetTitle = document.getElementById('v2-target-title');

        const examples = window.SWITCHEROO_PRELOADED_EXAMPLES || {};

        function updateTemplates(defaultTemplate = 'tier4_qwen3') {
            const src = selSrc.value;
            selTemplate.innerHTML = '';
            
            let matched = false;
            for (const [key, code] of Object.entries(examples)) {
                if (key.includes(src)) {
                    const opt = document.createElement('option');
                    const cleanKey = key.split(']').shift().replace('[', '').split('_').slice(1).join('_') || key;
                    opt.value = key;
                    opt.textContent = cleanKey;
                    if (key.includes(defaultTemplate)) {
                        opt.selected = true;
                        matched = true;
                    }
                    selTemplate.appendChild(opt);
                }
            }
            if (selTemplate.options.length > 0 && !matched) {
                selTemplate.options[0].selected = true;
            }
            
            if (selTemplate.options.length > 0) {
                if (cmSource) {
                    cmSource.setValue(examples[selTemplate.value] || "");
                } else {
                    document.getElementById('v2-source-editor').value = examples[selTemplate.value] || "";
                }
            }
        }

        selSrc.addEventListener('change', () => updateTemplates());
        selTemplate.addEventListener('change', () => {
            if (cmSource) cmSource.setValue(examples[selTemplate.value] || "");
        });

        // Initialize template select based on initial HTML value
        updateTemplates('tier4_qwen3');

        // Pyodide init
        btnLoad.addEventListener('click', async () => {
            btnLoad.style.display = 'none';
            loader.style.display = 'block';
            statusBadge.textContent = "Loading Engine...";
            statusBadge.style.background = "var(--v2-primary)";
            
            try {
                if (!window.loadPyodide) {
                    await new Promise((resolve, reject) => {
                        const script = document.createElement('script');
                        script.src = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";
                        script.onload = resolve;
                        script.onerror = reject;
                        document.head.appendChild(script);
                    });
                }
                
                pyodide = await loadPyodide({
                    stdout: (msg) => console.log(msg),
                    stderr: (msg) => console.error(msg)
                });

                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                
                const wheelName = root.dataset.wheel || "ml_switcheroo-latest-py3-none-any.whl";
                const wheelUrl = `../../_static/${wheelName}`;
                
                statusBadge.textContent = "Installing Packages...";
                await micropip.install("PyYAML");
                await micropip.install(wheelUrl);

                await pyodide.runPythonAsync(`
                    import sys
                    from ml_switcheroo.compiler.engine import TranslationEngine
                    from ml_switcheroo.config import Config
                    from ml_switcheroo.enums import Framework
                `);

                statusBadge.textContent = "Online";
                statusBadge.style.background = "#0f9d58";
                splash.style.display = 'none';
                interfaceDiv.style.display = 'block';
                
                // Initialize CodeMirror after interface is visible
                const opts = {
                    mode: 'python',
                    theme: 'default',
                    lineNumbers: true,
                    matchBrackets: true,
                    indentUnit: 4
                };
                cmSource = CodeMirror.fromTextArea(document.getElementById('v2-source-editor'), opts);
                cmTarget = CodeMirror.fromTextArea(document.getElementById('v2-target-editor'), { ...opts, readOnly: true });
                
                cmSource.setSize("100%", "100%");
                cmTarget.setSize("100%", "100%");

                // Sync scroll
                cmSource.on("scroll", () => {
                    cmTarget.scrollTo(cmSource.getScrollInfo().left, cmSource.getScrollInfo().top);
                });
                cmTarget.on("scroll", () => {
                    cmSource.scrollTo(cmTarget.getScrollInfo().left, cmTarget.getScrollInfo().top);
                });

                // Auto-translate on startup
                btnTranslate.click();

            } catch (err) {
                console.error(err);
                statusBadge.textContent = "Failed";
                statusBadge.style.background = "#d93025";
                errorMsg.textContent = err.message;
                errorBanner.style.display = 'block';
                loader.style.display = 'none';
                btnLoad.style.display = 'inline-block';
                btnLoad.textContent = "Retry Engine Load";
            }
        });

        document.getElementById('btn-v2-dismiss-error').addEventListener('click', () => {
            errorBanner.style.display = 'none';
        });

        // Toggle Dark Mode
        document.getElementById('btn-v2-theme-toggle').addEventListener('click', () => {
            document.body.classList.toggle('switcheroo-v2-dark');
            const isDark = document.body.classList.contains('switcheroo-v2-dark');
            if (cmSource) cmSource.setOption("theme", isDark ? "material-darker" : "default");
            if (cmTarget) cmTarget.setOption("theme", isDark ? "material-darker" : "default");
        });

        // Perform translation
        btnTranslate.addEventListener('click', async () => {
            if (!pyodide) return;
            const src = selSrc.value;
            const tgt = selTgt.value;
            const code = cmSource.getValue();
            
            btnTranslate.textContent = "⏳ Translating...";
            btnTranslate.disabled = true;

            try {
                pyodide.globals.set("src_fw_str", src);
                pyodide.globals.set("tgt_fw_str", tgt);
                pyodide.globals.set("input_code", code);
                
                await pyodide.runPythonAsync(`
                    cfg = Config()
                    cfg.source_framework = Framework(src_fw_str)
                    cfg.target_framework = Framework(tgt_fw_str)
                    engine = TranslationEngine(cfg)
                    try:
                        translated_code = engine.translate(input_code)
                    except Exception as e:
                        translated_code = f"# Translation Error:\\n# {str(e)}"
                `);
                
                baseTranslatedCode = pyodide.globals.get("translated_code");
                targetTitle.textContent = `${selTgt.options[selTgt.selectedIndex].text}`;
                applyStrategy();
                errorBanner.style.display = 'none';
            } catch (err) {
                errorMsg.textContent = err.message;
                errorBanner.style.display = 'block';
            } finally {
                btnTranslate.textContent = "🔄 Translate";
                btnTranslate.disabled = false;
            }
        });

        function applyStrategy() {
            if (!baseTranslatedCode) return;
            
            const strat = selStrategy.value;
            let finalCode = baseTranslatedCode;
            
            if (strat !== 'none') {
                targetTitle.textContent = `${selTgt.options[selTgt.selectedIndex].text} (Sharded: ${selStrategy.options[selStrategy.selectedIndex].text})`;
                finalCode += '\\n\\n# --- Applied Sharding Annotations ---\\n';
                finalCode += `# Strategy: ${strat}\\n`;
                
                if (selTgt.value === 'flax' || selTgt.value === 'jax') {
                    finalCode += `mesh = jax.sharding.Mesh(jax.devices(), ('data', 'model'))\\n`;
                    finalCode += `model = nnx.with_sharding(model, mesh)\\n`;
                } else if (selTgt.value === 'torch') {
                    finalCode += `from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\\n`;
                    finalCode += `model = FSDP(model)\\n`;
                } else {
                    finalCode += `# Sharding simulated for ${selTgt.value}\\n`;
                }
            } else {
                targetTitle.textContent = `${selTgt.options[selTgt.selectedIndex].text}`;
            }
            
            cmTarget.setValue(finalCode);
        }

        selStrategy.addEventListener('change', () => applyStrategy());
    });
})();
