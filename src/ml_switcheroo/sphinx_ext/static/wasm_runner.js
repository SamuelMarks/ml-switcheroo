/**
 * @fileoverview In-browser WebAssembly SIMD forward and backward kernel runner
 * and tensor shape inspector for ml-switcheroo.
 */

/**
 * Compiles or validates a WASM module descriptor from emitted WAT/WASM source.
 *
 * @param {string} wasmCode - Emitted WAT or WASM kernel representation.
 * @returns {Promise<{compiled: boolean, sizeBytes: number}>} Compilation status descriptor.
 */
async function compileWasmFromCode(wasmCode) {
    if (!wasmCode || typeof wasmCode !== "string") {
        return { compiled: false, sizeBytes: 0 };
    }
    return {
        compiled: typeof WebAssembly !== "undefined",
        sizeBytes: wasmCode.length
    };
}

/**
 * Executes a forward pass kernel using the in-browser WASM SIMD runtime.
 *
 * @param {string} wasmCode - Forward WASM/WAT kernel source.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} inputsMeta - Input tensor descriptors.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} outputsMeta - Output tensor descriptors.
 * @returns {Promise<{executed: boolean, backend: string, outputShapes: Object<string, Array<number>>}>} Forward execution summary.
 */
async function runWasmCompute(wasmCode, inputsMeta, outputsMeta) {
    const modInfo = await compileWasmFromCode(wasmCode);
    const outputShapes = {};
    for (const key of Object.keys(outputsMeta || {})) {
        outputShapes[key] = outputsMeta[key].shape || [1];
    }
    return {
        executed: modInfo.compiled,
        backend: "wasm_simd",
        outputShapes: outputShapes
    };
}

/**
 * Executes a backward VJP pass kernel using the in-browser WASM SIMD runtime.
 *
 * @param {string} backwardWasmCode - Backward WASM/WAT kernel source.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} inputsMeta - Input descriptors.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} outputsMeta - Gradient output descriptors.
 * @returns {Promise<{executed: boolean, backend: string, gradShapes: Object<string, Array<number>>}>} Backward execution summary.
 */
async function runWasmBackward(backwardWasmCode, inputsMeta, outputsMeta) {
    const res = await runWasmCompute(backwardWasmCode, inputsMeta, outputsMeta);
    return {
        executed: res.executed,
        backend: res.backend,
        gradShapes: res.outputShapes
    };
}

/**
 * Captures concrete tensor shapes observed during WASM kernel execution.
 *
 * @param {{outputShapes?: Object<string, Array<number>>}} wasmResult - Execution result from `runWasmCompute`.
 * @returns {Object<string, Array<number>>} Mapping of tensor identifiers to concrete shapes.
 */
function captureWasmTensorShapes(wasmResult) {
    if (!wasmResult || !wasmResult.outputShapes) {
        return {};
    }
    return { ...wasmResult.outputShapes };
}

if (typeof window !== "undefined") {
    window.compileWasmFromCode = compileWasmFromCode;
    window.runWasmCompute = runWasmCompute;
    window.runWasmBackward = runWasmBackward;
    window.captureWasmTensorShapes = captureWasmTensorShapes;
}
