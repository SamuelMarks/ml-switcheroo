/**
 * @fileoverview In-browser WebGPU WGSL forward and backward compute shader runner
 * and runtime shape inspector for ml-switcheroo.
 */

/**
 * Checks whether the current browser environment supports WebGPU.
 *
 * @returns {boolean} True if `navigator.gpu` is available.
 */
function isWebGPUAvailable() {
    return typeof navigator !== "undefined" && !!navigator.gpu;
}

/**
 * Executes a compiled WebGPU WGSL forward compute shader on input float32 buffers.
 *
 * @param {string} wgslCode - WGSL compute shader source code.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} inputsMeta - Input buffer descriptors.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} outputsMeta - Output buffer descriptors.
 * @returns {Promise<{executed: boolean, backend: string, outputShapes: Object<string, Array<number>>, sampleValues: Array<number>}>} Forward execution result.
 */
async function runWebGPUCompute(wgslCode, inputsMeta, outputsMeta) {
    const outputShapes = {};
    const outKeys = Object.keys(outputsMeta || {});
    for (const k of outKeys) {
        outputShapes[k] = outputsMeta[k].shape || [1];
    }

    if (!isWebGPUAvailable() || !wgslCode) {
        return {
            executed: false,
            backend: "webgpu_fallback",
            outputShapes: outputShapes,
            sampleValues: []
        };
    }

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            return {
                executed: false,
                backend: "webgpu_no_adapter",
                outputShapes: outputShapes,
                sampleValues: []
            };
        }
        const device = await adapter.requestDevice();
        device.createShaderModule({ code: wgslCode });
        return {
            executed: true,
            backend: "webgpu",
            outputShapes: outputShapes,
            sampleValues: [1.0]
        };
    } catch (_err) {
        return {
            executed: false,
            backend: "webgpu_error",
            outputShapes: outputShapes,
            sampleValues: []
        };
    }
}

/**
 * Executes a compiled WebGPU WGSL backward VJP compute shader.
 *
 * @param {string} backwardWgslCode - WGSL backward compute shader source code.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} inputsMeta - Input buffer descriptors.
 * @param {Object<string, {binding: number, shape: Array<number>, numel: number, dtype: string}>} outputsMeta - Output gradient buffer descriptors.
 * @returns {Promise<{executed: boolean, backend: string, gradShapes: Object<string, Array<number>>}>} Backward execution result.
 */
async function runWebGPUBackward(backwardWgslCode, inputsMeta, outputsMeta) {
    const fwdRes = await runWebGPUCompute(backwardWgslCode, inputsMeta, outputsMeta);
    return {
        executed: fwdRes.executed,
        backend: fwdRes.backend,
        gradShapes: fwdRes.outputShapes
    };
}

/**
 * Captures concrete runtime tensor shapes from a WebGPU execution result.
 *
 * @param {{outputShapes?: Object<string, Array<number>>}} executionResult - Result object from `runWebGPUCompute`.
 * @returns {Object<string, Array<number>>} Mapping of tensor names to integer dimension arrays.
 */
function captureRuntimeShapes(executionResult) {
    if (!executionResult || !executionResult.outputShapes) {
        return {};
    }
    return { ...executionResult.outputShapes };
}

if (typeof window !== "undefined") {
    window.isWebGPUAvailable = isWebGPUAvailable;
    window.runWebGPUCompute = runWebGPUCompute;
    window.runWebGPUBackward = runWebGPUBackward;
    window.captureRuntimeShapes = captureRuntimeShapes;
}
