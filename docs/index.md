ml-switcheroo 🔄🦘
==================

A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.

## Interactive Demo

Try it right in your browser! The engine runs entirely client-side using WebAssembly.

```{switcheroo_demo}
```

```mermaid
%%{init: {'flowchart': {'rankSpacing': 50, 'nodeSpacing': 20, 'padding': 35}}}%%
flowchart TD

%% --- 1. Font & Node Styling ---

%% Level 0: Red (Representations)
    classDef l0Node fill: #ea4335, stroke: #ff7daf, stroke-width: 2px, color: white, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 1: Blue (Frameworks)
    classDef l1Node fill: #4285f4, stroke: #57caff, stroke-width: 2px, color: white, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 2: Green (Numerical)
    classDef l2Node fill: #34a853, stroke: #5cdb6d, stroke-width: 2px, color: white, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 3: Yellow (Intermediate)
    classDef l3Node fill: #f9ab00, stroke: #ffd427, stroke-width: 2px, color: white, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 4: Teal (Native & Web)
    classDef l4Node fill: #00897b, stroke: #80cbc4, stroke-width: 2px, color: white, font-family: 'Roboto Mono Normal', font-size: 14px, rx: 3px, ry: 3px;

%% Level 5: Navy (Hardware ASM) - Roboto Mono
    classDef asmNode fill: #20344b, stroke: #57caff, stroke-width: 2px, color: white, font-family: 'Roboto Mono Normal', font-size: 14px, rx: 2px, ry: 2px;

%% --- 2. Subgraph Styling ---
%% White backgrounds to ensure text readability + visual grouping
    classDef containerL0 fill: white, stroke: #ea4335, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL1 fill: white, stroke: #4285f4, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL2 fill: white, stroke: #34a853, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL3 fill: white, stroke: #f9ab00, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL4 fill: white, stroke: #00897b, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerHW fill: white, stroke: #20344b, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;

%% --- 3. Diagram Structure ---

    subgraph L0 [Level 0: Representations]
        direction LR
        HTML
        TikZ
        LaTeX
    end

    subgraph L1 [Level 1: High-Level]
        direction LR
        PyTorch
        MLX
        TensorFlow
        Keras
        FlaxNNX[Flax NNX]
        Pax
    end

    subgraph L2 [Level 2: Numeric only]
        direction LR
        JAX
        NumPy
    end

    subgraph L3 [Level 3: Standard IR]
        direction LR
        StableHLO[Stable HLO]
        MLIR
        IR[ML-Switcheroo IR]
    end

    subgraph L4 [Level 4: Native & Binary]
        direction LR
        Cpp[C++ / PyBind11]
        WAT[WebAssembly WAT]
    end

    subgraph LBottom [Level 5: Hardware ASM]
        direction LR
        NVIDIA_SASS[NVIDIA SASS]
        RDNA[AMD RDNA]
    end

%% --- 4. Connections ---
    TikZ ~~~ TensorFlow
    TensorFlow ~~~ JAX
    JAX ~~~ StableHLO
    StableHLO ~~~ Cpp
    Cpp ~~~ NVIDIA_SASS

%% --- 5. Apply Styles ---
    class HTML,TikZ,LaTeX l0Node;
    class PyTorch,MLX,TensorFlow,Keras,FlaxNNX,Pax l1Node;
    class JAX,NumPy l2Node;
    class StableHLO,MLIR,IR l3Node;
    class Cpp,WAT l4Node;
    class NVIDIA_SASS,RDNA asmNode;
    class L0 containerL0;
    class L1 containerL1;
    class L2 containerL2;
    class L3 containerL3;
    class L4 containerL4;
    class LBottom containerHW;
```

```{toctree}
:maxdepth: 2
:caption: Documentation

README
ARCHITECTURE
EXTENDING
EXTENDING_WITH_DSL
MAINTENANCE
INTERNALS
IDEAS
```

```{toctree}
:maxdepth: 3
:caption: Reference

api/index
ops/index
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
