ml-switcheroo 🔄🦘
==================

**A Universal Compiler for Deep Learning: From High-Level APIs to Hardware Assembly.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/license/apache-2-0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test and release](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Interactive docs](https://img.shields.io/badge/interactive-docs-orange)](https://samuelmarks.github.io/ml-switcheroo/)

**ml-switcheroo** has evolved from a simple AST transpiler into a deterministic **Universal Compiler** for Machine Learning. It enables loss-less conversion between distinct levels of the ML stack: from high-level frameworks (PyTorch, JAX), to intermediate representations (StableHLO), down to hardware assembly (SASS, RDNA), and even into visual documentation formats (TikZ, HTML).

It solves the $O(N^2)$ interoperability problem using a **Hub-and-Spoke** architecture. Instead of writing translators for every pair of languages, we map every dialect to a central **Abstract Standard** (Hub).

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

%% Hardware: Navy (SASS) - Roboto Mono
    classDef asmNode fill: #20344b, stroke: #57caff, stroke-width: 2px, color: white, font-family: 'Roboto Mono Normal', font-size: 14px, rx: 2px, ry: 2px;

%% --- 2. Subgraph Styling --- 
%% White backgrounds to ensure text readability + visual grouping
    classDef containerL0 fill: white, stroke: #ea4335, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL1 fill: white, stroke: #4285f4, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL2 fill: white, stroke: #34a853, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL3 fill: white, stroke: #f9ab00, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
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
    end

    subgraph LBottom [Level 4: ASM]
        direction LR
        SASS[NVIDIA SASS]
        RDNA[AMD RDNA]
    end

%% --- 4. Connections ---
    TikZ ~~~ TensorFlow
    TensorFlow ~~~ JAX
    JAX ~~~ StableHLO
    StableHLO ~~~ SASS
%% --- 5. Apply Styles ---
    class HTML,TikZ,LaTeX l0Node;
    class PyTorch,MLX,TensorFlow,Keras,FlaxNNX,Pax l1Node;
    class JAX,NumPy l2Node;
    class StableHLO,MLIR l3Node;
    class SASS asmNode;
    class RDNA asmNode;
    class L0 containerL0;
    class L1 containerL1;
    class L2 containerL2;
    class L3 containerL3;
    class LBottom containerHW;
```

---

## 🚀 Key Capabilities

### 1. Syntactic Transpilation (Python ↔ Python)
Convert model code between frameworks with semantic fidelity.
*   **PyTorch** ↔ **JAX / Flax**
*   **Keras** ↔ **TensorFlow** ↔ **MLX**
*   Handles class rewriting (`nn.Module` -> `nnx.Module`), state injection (RNG keys), and functional unwrapping.

### 2. Architecture Visualization (Python → Visuals)
Compile your Python code directly into diagramming languages.
*   **Target: TikZ**: Generates professional LaTeX code for academic papers.
*   **Target: HTML**: Generates interactive Grid CSS layouts for documentation.

### 3. Assembly Decompilation (ASM → Python)
Lift low-level hardware instructions into readable high-level logic.
*   **Sources**: NVIDIA SASS (Ampere/Hopper), AMD RDNA (GFX10/11).
*   Reconstructs loops (e.g. `Conv2d` kernels) from raw assembly streams using topological graph analysis.

### 4. Weight Migration (Checkpointing)
Generate standalone scripts to convert model weights between formats.
*   Reads source AST to determine layer mappings.
*   Generates `orbax` / `torch.save` / `safetensors` migration logic.
*   Automatically handles NCHW ↔ NHWC layout permutation.

---

## 🏗️ Architecture

The engine uses a dual-path pipeline to handle both structured code (Python) and linear streams (ASM).

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef default font-family:'Google Sans',color:#20344b,stroke:#20344b,stroke-width:1px;
    classDef title font-family:'Google Sans Medium',font-size:12px,color:white,stroke-width:0px,rx:4px;
    classDef code font-family:'Roboto Mono',font-size:10px,text-align:left,fill:white,color:#20344b,stroke:#20344b,stroke-dasharray:2 2,rx:0;
    classDef db font-family:'Google Sans',font-size:11px,fill:#fff4c7,stroke:#f9ab00,stroke-width:1px,rx:2px;

    classDef src fill:#ea4335,color:white;
    classDef eng fill:#4285f4,color:white;
    classDef hub fill:#f9ab00,color:#20344b;
    classDef plug fill:#57caff,color:#20344b;
    classDef tgt fill:#34a853,color:white;
    classDef ghost fill:#20344b,color:white,stroke-dasharray:2 2;

    %% 1. SOURCE
    S_HEAD("<b>1. Source Code (PyTorch)</b>"):::src
    S_HEAD:::title

    S_CODE["import torch.nn as nn<br/>class ConvNet(nn.Module):<br/>  def __init__(self):<br/>    self.conv = nn.Conv2d(1, 32, 3)<br/>  def forward(self, x):<br/>    x = torch.flatten(x, 1)"]:::code
    S_HEAD --- S_CODE

    %% 2. PARSING & ANALYSIS
    P_LIBCST("<b>LibCST Parser</b><br/><i>Generates AST</i>"):::eng
    P_LIBCST:::title
    S_CODE --> P_LIBCST

    subgraph CONTEXT ["Reflection Context"]
      direction TB
      GHOST("<b>Ghost Snapshot</b><br/><i>torch_v2.1.json</i>"):::ghost
      LIVE("<b>Live Library</b><br/><i>import torch</i>"):::ghost
    end
    GHOST -.->|" API Signatures "|P_LIBCST
    LIVE -.->|" Introspection "|P_LIBCST

    %% 3. KNOWLEDGE LOOKUP
    HUB_HEAD("<b>Semantics Manager</b>"):::hub
    HUB_HEAD:::title
    P_LIBCST --> HUB_HEAD

    JSON_DB[("<b>Knowledge Base</b><br/><i>semantics/k_neural.json</i><br/><i>snapshots/jax_map.json</i>")]:::db
    JSON_DB -.->|" 1. Lookup 'Conv2d'<br/>2. Read Constraints "|HUB_HEAD

    ABS_NODE("<b>Abstract Operation Found:</b><br/>Op: Conv2d<br/>Tier: Neural (Stateful)<br/>Args: {in: 1, out: 32, k: 3}"):::code
    HUB_HEAD --- ABS_NODE

    %% 4. REWRITING REWIRING
    REWRITE("<b>Pivot Rewriter</b>"):::eng
    REWRITE:::title
    ABS_NODE --> REWRITE

    subgraph PLUGINS ["Extension System"]
      direction TB
      target_trait("<b>Target Traits (JAX)</b><br/>requires_explicit_rng: True"):::db
      
      HOOK_DEF("<b>Plugin: rng_threading</b><br/><i>Injects 'rngs' arg into<br/>stateful layer calls</i>"):::plug
      HOOK_DEF:::title
      
      HOOK_FLAT("<b>Plugin: flatten_range</b><br/><i>Maps flatten(x, 1)<br/>to nnx.Flatten</i>"):::plug
      HOOK_FLAT:::title

      target_trait -.-> HOOK_DEF
    end
    
    REWRITE <-->|" State Injection "|HOOK_DEF
    REWRITE <-->|" API Swap "|HOOK_FLAT

    %% 5. REFINEMENT
    FIXER("<b>Import Fixer</b><br/><i>Resolves 'nnx' alias</i>"):::plug
    FIXER:::title
    REWRITE --> FIXER

    %% 6. TARGET
    T_HEAD("<b>Target Code (Flax NNX)</b>"):::tgt
    T_HEAD:::title
    FIXER --> T_HEAD

    T_CODE["from flax import nnx<br/>class ConvNet(nnx.Module):<br/>  def __init__(self, rngs: nnx.Rngs):<br/>    # Variable Injection<br/>    self.conv = nnx.Conv(1, 32, 3, rngs=rngs)<br/>  def __call__(self, x):<br/>    x = nnx.Flatten(x, 1)"]:::code
    T_HEAD --- T_CODE
```

---

## 📦 Installation

```bash
# Install from source
pip install .

# Install with testing dependencies (necessary for Fuzzer/Verification)
pip install ".[test]"
```

---

## 🛠️ CLI Usage

The `ml_switcheroo` CLI is your gateway to the compiler stack.

### 1. Code Conversion (`convert`)
Transpile source code or decompile assembly.

```bash
# Standard: Torch -> JAX
ml_switcheroo convert ./models/resnet.py --target jax --out ./resnet_jax.py

# Visualization: Python -> LaTeX (TikZ)
ml_switcheroo convert ./models/transformer.py --target tikz --out ./diagram.tex

# Decompilation: SASS -> Python
ml_switcheroo convert ./kernels/gemm.sass --source sass --target python
```

### 2. Weight Migration (`gen-weight-script`)
Generate a script to migrate trained weights (checkpoints) to a new framework.

```bash
# Generate a conversion script
ml_switcheroo gen-weight-script ./src_model.py \
    --source torch --target jax \
    --out ./migrate_weights.py

# Run the generated script
python migrate_weights.py input.pth output_ckpt/
```

### 3. Verification (`ci`)
Run the mathematical fuzzer to verify the correctness of the Knowledge Base.

```bash
# Runs hypothesis tests on all mapped operations across installed frameworks
ml_switcheroo ci --json-report verified_ops.json
```

### 4. Discovery & Autogen (`suggest`, `define`)
"Teach" the compiler new operations using LLM assistance and ODL (Operation Definition Language).

```bash
# 1. Generate an LLM prompt with introspection data
ml_switcheroo suggest 'torch.nn.functional.scaled_dot_product_attention' > prompt.md

# 2. (Paste prompt to LLM, get YAML back)

# 3. Inject the new definition into the Knowledge Base
ml_switcheroo define new_ops.yaml
```

---

## ✅ Compatibility Matrix

Core framework support status via **Zero-Edit Adapters**:

| Framework      |   Status   | Specialized Features Supported                                         |
|:---------------|:----------:|:-----------------------------------------------------------------------|
| **PyTorch**    | 🟢 Primary | Source/Target, `nn.Module`, `functional`, Optimizers, DataLoaders      |
| **JAX / Flax** | 🟢 Primary | Source/Target (`flax.nnx`), `vmap`, `grad`, `jit`, Orbax Checkpointing |
| **TensorFlow** |  🔵 Beta   | Keras Layer conversion, `tf.data`, IO operations                       |
| **NumPy**      | 🟡 Stable  | Array operations, fallback target for pure math                        |
| **Keras 3**    |  🔵 Beta   | Multi-backend layers, `keras.ops` math                                 |
| **Apple MLX**  |  🔵 Beta   | `mlx.nn` layers, `mlx.core` array ops, Optimizers                      |
| **PaxML**      |  ⚪ Alpha   | `praxis` layer structure translation                                   |

To view the live compatibility table for your installed version:

```bash
ml_switcheroo matrix
```

---

## 🧠 Advanced Capabilities

### Functional Unwrapping
Frameworks like **JAX** require pure functions. ml-switcheroo automatically detects stateful imperative patterns (like `drop_last=True` in loops or in-place lists) and warns via the **Purity Scanner**.
When converting **Flax NNX** (functional) to **Torch** (OO), it unwraps `layer.apply(params, x)` calls into standard `layer(x)` calls using `Assign` restructuring.

### State Injection (RNG Threading)
When converting **PyTorch** (global RNG state) to **JAX** (explicit RNG keys), the engine:
1. Detects stochastic operations (Dropout, Random init) via the **Analyzer**.
2. Injects an `rng` argument into function signatures.
3. Injects `rng, key = jax.random.split(rng)` preambles.
4. Threads the `key` argument into relevant function calls.

### Intelligent Import Management
The **Import Fixer** does not just swap strings; it analyzes usage logic:
* Removes unused source imports (`import torch`).
* Injects required target imports (`import jax.numpy as jnp`) only if referenced.
* Handles alias conflicts (`import torch as t`).

---

## 🔌 Extensibility

ml-switcheroo is designed to be extended without modifying the core engine.

1. **Add Operations (ODL)**: Use the **Operation Definition Language (YAML)** to define math/neural ops. This is the recommended way to add missing functionality.
   See [EXTENDING_WITH_DSL.md](EXTENDING_WITH_DSL.md) for the full guide.

2. **Add a Framework**: Create a class inheriting `FrameworkAdapter` in `src/ml_switcheroo/frameworks/`.
   See [EXTENDING.md](EXTENDING.md) for architectural details on Adapters and Plugins.

---

## License

[Apache-2.0](LICENSE)
