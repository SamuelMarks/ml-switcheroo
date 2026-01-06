ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/license/apache-2-0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Interactive docs](https://img.shields.io/badge/interactive-docs-orange)](https://samuelmarks.github.io/ml-switcheroo/)

**ml-switcheroo** is a rigorous AST-based transpiler designed to convert Deep Learning code between frameworks (e.g., *
*PyTorch** $\leftrightarrow$ **JAX**; **Keras** $\leftrightarrow$ **TensorFlow**; etc.) without hallucination.

It uses a **Hub-and-Spoke** architecture to solve the $O(N^2)$ translation problem. Instead of writing translators for
every pair of frameworks, `ml-switcheroo` maps all frameworks to a central **Abstract Standard** (Hub). This allows
for "Zero-Edit" support for new frameworks via isolated JSON snapshots (Spokes).

Recently this evolved from a transpiler into a fully-fledged compiler with multiple higher and lower levels of
abstractions, all of which are almost completely interchangeable ($\leftrightarrow$ to each other):

```mermaid
%%{init: {'flowchart': {'rankSpacing': 50, 'nodeSpacing': 20, 'padding': 35}}}%%
flowchart TD

%% --- 1. Font & Node Styling ---

%% Level 0: Red (Representations)
    classDef l0Node fill: #ea4335, stroke: #ff7daf, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 1: Blue (Frameworks)
    classDef l1Node fill: #4285f4, stroke: #57caff, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 2: Green (Numerical)
    classDef l2Node fill: #34a853, stroke: #5cdb6d, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Level 3: Yellow (Intermediate)
    classDef l3Node fill: #f9ab00, stroke: #ffd427, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Normal', font-size: 16px, rx: 5px, ry: 5px;

%% Hardware: Navy (SASS) - Roboto Mono
    classDef sassNode fill: #20344b, stroke: #57caff, stroke-width: 2px, color: #ffffff, font-family: 'Roboto Mono Normal', font-size: 14px, rx: 2px, ry: 2px;

%% --- 2. Subgraph Styling --- 
%% White backgrounds to ensure text readability + visual grouping
    classDef containerL0 fill: #ffffff, stroke: #ea4335, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL1 fill: #ffffff, stroke: #4285f4, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL2 fill: #ffffff, stroke: #34a853, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerL3 fill: #ffffff, stroke: #f9ab00, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;
    classDef containerHW fill: #ffffff, stroke: #20344b, stroke-width: 3px, color: #20344b, font-family: 'Google Sans Medium', font-size: 20px;

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
    class SASS sassNode;
    class L0 containerL0;
    class L1 containerL1;
    class L2 containerL2;
    class L3 containerL3;
    class LBottom containerHW;
```

---

## 🚀 Key Features

* **🚫 No Hallucinations**: Uses static analysis (AST) and deterministic mapping rules. If it compiles, it's
  mathematically grounded.
* **📝 ODL (Operation Definition Language)**: Define new mathematical operations using a simple YAML syntax without
  writing Python AST code.
* **🔌 Hub-and-Spoke Architecture**: Decouples the *semantic definition* of an operation (e.g., `Conv2d`) from its
  *implementation* (e.g., `torch.nn.Conv2d`).
* **👻 Ghost Mode**: Can analyze and transpile code for frameworks *not installed* on the local machine using cached API
  snapshots.
* **🛡️ Safety Logic**: Automatically detects side-effects (IO, globals) that break functional compilation (JIT) via the
  **Purity Scanner**.
* **🧬 Structural Rewriting**: Handles complex transformations for class hierarchies (e.g., `nn.Module` $\leftrightarrow$
  `flax.nnx.Module`), random number threading, and state management.

---

## 🏗️ Architecture

Code is parsed into an Abstract Syntax Tree (AST), analyzed for safety, pivoted through the Abstract Standard, and
reconstructed for the target framework.

```mermaid
graph TD
%% =========================================================================
%%  DESIGN SYSTEM & PALETTE
%% =========================================================================

%% Colors
%% Blue 500: #4285f4 | Green 500: #34a853 | Yellow 600: #f9ab00 | Red 500: #ea4335
%% Navy: #20344b | White: #ffffff 
%% Halftone Blue: #57caff | Halftone Green: #5cdb6d 
%% Halftone Yellow: #ffd427 | Halftone Red: #ff7daf

%% Fonts
    classDef default font-family: 'Google Sans Normal', color: #20344b, stroke: #20344b, stroke-width: 1px;
    classDef title font-family: 'Google Sans Medium', font-size: 12px, color: #ffffff, stroke-width: 0px, rx: 4px;
    classDef code font-family: 'Roboto Mono Normal', font-size: 11px, text-align: left, fill: #ffffff, color: #20344b, stroke: #20344b, stroke-dasharray: 2 2, rx: 0;
    classDef db font-family: 'Google Sans Normal', font-size: 11px, fill: #fff4c7, stroke: #f9ab00, stroke-width: 1px, rx: 2px;

%% Node Types
    classDef src fill: #ea4335, color: #ffffff;
    classDef eng fill: #4285f4, color: #ffffff;
    classDef hub fill: #f9ab00, color: #20344b;
    classDef plug fill: #57caff, color: #20344b;
    classDef tgt fill: #34a853, color: #ffffff;
    classDef ghost fill: #20344b, color: #ffffff, stroke-dasharray: 2 2;

%% =========================================================================
%%  1. SOURCE INPUT
%% =========================================================================
    S_HEAD("<b>1. Source Code (PyTorch)</b>"):::src
    S_HEAD:::title
%% Syntax Highlighted HTML Label
    S_CODE("
    <span style='color:#4285f4'>import</span> torch.nn <span style='color:#4285f4'>as</span> nn<br/>
<span style='color:#4285f4'>class</span> <span style='color:#ea4335'>Net</span>(nn.Module):<br/>
&nbsp;&nbsp;<span style='color:#4285f4'>def</span> <span style='color:#f9ab00'>__init__</span>(<span style='color:#57caff'>self</span>):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#57caff'>self</span>.fc = nn.Linear(<span style='color:#34a853'>20</span>, <span style='color:#34a853'>30</span>)<br/>
"):::code

S_HEAD --- S_CODE

%% =========================================================================
%%  2. INGESTION & ANALYSIS
%% =========================================================================

P_LIBCST("<b>LibCST Parser</b><br/><i>Generates AST</i>"):::eng
P_LIBCST:::title

S_CODE --> P_LIBCST

%% Ghost / Live Context
subgraph CONTEXT ["Reflection Context"]
direction TB
GHOST("<b>Ghost Snapshot</b><br/><i>torch_v2.1.json</i>"):::ghost
LIVE("<b>Live Library</b><br/><i>import torch</i>"):::ghost
end

GHOST -.->|" API Signatures "|P_LIBCST
LIVE -.->|" Introspection "|P_LIBCST

%% =========================================================================
%%  3. SEMANTIC PIVOT (The Hub)
%% =========================================================================

HUB_HEAD("<b>Semantics Manager</b>"):::hub
HUB_HEAD:::title
P_LIBCST --> HUB_HEAD

%% The Knowledge Base (JSONs)
JSON_DB[("<b>Knowledge Base</b><br/><i>semantics/k_neural.json</i><br/><i>snapshots/jax_map.json</i>")]:::db
JSON_DB -.->|" 1. Lookup 'Linear'<br/>2. Read Constraints "|HUB_HEAD

%% Intermediate Representation
ABS_NODE("
<b>Abstract Operation found:</b><br/>
Op: <span style='color:#f9ab00'>Linear</span><br/>
Tier: <span style='color:#ea4335'>Neural</span> (Stateful)<br/>
Args: {in: 20, out: 30}<br/>
"):::code
HUB_HEAD --- ABS_NODE

%% =========================================================================
%%  4. REWRITING & PLUGINS
%% =========================================================================

REWRITE("<b>Pivot Rewriter</b>"):::eng
REWRITE:::title
ABS_NODE --> REWRITE

%% Plugin Logic
subgraph PLUGINS ["Extension System"]
target_trait("<b>Target Traits (JAX)</b><br/>requires_explicit_rng: <span style='color:#34a853'>True</span>"):::db

HOOK_DEF("<b>Plugin: rng_threading</b><br/><i>Injects 'rngs' arg into<br/>stateful layer calls</i>"):::plug
HOOK_DEF:::title

target_trait -.-> HOOK_DEF
end

REWRITE <-->|" AST Transformation "|HOOK_DEF

%% =========================================================================
%%  5. OUTPUT GENERATION
%% =========================================================================

FIXER("<b>Import Fixer</b><br/><i>Resolves 'nnx' alias</i>"):::plug
FIXER:::title
REWRITE --> FIXER

T_HEAD("<b>Target Code (Flax NNX)</b>"):::tgt
T_HEAD:::title
FIXER --> T_HEAD

%% Final Code
T_CODE("
<span style='color:#4285f4'>from</span> flax <span style='color:#4285f4'>import</span> nnx<br/>
<span style='color:#4285f4'>class</span> <span style='color:#34a853'>Net</span>(nnx.Module):<br/>
&nbsp;&nbsp;<span style='color:#4285f4'>def</span> <span style='color:#f9ab00'>__init__</span>(<span style='color:#57caff'>self</span>, <span style='color:#ea4335'>rngs</span>: nnx.Rngs):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#90a4ae'># Hook injected 'rngs'</span><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#57caff'>self</span>.fc = nnx.Linear(<span style='color:#34a853'>20</span>, <span style='color:#34a853'>30</span>, <span style='color:#ea4335'>rngs=rngs</span>)<br/>
"):::code

T_HEAD --- T_CODE
```

---

## 📦 Installation

```bash
# Install form source
pip install .

# Install with testing dependencies (for running the fuzzer/verification)
pip install ".[test]"
```

---

## 🛠️ CLI Usage

The `ml_switcheroo` tool provides a suite of commands for conversion, auditing, and knowledge base maintenance.

### 1. Transpilation (`convert`)

Convert a file or directory from one framework to another.

```bash
# Convert a PyTorch model to JAX (Flax NNX)
ml_switcheroo convert ./models/resnet.py \
    --source torch \
    --target jax \
    --out ./resnet_jax.py

# Convert an entire directory, enabling strict mode
# Strict mode fails if an API mapping is missing, rather than passing it through.
ml_switcheroo convert ./src/ --out ./dst/ --strict
```

### 2. Codebase Audit (`audit`)

Analyze a codebase to check "Translation Readiness". This scans API calls and checks coverage against the Knowledge
Base.

```bash
ml_switcheroo audit ./my_project/ --roots torch
```

### 3. Verification (`ci`)

The CI command runs the built-in **Fuzzer**. It generates random inputs (Tensors, Scalars) based on Type Hints in the
Spec, feeds them into both Source and Target frameworks, and mathematically verifies equivalence.

```bash
# Run full verification suite on the Knowledge Base
ml_switcheroo ci

# Generate a lockfile of verified operations
ml_switcheroo ci --json-report verified_ops.json
```

### 4. Knowledge Discovery (`scaffold` & `wizard`)

Populate the Knowledge Base automatically by scanning installed libraries.

```bash
# 1. Scaffold: Scan installed libs and generate JSON mappings via heuristics
ml_switcheroo scaffold --frameworks torch jax

# 2. Wizard: Interactive tool to manualy categorize obscure APIs
ml_switcheroo wizard torch
```

### 5. Operation Definition (`define`)

Inject new operations into the Knowledge Base using declarative YAML files.

```bash
ml_switcheroo define my_ops.yaml
```

---

## ✅ API Support Matrix

Supported Frameworks via **Zero-Edit Adapters**:

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

Frameworks like **JAX** require pure functions. ml-switcheroo automatically detects stateful imperative patterns (like
`drop_last=True` in loops or in-place lists) and warns via the **Purity Scanner**.
When converting **Flax NNX** (functional) to **Torch** (OO), it unwraps `layer.apply(params, x)` calls into standard
`layer(x)` calls using `Assign` restructuring.

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

1. **Add Operations (ODL)**: Use the **Operation Definition Language (YAML)** to define math/neural ops. This is the
   recommended way to add missing functionality.

   ```yaml
   operation: "Erf"
   std_args: [ "input" ]
   variants:
     torch: { api: "torch.erf" }
     jax: { api: "jax.lax.erf" }
   ```
   See [EXTENDING_WITH_DSL.md](EXTENDING_WITH_DSL.md) for the full guide. Alternative to the YAML DSL you can manually
   update:
    - `src/ml_switcheroo/semantics/standards_internal.py` and
    - `src/ml_switcheroo/frameworks/definitions/*.json` (for torch, mlx, tensorflow, jax, etc.)

2. **Add a Framework**: Create a class inheriting `FrameworkAdapter` in `src/ml_switcheroo/frameworks/`.
3. **Add Logic**: Write a localized hook in `src/ml_switcheroo/plugins/` (e.g., for custom layer rewrites like
   `MultiHeadAttention` packing).

See [EXTENDING.md](EXTENDING.md) for architectural details on Adapters and Plugins.

---

## License

[Apache-2.0](LICENSE)
