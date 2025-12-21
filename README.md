ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/license/apache-2-0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**ml-switcheroo** is a rigorous AST-based transpiler designed to convert Deep Learning code between frameworks (e.g., *
*PyTorch** $\leftrightarrow$ **JAX**, **Keras** $\to$ **TensorFlow**) without hallucination.

It uses a **Hub-and-Spoke** architecture to solve the $O(N^2)$ translation problem. Instead of writing translators for
every pair of frameworks, `ml-switcheroo` maps all frameworks to a central **Abstract Standard** (Hub). This allows
for "Zero-Edit" support for new frameworks via isolated JSON snapshots (Spokes).

---

## 🚀 Key Features

* **🚫 No Hallucinations**: Uses static analysis (AST) and deterministic mapping rules. If it compiles, it's
  mathematically grounded.
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

<!-- prettier-ignore -->
```mermaid
graph TD
    %% Theme
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b,stroke-width:1px;
    classDef source fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef engine fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef spoke fill:#fff4c7,stroke:#f9ab00,stroke-width:2px,stroke-dasharray: 5 5,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef target fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef codeBlock fill:#ffffff,stroke:#20344b,stroke-width:1px,font-family:'Roboto Mono Normal',text-align:left,font-size:12px;

    SRC_HEADER("<b>0. Source Code</b><br/>(e.g., PyTorch)"):::source
    PARSER("<b>1. Analysis Phase</b><br/>Parsing, Purity Check,<br/>Lifecycle Scans"):::engine
    
    SRC_HEADER --> PARSER

    %% The Knowledge Base
    subgraph KB [Distributed Knowledge Base]
        direction TB
        SPECS[("<b>The Hub (Specs)</b><br/>semantics/*.json<br/><i>Abstract Operations</i>")]:::hub
        MAPS[("<b>The Spokes (Overlays)</b><br/>snapshots/*_mappings.json<br/><i>Framework Variants</i>")]:::spoke
        
        MAPS -.->|" Hydrates "| SPECS
    end
    
    REWRITER("<b>2. Pivot Rewriter</b><br/><i>Semantic Translation</i>"):::engine
    KB -.->|" Lookup API "| REWRITER
    PARSER --> REWRITER

    PIVOT_LOGIC("<b>1. Ingest:</b> torch.abs(x)<br/><b>2. Pivot:</b> Abs(x) [Standard]<br/><b>3. Project:</b> jnp.abs(x)"):::codeBlock
    REWRITER --- PIVOT_LOGIC
    
    FIXER("<b>3. Refinement</b><br/>Import Injection & Pruning"):::engine
    PIVOT_LOGIC --> FIXER

    TGT_HEADER("<b>4. Target Code</b><br/>(e.g., JAX/Flax)"):::target
    FIXER --> TGT_HEADER
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

1. **Add a Framework**: Create a class inheriting `FrameworkAdapter` in `src/ml_switcheroo/frameworks/`.
2. **Add Definitions**: Run `ml_switcheroo scaffold` or edit `snapshots/{fw}_mappings.json`.
3. **Add Logic**: Write a localized hook in `src/ml_switcheroo/plugins/` (e.g., for custom layer rewrites like
   `MultiHeadAttention` packing).

See [EXTENDING.md](EXTENDING.md) for a detailed guide.

---

## License

[Apache-2.0](LICENSE)
