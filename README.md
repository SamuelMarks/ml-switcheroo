ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/license/apache-2-0)
[![Test and release](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Interactive docs](https://img.shields.io/badge/interactive-docs-silver)](https://samuelmarks.github.io/ml-switcheroo/)

**ml-switcheroo** is a rigorous AST-based transpiler designed to convert Deep Learning code between frameworks (PyTorch,
JAX/Flax, TensorFlow, etc.) without hallucination.

Unlike LLM-based assistants, it uses a strict **Semantic Knowledge Base** derived from official
specifications ([ONNX](https://github.com/onnx/onnx), [Python Array API](https://data-apis.org/array-api/latest/)) to
perform mathematically guaranteed translations. It solves the $O(N^2)$ translation problem using a Hub-and-Spoke model:
all frameworks map to an **Abstract Standard**, which then maps to the target.

---

### 🏗️ Architecture

Code is parsed into an Abstract Syntax Tree (AST), analyzed for safety, pivoted through the Abstract Standard, and
reconstructed for the target framework.

<!-- prettier-ignore -->

```mermaid
graph TD
    %% =================================================================================
    %% THEME CONFIGURATION
    %% Colors: Blue 500 (#4285f4), Green 500 (#34a853), Yellow 600 (#f9ab00), Red 500 (#ea4335)
    %% Navy (#20344b), White (#ffffff)
    %% =================================================================================
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b,stroke-width:1px;
    
    %% Input: Red 500
    classDef source fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    
    %% Processing: Blue 500
    classDef engine fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    
    %% Data/KB: Yellow 600
    classDef kb fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    
    %% Output: Green 500
    classDef target fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    
    %% Extensions: Halftone Blue
    classDef plugin fill:#57caff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    
    %% Verification: Halftone Green
    classDef verify fill:#5cdb6d,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;

    %% Code Blocks: White with Monospace
    classDef codeBlock fill:#ffffff,stroke:#20344b,stroke-width:1px,font-family:'Roboto Mono Normal',text-align:left,font-size:12px;

    %% =================================================================================
    %% 0. INPUT PHASE
    %% =================================================================================
    SRC_HEADER("<b>0. Input Source (PyTorch)</b>"):::source
    SRC_CODE("class Model(<b>nn.Module</b>):<br/>&nbsp;&nbsp;def <b>forward</b>(self, x):<br/>&nbsp;&nbsp;&nbsp;&nbsp;return <b>torch.abs</b>(x)"):::codeBlock
    
    SRC_HEADER --- SRC_CODE
    SRC_CODE --> PARSER

    %% =================================================================================
    %% 1. ANALYSIS PHASE
    %% =================================================================================
    PARSER("<b>1. AST Analysis</b><br/>Purity & Lifecycle Scanners"):::engine
    
    PARSER --> REWRITER

    %% =================================================================================
    %% 2. REWRITE PHASE (The Hub)
    %% =================================================================================
    
    %% Knowledge Base Side-Car
    KB[("<b>Knowledge Base</b><br/>semantics/*.json<br/>(Specs + Adapters)")]:::kb
    
    REWRITER("<b>2. Pivot Rewriter</b><br/><i>Semantic Translation</i>"):::engine
    
    KB -.->|" Map IDs "| REWRITER
    
    %% Logic Visualization inside Rewriter
    PIVOT_LOGIC("<b>1. Ingest:</b> torch.abs(x)<br/><b>2. Pivot:</b> Abs(x) [Standard]<br/><b>3. Project:</b> jnp.abs(x)"):::codeBlock
    REWRITER --- PIVOT_LOGIC
    
    PIVOT_LOGIC --> FIXER

    %% =================================================================================
    %% 3. REFINEMENT
    %% =================================================================================
    FIXER("<b>3. Import Fixer</b><br/>Injects 'jax.numpy'"):::engine
    
    FIXER --> TGT_HEADER

    %% =================================================================================
    %% 4. OUTPUT PHASE
    %% =================================================================================
    TGT_HEADER("<b>4. Output Target (JAX)</b>"):::target
    TGT_CODE("class Model(<b>nnx.Module</b>):<br/>&nbsp;&nbsp;def <b>__call__</b>(self, x):<br/>&nbsp;&nbsp;&nbsp;&nbsp;return <b>jnp.abs</b>(x)"):::codeBlock
    
    TGT_HEADER --- TGT_CODE
    TGT_CODE --> VERIFY

    %% =================================================================================
    %% 5. VERIFICATION LOOP
    %% =================================================================================
    VERIFY("<b>5. Verification Engine</b><br/>Symbolic Fuzzer & Harness"):::verify
    
    RESULT("<b>✅ Equivalence Confirmed</b><br/>Input: Array['B', 'C'] (f32)<br/>Diff: 0.000"):::codeBlock
    
    VERIFY --- RESULT
```

---

## ⚡ Core Capabilities

### 1. The Semantic Pivot

The engine maps source code to **Abstract Operations** (e.g., `Math.Abs`, `Neural.Conv2d`), then projects them to the
target framework.

* **Argument Normalization:** Pivots arguments via the spec standard (e.g., `torch.sum(input, dim)` $\rightarrow$
  `Standard(x, axis)` $\rightarrow$ `jax.sum(a, axis)`).
* **Logic Swaps:** Handles In-place unrolling (`x.add_(y)` $\rightarrow$ `x = x + y`), Decomposition (
  `torch.add(alpha=2)` $\rightarrow$ `x + y * 2`), and Infix/Prefix operators.

### 2. Structural & State Rewriting

Deep Learning isn't just math; it's state management.

* **Class Transpilation:** Converts `torch.nn.Module` $\leftrightarrow$ `flax.nnx.Module` $\leftrightarrow$
  `praxis.base_layer.BaseLayer`. Handles `super().__init__` stripping/injection and method renaming (
  `forward` $\leftrightarrow$ `__call__`).
* **RNG Threading (The "JAX Pointer"):** Detects stochastic operations (`dropout`) and injects explicit PRNG variables (
  `rng`, `key`) into signatures and function bodies.
* **Lifecycle Management:** Strips framework-specific idioms (`.to(device)`, `.detach()`, `.cpu()`) while preserving
  logic. Flags imperative state changes like `.eval()` or `.train()`.

### 3. Safety & Analysis

* **Purity Scanning:** Static analysis detects side effects (I/O, global mutation, list appends) unsafe for JIT
  compilation.
* **Dependency Scanning:** Flags 3rd-party imports (`pandas`, `cv2`) not covered by the semantic map to prevent runtime
  crashes.
* **Lifecycle Tracking:** Ensures class members used in `forward` are validly initialized in `__init__`, critical for
  static graph compilation (XLA).
* **Smart Imports:** The `ImportFixer` intelligently injects imports (`import jax.numpy as jnp`) *only* if the
  translated code actually uses them.

### 4. Verification Engine

* **Symbolic Fuzzing:** The `InputFuzzer` generates valid inputs based on Type Hints extracted from specs (support for
  `Tensor['B', 'C']` symbolic shapes).
* **Harness Generator:** Produces standalone verification scripts that run Source vs Target logic side-by-side to prove
  equivalence.

---

## 📦 Installation

```bash
pip install .
# OR for development
pip install -e ".[test]"
```

---

## 🛠 CLI Usage

### Transpilation

Convert files or entire directories. Defaults to **PyTorch $\rightarrow$ JAX**.

```bash
# Basic conversion
ml_switcheroo convert ./models/resnet.py --out ./models_jax/

# Specify Frameworks manually
ml_switcheroo convert ./src --source torch --target tensorflow --out ./dst

# Strict Mode: Fails/Marks unknown APIs instead of passing them through
ml_switcheroo convert ./src --strict --out ./out --json-trace trace.json
```

### Discovery & Learning

Populate the Knowledge Base automatically.

```bash
# 1. Scaffold: Scan installed libs (e.g. torch, jax) and align against Standards
ml_switcheroo scaffold --frameworks torch jax

# 2. Wizard: Interactive tool to categorize unmapped APIs and assign plugins
ml_switcheroo wizard torch

# 3. Harvest: Learn mappings from your manually written test cases
ml_switcheroo harvest tests/examples/test_custom_layer.py --target jax
```

### Verification (CI)

```bash
# Run validation suite on all known mappings
ml_switcheroo ci

# Generate a standalone lockfile of verified operations
ml_switcheroo ci --json-report verified_ops.json
```

---

## ✅ Compatibility Matrix

Supported Frameworks via **Zero-Edit Adapters**:

| Framework          | Adapter Status  | Features Supported                      |
|:-------------------|:---------------:|:----------------------------------------|
| **PyTorch**        |   🟢 Primary    | Source / Target, NN Modules, Lifecycle  |
| **JAX / Flax**     |   🟢 Primary    | Source / Target (NNX), RNG Threading    |
| **NumPy**          |  🟡 Supported   | Fallback Target, Verification Backend   |
| **TensorFlow**     |     🔵 Beta     | Keras Layers, IO, Device Placements     |
| **Apple MLX**      | 🔵 Experimental | Basic Array Ops, Device Abstraction     |
| **PaxML / Praxis** | 🔵 Experimental | Layer Setup Migration, Context Patterns |

---

## 🔌 Extensibility

ml-switcheroo is designed to be extended without modifying the core engine.

1. **New Frameworks:** Add a file to `src/ml_switcheroo/frameworks/` implementing the `FrameworkAdapter` protocol.
2. **New Logic:** Add a hook to `src/ml_switcheroo/plugins/` (e.g. `register_hook("my_custom_pattern")`) and link it in
   the semantics JSON.
3. **New Mappings:** Use `ml_switcheroo wizard` or edit `semantics/*.json` directly.

See [EXTENDING.md](EXTENDING.md) for a detailed guide.

---

## License

[Apache-2.0 License](LICENSE)
