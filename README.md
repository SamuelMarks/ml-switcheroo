ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/)
[![Test and release](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Interactive docs](https://img.shields.io/badge/interactive-docs-silver)](https://samuelmarks.github.io/ml-switcheroo/)

**ml-switcheroo** converts Deep Learning models between frameworks (primarily **PyTorch** ↔ **JAX/Flax**) using strict
Abstract Syntax Tree (AST) / Concrete Syntax Tree (CST) manipulation.

Unlike LLM-based coding assistants which "guess" output, ml-switcheroo uses a **Knowledge Base** derived from official
specifications ([ONNX](https://github.com/onnx/onnx), [Python Array API]((https://data-apis.org/array-api/latest/))) to perform mathematically guaranteed translations. If a translation is
ambiguous, it protects your code with an "Escape Hatch" rather than generating broken logic.

---

Here is a diagram visualizing the flow from Specifications/Code → Discovery → The Knowledge Base → The
Transpilation Engine.

<!-- prettier-ignore -->

```mermaid
graph TD
%% =================================================================================
%% STYLE DEFINITIONS
%% =================================================================================
    classDef source fill: #ea4335, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Medium', rx: 5px, ry: 5px;
    classDef semantics fill: #f9ab00, stroke: #20344b, stroke-width: 2px, color: #20344b, font-family: 'Google Sans Medium', rx: 5px, ry: 5px;
    classDef engine fill: #4285f4, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Medium', rx: 5px, ry: 5px;
    classDef plugin fill: #57caff, stroke: #20344b, stroke-width: 2px, color: #20344b, font-family: 'Google Sans Medium', rx: 5px, ry: 5px;
    classDef target fill: #34a853, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans Medium', rx: 5px, ry: 5px;
    classDef codeBlock fill: #ffffff, stroke: #20344b, stroke-width: 1px, color: #20344b, font-family: 'Roboto Mono Normal', text-align: left;

%% =================================================================================
%% 1. INPUT LAYER
%% =================================================================================
    subgraph Inputs [" Phase 1: Inputs "]
        direction TB
        style Inputs fill: #ffffff, stroke: #20344b, stroke-width: 0px, color: #20344b
        SRC_FILE("<b>📄 Input Source (PyTorch)</b>"):::source
        SRC_CODE("import torch.nn as nn<br/>import torch<br/><br/>class Model(<b>nn.Module</b>):<br/>&nbsp;&nbsp;def <b>forward</b>(self, x):<br/>&nbsp;&nbsp;&nbsp;&nbsp;return <b>torch.abs</b>(x)"):::codeBlock
        SRC_FILE --- SRC_CODE
    end

    SRC_CODE -->|" Parse AST "| PARSER
%% =================================================================================
%% 2. ENGINE LAYER
%% =================================================================================
    subgraph Core [" ml-switcheroo Engine "]
        direction TB
        style Core fill: #edf2fa, stroke: #20344b, stroke-width: 2px, color: #20344b, font-family: 'Google Sans Medium'
        PARSER(AST Parser):::engine
        ANALYSIS("<b>🔍 Analysis</b><br/>Purity, Lifecycle & Deps"):::engine
    %% Semantic Lookup
        KB_HEADER("<b>📚 Knowledge Base</b>"):::semantics
        KB_DATA("{<br/>&nbsp;'abs': {'jax': 'jax.numpy.abs'},<br/>&nbsp;'Linear': {'jax': 'flax.nnx.Linear'}<br/>}"):::codeBlock
    %% Transformation logic
        REWRITER("<b>🔄 Pivot Rewriter</b><br/>Maps 'torch' to Abstract Spec<br/>Lowers to 'jax'"):::engine
        PLUGIN("<b>🔌 Structural Plugins</b><br/>nn.Module → nnx.Module<br/>IO & Device Abstraction"):::plugin
        FIXER("<b>🧹 Import Fixer</b><br/>Prune 'torch'<br/>Inject 'jax.numpy'"):::engine
    %% Connections
        PARSER --> ANALYSIS
        ANALYSIS --> REWRITER
        KB_HEADER --- KB_DATA
        KB_DATA -.->|" Map API "| REWRITER
        REWRITER --> PLUGIN
        PLUGIN --> FIXER
    end

%% =================================================================================
%% 3. OUTPUT LAYER
%% =================================================================================
    FIXER -->|" Emit Code "| TGT_FILE

    subgraph Outputs [" Phase 3: Outputs "]
        direction TB
        style Outputs fill: #ffffff, stroke: #20344b, stroke-width: 0px
        TGT_FILE("<b>🚀 Output Target (JAX/Flax)</b>"):::target
        TGT_CODE("from flax import nnx<br/>import jax.numpy as jnp<br/><br/>class Model(<b>nnx.Module</b>):<br/>&nbsp;&nbsp;def <b>__call__</b>(self, x):<br/>&nbsp;&nbsp;&nbsp;&nbsp;return <b>jnp.abs</b>(x)"):::codeBlock
        TGT_FILE --- TGT_CODE
    end
```

---

## ⚡ Key Features

### 1. The Semantic Pivot (AST Engine)

The core engine does not map `torch` directly to `jax`. It maps `torch` to an **Abstract Operation** (e.g., "Math.Abs", "Neural.Conv2d"), then lowers that abstract operation to the target framework.

*   **Renaming:** `torch.sum(input, dim)` → `jax.numpy.sum(a, axis)`.
*   **Unwrapping:** Converts Functional calls (`layer.apply(vars, x)`) to OOP styles or vice-versa.
*   **Infix/Prefix rewriting:** Transforms function calls like `torch.add(a, b)` into operators like `a + b` or `torch.neg(x)` into `-x`.
*   **Context Managers:** Rewrites global states like `torch.no_grad()` into JAX-compatible shims (`nullcontext`) or functional transformations.

### 2. Structural & State Rewriting

Deep learning isn't just about math operations; it's about state and type management.

*   **Class Transpilation:** Converts `torch.nn.Module` ↔ `flax.nnx.Module` ↔ `keras.Layer`, handling `super().__init__` injection and method renaming (`forward` ↔ `__call__`).
*   **RNG Threading:** Detects stateful Torch randomness (`dropout`) and injects explicit PRNG keys (`rng`) into signatures and function bodies for JAX compliance.
*   **Type Hint Rectification:** Parses and rewrites type annotations (e.g., `x: torch.Tensor` → `x: jax.Array`).
*   **IO & Devices:** Plugins automatically map serialization (`torch.save` → `orbax.checkpoint`) and device placement (`torch.device('cuda')` → `jax.devices('gpu')[0]`).

### 3. Safety & Analysis

*   **Purity Scanning:** Static analysis detects side effects unsafe for JAX JIT (I/O, Global mutation, List appends).
*   **Lifecycle Analysis:** specifically detects if class attributes are defined dynamically in `forward` rather than `__init__`, ensuring valid static graph compilation.
*   **Dependency Scanning:** Flags 3rd-party imports (`pandas`, `cv2`) not covered by the translation map.
*   **Smart Imports:** The `ImportFixer` intelligently injects imports (`import jax.numpy as jnp`) *only* if the translated code actually uses them.
*   **The Escape Hatch:** Code that cannot be safely translated is passed through verbatim, wrapped in error markers (`# <SWITCHEROO_FAILED_TO_TRANS>`) for easy manual review.

### 4. Discovery & Verification

*   **Automated Discovery:** The `Scaffolder` utilizes **Griffe** to inspect installed libraries and align them against Standards (ONNX/Array API) using fuzzy matching and signature analysis.
*   **Semantic Harvester:** "Human-in-the-Loop" learning. It scans your manually fixed test files to reverse-engineer valid mappings and update the Knowledge Base automatically.
*   **Fuzzing Engine:** Includes `InputFuzzer` (with symbolic shape constraints) and `HarnessGenerator`. It validates translations by running random inputs through both Source and Target code in isolated subprocesses.

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

Convert a file or directory.

```bash
# Basic conversion (PyTorch -> JAX)
ml_switcheroo convert ./models/resnet.py --out ./models_jax/ --source torch --target jax

# Strict Mode: Fails/Marks unknown APIs instead of passing them through
ml_switcheroo convert ./src --strict --out ./out

# Verify immediately after converting (generates a harness)
ml_switcheroo convert ./math_lib.py --out ./out_lib.py --verify
```

### Knowledge Base Management

Build mappings without writing JSON manually.

```bash
# 1. View current support matrix
ml_switcheroo matrix

# 2. Interactive Wizard: Categorize missing APIs in a package
ml_switcheroo wizard torch

# 3. Harvest rules from a manual test file (Learn from humans)
ml_switcheroo harvest tests/examples/test_custom_layer.py --target jax
```

### Verification (CI)

```bash
# Run validation suite on all known mappings
ml_switcheroo ci

# Generate a standalone lockfile of valid operations
ml_switcheroo ci --json-report verified_ops.json
```

---

## 🏗 Architecture

The system is split into **Core Logic** (the engine) and **Semantics** (the data).

1.  **Semantics (`src/ml_switcheroo/semantics`):** JSON files defining standards.
    *   `k_array_api.json`: Math operations (abs, sum, matmul).
    *   `k_neural_net.json`: Stateful layers (Linear, Conv2d).
    *   `k_framework_extras.json`: Utilities (DataLoader, seeds, devices).
2.  **Importers:** Ingests upstream specs (ONNX Markdown, Array API Stubs).
3.  **Core:** `PivotRewriter` visits the AST, looks up the Semantics, applies `Plugins` (Hook system), and emits code.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flow.

---

## ✅ Compatibility Matrix

Supported Frameworks via Adapters & Scanners:

| Framework      | Status                   | Role                  |
|:---------------|:-------------------------|:----------------------|
| **PyTorch**    | 🟢 Primary               | Source / Target       |
| **JAX / Flax** | 🟢 Primary               | Source / Target (NNX) |
| **NumPy**      | 🟡 Supported             | Fallback / Inputs     |
| **TensorFlow** | 🔵 Experimental          | Adapter / Spec Sync   |
| **Apple MLX**  | 🔵 Experimental          | Adapter / Spec Sync   |

---

## Contributing

1.  Check definitions in `src/ml_switcheroo/semantics/*.json`.
2.  Missing an OP? Run `ml_switcheroo wizard torch` to add it interactively.
3.  Need complex logic? Add a hook in `src/ml_switcheroo/plugins/`.

## License

[Apache-2.0 License](LICENSE)
