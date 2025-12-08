ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.**

[![License: MIT](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/)
[![Tests](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/tests_release.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/tests_release.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**ml-switcheroo** converts Deep Learning models between frameworks (primarily **PyTorch** ↔ **JAX/Flax**) using strict
Abstract Syntax Tree (AST) manipulation.

Unlike LLM-based coding assistants which "guess" output, ml-switcheroo uses a **Knowledge Base** derived from official
specifications (ONNX, Python Array API) to perform mathematically guaranteed translations. If a translation is
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
        ANALYSIS("<b>🔍 Analysis</b><br/>Purity Check & Dependency Scan"):::engine
    %% Semantic Lookup
        KB_HEADER("<b>📚 Knowledge Base</b>"):::semantics
        KB_DATA("{<br/>&nbsp;'abs': {'jax': 'jax.numpy.abs'},<br/>&nbsp;'Linear': {'jax': 'flax.nnx.Linear'}<br/>}"):::codeBlock
    %% Transformation logic
        REWRITER("<b>🔄 Pivot Rewriter</b><br/>Maps 'torch' to Abstract Spec<br/>Lowers to 'jax'"):::engine
        PLUGIN("<b>🔌 Structural Plugins</b><br/>nn.Module → nnx.Module<br/>forward → __call__"):::plugin
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

The core engine does not map `torch` directly to `jax`. It maps `torch` to an **Abstract Operation** (e.g., "
Math.Abs", "Neural.Linear"), then lowers that abstract operation to the target framework.

* **Renaming:** `torch.sum(input, dim)` → `jax.numpy.sum(a, axis)`.
* **Unwrapping:** Converts Functional calls (`layer.apply(vars, x)`) to OOP styles or vice-versa.
* **Infix/Prefix rewriting:** Transforms function calls like `torch.add(a, b)` into operators like `a + b`.

### 2. Structural & State Rewriting

Deep learning isn't just about math operations; it's about state management.

* **Class Transpilation:** Converts `torch.nn.Module` → `flax.nnx.Module`.
* **RNG Threading:** Detects stateful Torch randomness (`dropout`) and injects explicit PRNG keys (`rng`) into
  signatures and function bodies for JAX compliance.
* **Lifecycle Management:** Strips framework-specific artifacts (e.g., `x.to(device)`, `x.detach()`) while preserving
  logic.
* **Context Managers:** Rewrites global states like `torch.no_grad()` into JAX-compatible shims/no-ops.

### 3. Safety & Analysis

* **Purity Scanning:** Static analysis detects side effects unsafe for JAX JIT (Input/Output, Global mutation, List
  appends).
* **Dependency Scanning:** Flags 3rd-party imports (`pandas`, `cv2`) not covered by the translation map.
* **Smart Imports:** The `ImportFixer` intelligently injects imports (`import jax.numpy as jnp`) *only* if the
  translated code actually uses them.
* **The Escape Hatch:** Code that cannot be safely translated is passed through verbatim, wrapped in error markers (
  `# <SWITCHEROO_FAILED_TO_TRANS>`) for easy manual review.

### 4. Discovery & verification

* **Fuzzing Engine:** Includes `InputFuzzer` and `HarnessGenerator` to validate translations by running random inputs
  through both Source and Target code in isolated subprocesses.
* **Automated Discovery:** The `Scaffolder` utilizes **Griffe** to inspect installed libraries and align them against
  Standards (ONNX/Array API) using fuzzy matching and signature analysis.
* **Semantic Harvester:** "Human-in-the-Loop" learning. It scans your manually fixed test files to reverse-engineer
  valid mappings and update the Knowledge Base automatically.

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
# Basic conversion
ml_switcheroo convert ./models/resnet.py --out ./models_jax/ --source torch --target jax

# Strict Mode: Fails/Marks unknown APIs instead of passing them through
ml_switcheroo convert ./src --strict --out ./out

# Verify immediately after converting (generates a harness)
ml_switcheroo convert ./math_lib.py --out ./out_lib.py --verify
```

### Knowledge Base Management

Build mappings without writing JSON manually.

```bash
# 1. View current support
ml_switcheroo matrix

# 2. Interactive Wizard: Categorize missing APIs in a package
ml_switcheroo wizard torch

# 3. Harvest rules from a manual test file
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

1. **Semantics (`src/ml_switcheroo/semantics`):** JSON files defining standards.
    * `k_array_api.json`: Math operations (abs, sum, matmul).
    * `k_neural_net.json`: Stateful layers (Linear, Conv2d).
    * `k_framework_extras.json`: Utilities (DataLoader, seeds).
2. **Importers:** Ingests upstream specs (ONNX Markdown, Array API Stubs).
3. **Core:** `PivotRewriter` visits the AST, looks up the Semantics, applies `Plugins` (Hook system), and emits code.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flow.

---

## ✅ Compatibility Matrix

Supported Frameworks via Adapters:

| Framework      | Status                   | Role                  |
|:---------------|:-------------------------|:----------------------|
| **PyTorch**    | 🟢 Primary               | Source / Target       |
| **JAX / Flax** | 🟢 Primary               | Source / Target (NNX) |
| **NumPy**      | 🟡 Barely more than idea | Fallback / Inputs     |
| **TensorFlow** | 🔵 Idea only             | Adapter / Spec Sync   |
| **Apple MLX**  | 🔵 Idea only             | Adapter / Spec Sync   |

---

## Contributing

1. Check definitions in `src/ml_switcheroo/semantics/*.json`.
2. Missing an OP? Run `ml_switcheroo wizard torch` to add it interactively.
3. Need complex logic? Add a hook in `src/ml_switcheroo/plugins/`.

## License

[Apache-2.0 License](LICENSE)
