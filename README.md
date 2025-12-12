ml-switcheroo 🔄🦘
==================

**A Deterministic, Specification-Driven Transpiler for Deep Learning Frameworks.** 

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/license/apache-2-0) 
[![Test and release](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml/badge.svg)](https://github.com/SamuelMarks/ml-switcheroo/actions/workflows/test_and_release.yml) 
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) 
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff) 

**ml-switcheroo** is a rigorous AST-based transpiler designed to convert Deep Learning code between frameworks (PyTorch $\leftrightarrow$ JAX/Flax, TensorFlow, etc.) without hallucination. 

It uses a **Hub-and-Spoke** architecture to solve the $O(N^2)$ translation problem. All frameworks map to an **Abstract Standard** (Hub), allowing "Zero-Edit" support for new frameworks via isolated JSON snapshots (Spokes).

--- 

### 🏗️ Architecture

Code is parsed into an Abstract Syntax Tree (AST), analyzed for safety, pivoted through the Abstract Standard, and reconstructed for the target framework. 

<!-- prettier-ignore -->

```mermaid
graph TD
    %% =================================================================================
    %% THEME CONFIGURATION
    %% Colors: Blue 500 (#4285f4), Green 500 (#34a853), Yellow 600 (#f9ab00), Red 500 (#ea4335) 
    %% Navy (#20344b), White (#ffffff) 
    %% =================================================================================
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b,stroke-width:1px; 
    classDef source fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px; 
    classDef engine fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px; 
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px; 
    classDef spoke fill:#fff4c7,stroke:#f9ab00,stroke-width:2px,stroke-dasharray: 5 5,color:#20344b,font-family:'Google Sans Medium',rx:5px; 
    classDef target fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px; 
    classDef codeBlock fill:#ffffff,stroke:#20344b,stroke-width:1px,font-family:'Roboto Mono Normal',text-align:left,font-size:12px; 

    SRC_HEADER("<b>0. Input Source (PyTorch)</b>"):::source
    PARSER("<b>1. AST Analysis</b><br/>Purity & Lifecycle Scanners"):::engine
    
    SRC_HEADER --> PARSER

    %% --- THE HUB AND SPOKE KNOWLEDGE BASE ---
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
    
    FIXER("<b>3. Import Fixer</b><br/>Injects 'jax.numpy'"):::engine
    PIVOT_LOGIC --> FIXER

    TGT_HEADER("<b>4. Output Target (JAX)</b>"):::target
    FIXER --> TGT_HEADER
```

--- 

## ⚡ Core Capabilities

### 1. The Semantic Pivot
The engine maps source code to **Abstract Operations** (e.g., `Math.Abs`, `Neural.Conv2d`), then projects them to the target framework. 

* **Hub (Specs):** Defined in `semantics/`. Contains the abstract definition (e.g., `Abs(x)`).
* **Spokes (Mappings):** Defined in `snapshots/`. Contains the framework-specific implementation (e.g., `torch` maps `Abs` to `torch.abs`, `jax` maps it to `jax.numpy.abs`).

### 2. Structural & State Rewriting
* **Class Transpilation:** Converts `torch.nn.Module` $\leftrightarrow$ `flax.nnx.Module`. Handles `super().__init__` stripping/injection and method renaming (`forward` $\leftrightarrow$ `__call__`). 
* **RNG Threading:** Detects stochastic operations (`dropout`) and injects explicit PRNG variables (`rng`, `key`) for JAX compliance.
* **Lifecycle Management:** Strips framework-specific idioms (`.to(device)`, `.detach()`) while preserving logic.

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

```bash
# Basic conversion
ml_switcheroo convert ./models/resnet.py --out ./models_jax/ 

# Specify Frameworks manually
ml_switcheroo convert ./src --source torch --target tensorflow --out ./dst
```

### Discovery & Learning

Populate the Knowledge Base automatically. 

```bash
# 1. Scaffold: Scan installed libs and write mappings to snapshots/
ml_switcheroo scaffold --frameworks torch jax

# 2. Wizard: Interactive tool to categorize unmapped APIs
ml_switcheroo wizard torch
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

1. **New Frameworks:** Add a file to `src/ml_switcheroo/frameworks/` and a mapping file to `src/ml_switcheroo/snapshots/`.
2. **New Logic:** Add a hook to `src/ml_switcheroo/plugins/`.

See [EXTENDING.md](EXTENDING.md) for a detailed guide. 

--- 

## License

[Apache-2.0 License](LICENSE) 
