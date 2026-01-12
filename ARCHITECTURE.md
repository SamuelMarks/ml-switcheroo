Architecture
============

**ml-switcheroo** is a deterministic, specification-driven transpiler designed to convert Deep Learning code between
frameworks (e.g., PyTorch to JAX/Flax, Keras 3 to TensorFlow) with mathematical rigor. 

It solves the $O(N^2)$ translation problem by decoupling **Specification** (the Abstract Operation) from
**Implementation** (the Framework API) using a **Hub-and-Spoke** architecture. Rather than writing translators for every
pair of frameworks, we map every framework to a central "Abstract Standard." 

--- 

## 🏗️ The Semantic Pivot Strategy

The conversion process is a three-step movement through an abstract intermediate state: 

1. **Ingest (Source $\to$ Hub):** The system identifies a framework call (e.g., `torch.permute`) and maps it to an * 
   *Abstract Operation** (e.g., `permute_dims`) using the source framework's snapshot or adapter definition. It also
   normalizes input formats (assignments, attributes) into a unified AST. 
2. **Pivot (Normalization):** Arguments are reordered, renamed, unpacked, and validated to match the Abstract Standard ( 
   The "Hub" signature). 
3. **Project (Hub $\to$ Target):** The system looks up the implementation for the target framework (e.g., 
   `jax.numpy.transpose`) and generates the corresponding AST, applying any necessary DSL logic (Layout Permutation, 
   Macros) or Plugin hooks. 

--- 

## 🧩 1. The Knowledge Base (Hub & Spoke) 

The core dataset driving the transpiler is distributed across two layers. This separation allows the "What" (Standard) 
to evolve independently of the "How" (Implementation). 

```mermaid
graph TD
    %% --- STYLE DEFINITIONS --- 
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b;
    classDef input fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef build fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef spoke fill:#ffd427,stroke:#f9ab00,stroke-width:2px,stroke-dasharray:5 5,color:#20344b,font-family:'Google Sans Medium',rx:5px;

    %% --- PHASE 1: DISCOVERY --- 
    subgraph P1 [1. Ingestion Phase]
        direction TB
        STANDARDS("External Specs<br/>(ONNX / Array API)"):::input
        CODE("Adapter Classes<br/>(frameworks/*.py)"):::input
        
        LOADER("Registry & File<br/>Loaders"):::build
        
        STANDARDS --> LOADER
        CODE --> LOADER
    end

    %% --- PHASE 2: STORAGE --- 
    subgraph P2 [2. Semantics Manager]
        direction TB
        HUB[("<b>The Hub (Specs)</b><br/>semantics/*.json<br/><i>Abstract Operations</i>")]:::hub
        SPOKE[("<b>The Spokes (Variants)</b><br/>snapshots/*.json<br/><i>Framework Implementations</i>")]:::spoke
        
        %% Internal Context Link
        SPOKE -.->|"Hydrates"| HUB
    end

    %% Flow P1 -> P2
    LOADER -->|"Populate"| HUB
    LOADER -->|"Populate"| SPOKE

    %% --- PHASE 3: VERIFICATION --- 
    subgraph P3 [3. Verification Phase]
        direction TB
        TESTER("TestGen & Fuzzer"):::build
    end

    %% Flow P2 -> P3
    HUB -.->|"Read Constraints"| TESTER
    SPOKE -.->|"Read Variants"| TESTER
```

### The Hub: Semantic Specifications

Defines **WHAT** an operation is. Populated from `src/ml_switcheroo/semantics/*.json` and internal defaults. 

* **Tier A (Math):** `k_array_api.json` — Array API Standard (NumPy-like). 
* **Tier B (Neural):** `k_neural_net.json` — ONNX Operators (Layers, Activations). 
* **Tier C (Extras):** `k_framework_extras.json` — Framework utilities, IO, and internals. 

### The Spokes: Framework Overlays

Defines **HOW** a specific framework implements the standard. Populated from `src/ml_switcheroo/frameworks/*.py` (Live) 
or `src/ml_switcheroo/snapshots/` (Ghost). 

* **API Path:** E.g., `torch.abs`, `jax.numpy.abs`. 
* **Argument Map:** E.g., `{"input": "x", "dim": "axis"}`. 
* **DSL Config:** Layout maps, Macros, Type Casting rules. 

This supports **Ghost Mode**, allowing the engine to transpile code even if source/target libraries are not installed
locally. 

--- 

## ⚡ 2. The Transpilation Engine

The `ASTEngine` orchestrates the conversion pipeline. It handles parsing, deep analysis, optimization, and
transformation. 

```mermaid
graph TD
    %% --- STYLE DEFINITIONS --- 
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b;
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono Normal',stroke-dasharray: 0;
    classDef process fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef optimization fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef kb fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef plugin fill:#57caff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef output fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;

    %% --- NODES --- 
    SRC("Source Code"):::artifact
    
    subgraph ENGINE [AST Engine]
        direction TB
        INGEST("1. Ingestion<br/>(Python/MLIR/TikZ)"):::process
        
        subgraph ANALYSIS_BLOCK [Analysis Phase]
            direction TB
            SYMBOLS("Symbol Table"):::process
            PURITY("Purity & Safety"):::process
        end

        SERVER[("Semantics<br/>Manager")]:::kb
        
        GRAPH_OPT("2. Graph Optimizer<br/>(Fusion)"):::optimization
        
        REWRITER("3. Rewriter Pipeline<br/>(Structure/API/Aux Passes)"):::process
        PLUGINS{{Plugin & DSL System}}:::plugin
        
        FIXER("4. Refinement<br/>(Import Fixer)"):::process
    end
    
    TGT("Target Code"):::output

    %% --- EDGES --- 
    SRC --> INGEST
    INGEST --> SYMBOLS
    SYMBOLS --> PURITY
    PURITY --> GRAPH_OPT
    
    GRAPH_OPT --> REWRITER
    SERVER -.->|"Lookup API"| REWRITER
    REWRITER <-->|"Complex Logic"| PLUGINS
    
    REWRITER --> FIXER
    FIXER --> TGT
```

### 1. Ingestion & Analysis

Before touching the code, the engine parses and scans for context. 

* **Parsers:** Supports Python (LibCST), MLIR Text, and TikZ/Latex sources via Adapters. 
* **SymbolTableAnalyzer:** Infers variable types (e.g., "is `x` a Tensor or a Module?") and scopes to drive intelligent
  rewriting. 
* **PurityScanner:** Detects side effects (IO, Globals, in-place mutation) unsafe for JAX/XLA targets (Enabled via
  `PluginTraits`). 
* **LifecycleTracker:** Verification pass ensuring all class attributes used in `forward` are initialized in `__init__`. 

### 2. Optimization Phase (`GraphOptimizer`) 

An optional pass that builds a `LogicalGraph` from the AST and applies fusion patterns. 

* Matches sequences like `Conv2d -> BatchNorm -> ReLU`. 
* Replaces them with fused macro operations (e.g., `FusedCBR`) based on `PatternDef` in Semantics. 

### 3. Rewriting Phase (`RewriterPipeline`) 

The core transformer is built on a **Pipeline** architecture. The `RewriterPipeline` orchestrates sequential **Passes** which share a mutable `RewriterContext`.

* **StructuralPass:** Handles Class/Function definitions (swaps Base classes, renames `forward`, injects `rngs` args, strips Lifecycle methods). 
* **ApiPass:** The workhorse. Resolves API calls and Attributes to Abstract IDs, applies Layout Permutation, Argument Normalization, and dispatches to Plugins. 
* **AuxiliaryPass:** Handles secondary syntax constructions (Decorators) and ensures safety for Control Flow (e.g., Loop Unrolling checks). 

### 4. Refinement Phase

* **ImportFixer:** An intelligent pass that post-processes the AST. It injects required imports (e.g., 
  `import jax.numpy as jnp`) only if used and prunes unused source imports. 
* **StructuralLinter:** A final sanity verification that flags any residual artifacts from the source framework. 

--- 

## 🔌 3. Framework Adapters (Traits & Hierarchy) 

Support for specific libraries resides in `src/ml_switcheroo/frameworks/`. Adapters provide **Traits** to the engine
rather than hardcoded logic. 

### Structural Traits

Adapters define a `StructuralTraits` configuration object that controls syntax generation: 

* `module_base`: The base class for layers (e.g., `"flax.nnx.Module"`). 
* `forward_method`: The inference method name (`"forward"` vs `"call"` vs `"__call__"`). 
* `inject_magic_args`: Tuple of arguments to inject into signatures (e.g., `[("rngs", "nnx.Rngs")]`). 
* `lifecycle_strip_methods`: Methods to silently remove (e.g., `.cuda()`, `.detach()`). 

### Plugin Traits

Adapters define `PluginTraits` to toggle logic blocks used by generic plugins: 

* `has_numpy_compatible_arrays`: Enables `.astype()` casting and tuple-padding logic (JAX/TF/NumPy/MLX). 
* `requires_explicit_rng`: Enables PRNG key threading logic (JAX/Flax). 
* `requires_functional_state`: Enables BatchNorm state unwrapping logic (JAX). 

--- 

## 🧠 4. DSL & Plugin System

The system favors declarative logic in the ODL (Operation Definition Language) over python code, but falls back to
Python Hooks for complex structural changes. 

### Core DSL Logic (In `PivotRewriter` via Passes) 

Common patterns are handled directly by the engine using the ODL schema: 

* **Variadic Packing:** `pack_to_tuple="axes"` converts `permute(x, 0, 1)` $\to$ `transpose(x, axes=(0, 1))`. 
* **Layout Mapping:** `layout_map={"input": "NCHW->NHWC"}` injects permutation calls via `inject_permute_call`. 
* **Macros:** `macro_template="{x} * sigmoid({x})"` expands composite ops inline. 
* **Dispatch Rules:** `dispatch_rules` switch APIs based on argument values at runtime (e.g. `mode="nearest"` uses a
  different function). 

### Plugin Hooks (In `src/ml_switcheroo/plugins/`) 

Complex architectural mismatches are handled by Python functions registered via `@register_hook`. 

* **RNG Threading (`rng_threading`):** Transforms global seeds to explicit JAX keys. Injects `rng, key = split(rng)`
  preambles. 
* **Context Wrappers (`context_to_function_wrap`):** Converts `torch.no_grad()` to `contextlib.nullcontext()` for JAX. 
* **State Containers (`state_container`):** Converts `register_buffer`/`Parameter` calls into framework-specific object
  wrappers (e.g., `nnx.BatchStat`, `nnx.Param`). 
* **IO Handlers (`io_handler`):** Maps `torch.save` to `orbax.checkpoint` or `tf.io.write_file` depending on the target
  adapter's I/O configuration. 
* **Data Loaders (`convert_dataloader`):** Injects a generic Shim class to replace `torch.utils.data.DataLoader`.
