Architecture
============

ml-switcheroo is a deterministic, specification-driven transpiler designed to convert Deep Learning code between frameworks (e.g., PyTorch to JAX/Flax). 

It solves the $O(N^2)$ translation problem by decoupling **Specification** (the Abstract Operation) from **Implementation** (the Framework API) using a **Hub-and-Spoke** architecture.

--- 

## 🏗️ The Semantic Pivot Strategy

1.  **Ingest (Source → Hub):** The system identifies a framework call (e.g., `torch.sum`) and maps it to an **Abstract Operation Standard** (e.g., `Sum`) using the framework's "Spoke" mapping.
2.  **Pivot (Normalization):** Arguments are reordered and renamed to match the Abstract Standard (The "Hub"). 
3.  **Project (Hub → Target):** The system looks up the implementation in the target framework's "Spoke" (e.g., `jax.numpy.sum`) and generates the corresponding AST.

--- 

## 🧩 1. The Ecosystem (Ingestion & Storage) 

The system uses a distributed Knowledge Base where standards and implementations are stored separately.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS --- 
    classDef default font-family:'Google Sans',color:#20344b,stroke:#20344b; 
    classDef input fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px; 
    classDef build fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px; 
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px; 
    classDef spoke fill:#fff4c7,stroke:#f9ab00,stroke-width:2px,stroke-dasharray: 5 5,color:#20344b,font-family:'Google Sans',rx:5px; 

    %% --- PHASE 1: DISCOVERY --- 
    subgraph P1 ["1. Ingestion Phase"]
        direction TB
        STANDARDS("External Specs<br/>(ONNX / Array API)"):::input
        LIBS("Installed Libs<br/>(Torch / JAX)"):::input
        
        INSPECTOR("Inspector &<br/>Scaffolder"):::build
        
        STANDARDS --> INSPECTOR
        LIBS --> INSPECTOR
    end

    %% --- PHASE 2: STORAGE --- 
    subgraph P2 ["2. Distributed Storage"]
        direction TB
        HUB[("<b>The Hub (Specs)</b><br/>semantics/*.json<br/><i>Abstract Operations</i>")]:::hub
        SPOKE[("<b>The Spokes (Overlays)</b><br/>snapshots/*_mappings.json<br/><i>Framework Variants</i>")]:::spoke
        
        %% Internal Context Link
        SPOKE -.->|" Hydrates "| HUB
    end

    %% Flow P1 -> P2 (Creates Vertical Spine)
    INSPECTOR -->|"Populate"| HUB
    INSPECTOR -->|"Populate"| SPOKE

    %% --- PHASE 3: VERIFICATION --- 
    subgraph P3 ["3. Verification Phase"]
        direction TB
        TESTER("TestGen & Fuzzer"):::build
    end

    %% Flow P2 -> P3 (Continues Vertical Spine)
    HUB -.->|"Read Spec"| TESTER
    SPOKE -.->|"Read Variant"| TESTER
```

### Components

*   **The Hub (Semantics Layer):** 
    Located in `src/ml_switcheroo/semantics/*.json`. Defines **WHAT** an operation is (Docstring, Standard Arguments).
    *   **Tier A (Math):** `k_array_api.json` from the Python Array API Standard.
    *   **Tier B (Neural):** `k_neural_net.json` from ONNX Operators.
    *   **Tier C (Extras):** `k_framework_extras.json` for IO, Devices, and Helpers.

*   **The Spokes (Snapshot Layer):**
    Located in `src/ml_switcheroo/snapshots/{framework}_mappings.json`. Defines **HOW** a specific framework implements an operation.
    *   **Mapping:** API path (`torch.abs`), argument pivots (`dim` -> `axis`), and plugin hooks.

--- 

## ⚡ 2. The Transpilation Engine (Conversion) 

The `ASTEngine` loads the Knowledge Base by merging the Hub and Spokes at runtime using the `SemanticsManager`.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS --- 
    classDef default font-family:'Google Sans',color:#20344b,stroke:#20344b; 
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono',stroke-dasharray: 0; 
    classDef process fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px; 
    classDef kb fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px; 
    classDef plugin fill:#57caff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px; 
    classDef output fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px; 

    %% --- NODES --- 
    SRC("Source Code"):::artifact
    
    subgraph ENGINE ["AST Engine"]
        direction TB
        ANALYSIS("1. Safety Analysis<br/>(Purity/Deps Check)"):::process
        
        SERVER[("Semantics<br/>Manager")]:::kb
        
        REWRITER("2. Pivot Rewriter"):::process
        PLUGINS{{Plugin System}}:::plugin
        
        FIXER("3. Refinement<br/>(Import Fixer)"):::process
    end
    
    TGT("Target Code"):::output

    %% --- EDGES --- 
    SRC --> ANALYSIS
    ANALYSIS --> REWRITER
    
    SERVER -.->|"Lookup API"| REWRITER
    REWRITER <-->|"Complex Logic"| PLUGINS
    
    REWRITER --> FIXER
    FIXER --> TGT
```

### Pipeline Steps

1.  **Analysis Phase:** 
    *   **PurityScanner:** Detects side effects unsafe for functional frameworks (global mutations, IO). 
    *   **InitializationTracker:** Ensures class attributes used in `forward` are properly defined in `__init__`. 

2.  **Rewriting Phase (`PivotRewriter`):** 
    *   **StructureMixin:** Handles Class inheritance warping (`nn.Module` $\leftrightarrow$ `flax.nnx.Module`) and method renaming (`forward` $\leftrightarrow$ `__call__`). 
    *   **CallMixin:** Handles function calls. It uses the Semantics Manager to find the Abstract ID for a source call, then looks up the target implementation.
    *   **Plugin System:** Hooks into specific operations where simple mapping fails (e.g., `requires_plugin: "decompose_alpha"`). 

3.  **Refinement Phase:** 
    *   **ImportFixer:** Prunes unused source imports and intelligently injects target imports and standard aliases (e.g., `import jax.numpy as jnp`) only if referenced. 

--- 

## 🔌 Plugin & Hook System

The engine is extensible via `src/ml_switcheroo/plugins`. Plugins register hooks targeting specific abstract operations. 

*   **Hook Context:** Provides plugins with access to global configuration, target framework traits, and code injection facilities.
*   **Common Use Cases:** 
    *   **Decomposition:** Turning `torch.add(x, y, alpha=2)` into `x + y * 2`. 
    *   **State Injection:** Threading `rng` keys for JAX stochasticity. 
    *   **IO Handling:** Delegating serialization (`torch.save`) to framework-specific adapters (`orbax.checkpoint`). 
