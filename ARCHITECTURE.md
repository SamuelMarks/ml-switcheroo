Architecture
============

ml-switcheroo is a deterministic, specification-driven transpiler designed to convert Deep Learning code between frameworks (e.g., PyTorch to JAX/Flax).

Unlike LLM-based assistants that statistically predict code, ml-switcheroo performs **Semantic Translation** based on a rigorous Knowledge Base. It solves the $O(N^2)$ translation problem (mapping every framework to every other) by decoupling **Specification** (the Abstract Operation) from **Implementation** (the Framework API).

---

## 🏗️ The Semantic Pivot Strategy

The core architectural principle is the **Hub-and-Spoke** model.

1.  **Ingest (Source → Abstract):** The system identifies a framework call (e.g., `torch.sum(input, dim)`) and maps it to an **Abstract Operation Standard** (e.g., `Sum(x, axis)`).
2.  **Pivot (Normalization):** Arguments are reordered and renamed to match the Abstract Standard.
3.  **Project (Abstract → Target):** The system looks up the equivalent implementation for the target framework (e.g., `jax.numpy.sum(a, axis)`) and generates the corresponding AST.

---

## 🧩 1. The Ecosystem (Ingestion & Storage)

Before translation can occur, the system must learn how frameworks map to standards. This is the **Discovery Ecosystem**.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b;
    classDef input fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef build fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef store fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:0px;
    classDef verify fill:#20344b,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;

    %% --- PHASE 1: DISCOVERY ---
    subgraph P1 [1. Ingestion Phase]
        direction TB
        STANDARDS("External Specs<br/>(ONNX / Array API)"):::input
        LIBS("Installed Libs<br/>(Torch / JAX)"):::input
        
        INSPECTOR("Inspector &<br/>Scaffolder"):::build
        
        STANDARDS --> INSPECTOR
        LIBS --> INSPECTOR
    end

    %% --- PHASE 2: STORAGE ---
    subgraph P2 [2. Storage Phase]
        direction TB
        JSON[("Knowledge Base<br/>semantics/*.json")]:::store
    end
    
    INSPECTOR -->|"Populate"| JSON

    %% --- PHASE 3: VERIFICATION ---
    subgraph P3 [3. Verification Phase]
        direction TB
        TESTER("TestGen & Fuzzer"):::verify
        HARVEST("Semantic Harvester"):::verify
        
        JSON -.->|"Read Spec"| TESTER
        TESTER --"Manual Fixes"--> HARVEST
        HARVEST -->|"Update"| JSON
    end
```

### Components

*   **The Knowledge Base (Semantics Layer):**
    Managed by the `SemanticsManager`, this layer aggregates API definitions into "Tiers":
    *   **Tier A (Math):** Pure functional operations (e.g., `abs`, `matmul`). Defined in `k_array_api.json`.
    *   **Tier B (Neural):** Stateful layers (e.g., `Linear`). Defined in `k_neural_net.json`.
    *   **Tier C (Extras):** Utilities and IO. Defined in `k_framework_extras.json`.

*   **Discovery Tools:**
    *   **ApiInspector:** Uses `griffe` for static analysis of installed libraries.
    *   **Scaffolder:** Use heuristics to align discovered APIs against upstream Specs.
    *   **Semantic Harvester:** A "Human-in-the-Loop" tool that learns from manually written tests to update the Knowledge Base.

---

## ⚡ 2. The Transpilation Engine (Conversion)

Once the Knowledge Base is populated, the `ASTEngine` orchestrates the conversion of user code using `LibCST`.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b;
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono Normal',stroke-dasharray: 0;
    classDef process fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef store fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:0px;
    classDef plugin fill:#57caff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef output fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;

    %% --- NODES ---
    SRC("Source Code"):::artifact
    
    subgraph ENGINE [AST Engine]
        direction TB
        ANALYSIS("1. Safety Analysis<br/>(Purity/Deps Check)"):::process
        
        JSON[("Knowledge<br/>Base")]:::store
        
        REWRITER("2. Pivot Rewriter"):::process
        PLUGINS{{Plugin System}}:::plugin
        
        FIXER("3. Refinement<br/>(Import Fixer)"):::process
    end
    
    TGT("Target Code"):::output

    %% --- EDGES ---
    SRC --> ANALYSIS
    ANALYSIS --> REWRITER
    
    JSON -.->|"Lookup API"| REWRITER
    REWRITER <-->|"Complex Logic"| PLUGINS
    
    REWRITER --> FIXER
    FIXER --> TGT
```

### Pipeline Steps

1.  **Analysis Phase:**
    *   **PurityScanner:** Detects side effects unsafe for functional frameworks (global mutations, IO).
    *   **InitializationTracker:** Ensures class attributes used in `forward` are properly defined in `__init__`.
    *   **DependencyScanner:** Flags 3rd-party imports (e.g., `cv2`) not covered by the semantic map.

2.  **Rewriting Phase (`PivotRewriter`):**
    *   **StructureMixin:** Handles Class inheritance warping (`nn.Module` $\leftrightarrow$ `nnx.Module`), method renaming (`forward` $\leftrightarrow$ `__call__`), and state injection (`rngs`).
    *   **CallMixin:** Handles function calls, argument pivoting, and lifecycle stripping (removing `.cpu()`, `.detach()`).
    *   **Plugin System:** Hooks into specific abstract operations to perform complicated transforms (e.g., `context_to_function_wrap` for `no_grad`).

3.  **Refinement Phase:**
    *   **ImportFixer:** Prunes unused source imports and intelligently injects target imports and standard aliases (e.g., `import jax.numpy as jnp`) only if referenced.
    *   **Escape Hatch:** Wraps untranslatable code segments in comments (`# <SWITCHEROO_FAILED_TO_TRANS>`) preserving validity.

---

## 🔌 Plugin & Hook System

The engine is extensible via `src/ml_switcheroo/plugins`. Plugins register hooks targeting specific semantic operations.

*   **Hook Context:** Provides plugins with access to global configuration, target framework traits, and code injection facilities (Preamble/Signature injection).
*   **Common Use Cases:**
    *   **Decomposition:** Turning `torch.add(x, y, alpha=2)` into `x + y * 2`.
    *   **State Injection:** Threading `rng` keys for JAX stochasticity.
    *   **IO Handling:** Delegating serialization (`torch.save`) to framework-specific adapters (`orbax.checkpoint`).

---

## 🔁 Operational Workflows

### Workflow A: Automated Scaffolding (Robot)
**Goal:** Auto-discover mappings between frameworks and the standard.
1. `ApiInspector` catalogues installed libraries.
2. `Scaffolder` fuzzy-matches signatures against the Spec Store.
3. Result: Populated `semantics/*.json` files.

### Workflow B: The Hybrid Verification Loop
**Goal:** Verify framework equivalence.
1. `TestGenerator` creates a physical test suite from templates (`k_test_templates.json`).
2. `pytest` executes the suite. The `InputFuzzer` generates data based on `std_args` types found in the specs.
3. Failures are fixed manually by humans.
4. `SemanticHarvester` scans the manual fixes to update the Knowledge Base.

### Workflow C: Production Transpilation
**Goal:** Convert user source code.
1. User invokes CLI (`ml_switcheroo convert`).
2. `ASTEngine` parses code and runs Analysis passes.
3. `PivotRewriter` transforms AST based on semantic JSONs and active traits.
4. `ImportFixer` reconciles dependencies.
5. Code is emitted (optionally verified via `HarnessGenerator`).
