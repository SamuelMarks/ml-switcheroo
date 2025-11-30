Architecture
============

ml-switcheroo is a deterministic, specification-driven transpiler. It solves the $O(N^2)$ framework translation problem by
decoupling **specification** (the API standard) from **implementation** (the framework mapping).

The system relies on a central Knowledge Base consisting of Semantic Specifications (derived from the Python Array API
and ONNX).

## 🏗️ System Data Flow

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef input fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef build fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef store fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans Medium',rx:5px;
    classDef verify fill:#20344b,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef convert fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans Medium',rx:5px;
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono Normal',rx:5px;

    %% --- 1. INGESTION ---
    subgraph S1 [Phase 1: Ingestion & Discovery]
        direction TB
        Standards(External Specs):::input
        Libs(Installed Frameworks):::input
        Builder(Inspector & Scaffolder):::build
        Harvester(Semantic Harvester):::build
        
        Standards --> Builder
        Libs --> Builder
        Libs --> Harvester
    end

    %% --- 2. STORAGE ---
    subgraph S2 [Phase 2: Knowledge Base]
        direction TB
        JSON[("semantics/*.json<br/>(The Pivot)")] 
    end
    class JSON store
    
    Builder --> |Populate| JSON
    Harvester --> |Update| JSON

    %% --- 3. VERIFICATION ---
    subgraph S3 [Phase 3: Verification]
        direction TB
        TestGen(Test Generator):::verify
        Suite(Physical Test Suite):::artifact
        Fuzzer(Input Fuzzer + Adapters):::verify
        
        JSON --> |Read Spec| TestGen
        TestGen --> |Create| Suite
        Suite --> |Execute| Fuzzer
        Fuzzer --> |Backend Adapters| Libs
    end

    %% --- 4. CONVERSION ---
    subgraph S4 [Phase 4: Conversion Pipeline]
        direction TB
        UserCode(User Source Code):::artifact
        Analyzer(Purity & Deps Analysis):::convert
        Rewriter(Pivot Rewriter & Hooks):::convert
        Fixer(Import Fixer):::convert
        Output(Target Code):::artifact
        
        UserCode --> Analyzer
        Analyzer --> |AST| Rewriter
        Rewriter --> |AST| Fixer
        Fixer --> Output
    end

    %% Cross-Phase Links
    JSON --> |Lookup verified API| Rewriter
    Fuzzer -.-> |Validates| Rewriter
```

---

## 🔁 Operational Workflows

### Workflow A: Automated Specification Scaffolding (Robot)

**Goal:** Auto-discover mappings between frameworks and the standard.

1. **Ingest:** `ApiInspector` consumes Array API / ONNX specs.
2. **Align:** `Scaffolder` introspects installed libraries (Torch/JAX) using heuristic matching (fuzzy name + signature
   arity check).
3. **Result:** Populated JSONs in `src/ml_switcheroo/semantics/`.

### Workflow B: The Hybrid Verification Loop (Human-in-the-Loop)

**Goal:** Verify framework equivalence via generated tests.

1. **Generate:** `TestGenerator` creates `tests/generated/test_math.py`.
2. **Execute:** `pytest` runs the suite. The `InputFuzzer` generates random inputs using type hints found in the specs.
3. **Failures:** If a generated test fails, a human engineer writes a manual override function in the same file. The
   system respects manual tests over generated ones.

### Workflow C: Production Transpilation Pipeline

**Goal:** Convert source code to target framework.

1. **Analysis:** The `ASTEngine` runs:
    * **PurityScanner:** Flags JAX-unsafe side effects (print, global, mutation).
    * **DependencyScanner:** Flags unknown 3rd-party libs.
2. **Rewrite:** `PivotRewriter` walks the AST.
    * Transforms classes (Structure rewriting for NNX).
    * Injects required state arguments (e.g. `rngs`, `variables`).
    * Applies Plugins (Decompositions, Context Managers).
3. **Refine:** `ImportFixer` prunes old imports and intelligently injects new ones only if used.

### Workflow E: Semantic Harvesting

**Goal:** Learn from human-written tests.

1. **Scan:** The `SemanticHarvester` reads manual test files (e.g., `tests/examples/`).
2. **Extract:** It finds passing calls to target APIs (e.g. `jax.numpy.sum(a=x, axis=y)`).
3. **Learn:** It reverse-engineers the mapping (`x->a`, `axis->axis`) and updates the Knowledge Base JSONs
   automatically.
