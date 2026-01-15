Internal Architecture & Theoretical Mechanics
=============================================

**Document Version**: 0.0.1

**Scope**: Core Engine, Compiler Mechanics, IR Design, and Knowledge Acquisition.

## 1. Abstract

`ml-switcheroo` is a deterministic, specification-driven source-to-source compiler architecture. Unlike traditional
transpilers that map syntax 1:1, it employs a **Hub-and-Spoke** semantic model to solve the $O(N^2)$ interoperability
problem.

The system treats every deep learning representation—whether high-level Python (PyTorch, JAX), intermediate
representation (StableHLO, MLIR), or hardware assembly (SASS, RDNA)—as a dialect of a central mathematical logic.

The runtime engine uses a **Dual-Pipeline Strategy** to handle the distinct topological requirements of structured
code (ASTs) versus linear instruction streams (Graphs/ASM).

---

## 2. The Grand Unified Architecture

The `ASTEngine` (`src/ml_switcheroo/core/engine.py`) is the central orchestration unit. Upon receiving source code, it
classifies the Input/Output languages to route execution through one of two isomorphic pipelines.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Google Sans', 'fontSize': '14px', 'lineColor': '#20344b'}}}%%
graph TD
    classDef process fill: #4285f4, stroke: #20344b, stroke-width: 1px, color: #ffffff, rx: 4px;
    classDef decision fill: #f9ab00, stroke: #20344b, stroke-width: 1px, color: #20344b, rx: 20px;
    classDef storage fill: #ea4335, stroke: #20344b, stroke-width: 1px, color: #ffffff, rx: 4px;
    classDef artifact fill: #ffffff, stroke: #20344b, stroke-width: 1px, stroke-dasharray: 2 2;
    SRC("Input Code"):::artifact

subgraph ENGINE [AST Engine]
direction TB
ROUTER{"Target Type?"}:::decision

subgraph PATH_A [🟢 Path A: Semantic Rewriter]
direction TB
INGEST("Ingestion (LibCST)"):::process
PIPELINE("Pass Pipeline<br/>(Structure &rarr; API &rarr; Aux)"):::process
FIXER("Refinement<br/>(Import Fixer)"):::process
INGEST --> PIPELINE --> FIXER
end

subgraph PATH_B [🔵 Path B: Graph Compiler]
direction TB
LIFT("Lifter / Parser<br/>(Source &rarr; IR)"):::process
OPT("Graph Optimizer<br/>(Fusion)"):::process
SYNTH("Backend Synthesizer<br/>(IR &rarr; Target)"):::process
LIFT --> OPT --> SYNTH
end
end

KB[("<b>Semantics Manager</b><br/>(The Hub)")]:::storage

TGT("Output Artifact"):::artifact

SRC --> ROUTER
ROUTER -->|" High-Level<br/>(Python/MLIR) "|PATH_A
ROUTER -->|" Low-Level / Visual<br/>(ASM/TikZ/HTML) "|PATH_B

PIPELINE <--> KB
SYNTH <--> KB

FIXER --> TGT
SYNTH --> TGT
```

### 2.1. Path A: The High-Fidelity Rewriter

**Used for:** Python $\leftrightarrow$ Python (Torch, JAX, TF, Keras, Flax)

This path treats code as a mutable, structured document. The goal is **preservation**. Comments, whitespace, and
variable naming conventions are retained where possible.

* **Intermediate Representation:** Concrete Syntax Tree (LibCST).
* **Mechanism:** A pipeline of visitor passes modifies the tree in-place, guided by the Semantic Knowledge Base.

### 2.2. Path B: The Graph Compiler

**Used for:** Assembly (SASS, RDNA), Visuals (TikZ, HTML), and IR roundtrips.

This path treats code as a reconstructible logic flow. It "lifts" linear instruction streams into a topological graph,
optimizing and fusing nodes before synthesizing completely fresh output code.

* **Intermediate Representation:** `LogicalGraph` (DAG).
* **Mechanism:** Parsers convert text to Nodes; Backends synthesize target text from the graph topology.

---

## 3. The Knowledge Base (The Hub)

The system intelligence resides in `src/ml_switcheroo/semantics/`. It decouples **Specification** (What) from *
*Implementation** (How).

### 3.1. Distributed Specifications

The knowledge base is not a single file; it is an aggregate view composed of:

1. **JSON Specs (`semantics/`)**: Defines abstract operations (e.g., `LogSoftmax`, `Conv2d`).
2. **Snapshots (`snapshots/`)**: JSON overlays defining how specific frameworks implement those specs.
3. **Registry (`frameworks/`)**: Live Python adapter classes that provide logic traits.

The `SemanticsManager` merges these sources at runtime using a priority system (Neural > Math > Extras).

### 3.2. Lifecycle: Discovery & Consensus

New knowledge is acquired via the **Discovery** package:

1. **Inspection (`ApiInspector`)**: Scans installed libraries or JSON snapshots (`GhostRef`).
2. **Consensus (`ConsensusEngine`)**: Clusters APIs from different frameworks (e.g., grouping `HuberLoss`, `huber_loss`)
   to propose new standards.
3. **Persistence (`SemanticPersister`)**: Writes abstract definitions to the Hub and implementation maps to the Spokes.

---

## 4. Path A: The Rewriter Pipeline

Implemented in `src/ml_switcheroo/core/rewriter/`.

The transformation is orchestrated by a `RewriterPipeline` executing sequential **Passes** over a shared
`RewriterContext`.

### 4.1. Core Passes

1. **StructuralPass**: Handles class inheritance rewriting (e.g., `nn.Module` $\to$ `nnx.Module`), method renaming (
   `forward` $\to$ `__call__`), and signature injection (threading `rng` keys).
2. **ApiPass**: The workhorse. Resolves function calls and attributes.
    * **Dispatch Rules**: Runtime checks on argument values (e.g., if `mode='nearest'`, swap API).
    * **Argument Pivoting**: Renames/reorders arguments to match the Standard.
    * **Strategy Execution**: Applies transforms (Macros, Infix operators, Inline Lambdas).
3. **AuxiliaryPass**: Handles decorators (`@jit`) and control flow safety checks (loop unrolling warnings).

### 4.2. Plugin System & Hooks (`src/ml_switcheroo/plugins`)

Complex architectural mismatches are handled by Python hooks.

* **Registration**: Plugins use `@register_hook("trigger")`.
* **Auto-Wiring**: Plugins can declare their own semantic definitions via the `auto_wire` parameter, removing the need
  for manual JSON editing.
* **HookContext**: Provides plugins access to the Symbol Table and Semantic/Trait configuration of the target framework.

### 4.3. Import Fixer

A post-processing phase (`src/ml_switcheroo/core/import_fixer`). It generates a `ResolutionPlan` based on:

* **Injection**: Adding imports required by the target (e.g., `import jax.numpy as jnp`).
* **Pruning**: Removing unused source imports.
* **Refinement**: Collapsing fully qualified names (`jax.numpy.abs`) into aliases (`jnp.abs`).

---

## 5. Path B: The Graph Compiler

Implemented in `src/ml_switcheroo/compiler/`.

### 5.1. Intermediate Representation (IR)

The `LogicalGraph` is the lingua franca.

* **LogicalNode**: Represents an operation (e.g., `Conv2d`, `Add`). Contains metadata (kernel size, stride).
* **LogicalEdge**: Represents data flow dependencies.

### 5.2. Frontends (Lifters)

* **ASM Lifters (`sass`, `rdna`)**: Parse assembly text. They use `Analyzer` heuristics to reverse-engineer high-level
  semantics (e.g., detecting loop bounds to infer kernel size 3x3).
* **Python Frontend**: Extracts a graph from Python ASTs using provenance tracking to link Logic back to Source Lines.

### 5.3. Backends (Synthesizers)

* **Python/Code Backends**: Reconstruct class definitions and forward passes.
* **Assembler Backends**: Use `RegisterAllocator` to map symbolic variables to physical registers (`R0`, `v0`) and
  expand macros (e.g., generating nested loops for `Conv2d`).
* **Visual Backends (`tikz`, `html`)**: Perform topological sorting to calculate rank-based layouts for diagrams.

### 5.4. Graph Optimization

The `GraphOptimizer` applies fusion patterns to the IR (e.g., `Conv -> BN -> ReLU` becomes `FusedCBR`). This is used for
generating optimized kernels or simplified visualizations.

---

## 6. Verification & Fuzzing

Fidelity is ensured via `src/ml_switcheroo/testing/`.

### 6.1. The Harness Generator

Generates standalone Python scripts that verify `source(x) == target(x)`.

* **Code Extraction**: Uses `inspect` to serialize the `InputFuzzer` logic into the generated script, solving the "
  split-brain" dependency problem.
* **Runtime Injection**: Injects a `runtime.py` module containing cross-framework comparison logic (`verify_results`)
  and determinism fixtures.

### 6.2. Input Fuzzer

Uses `Hypothesis` strategies derived from ODL type hints.

* **Symbolic Shapes**: Resolves constraints like `Array['B', 'N']` to ensure consistent tensor dimensions across
  arguments.
* **Rich Constraints**: Respects `min`, `max`, `dtype`, and `options` from the Spec.

---

## 7. Extensions: MLIR & StableHLO

`ml-switcheroo` bridges Python and Textual IRs.

* **MLIR Bridge**: Parsing MLIR text preserves comments and whitespace (Trivia).
* **StableHLO**: Handled as a dialect. The `StableHloEmitter` maps Python ASTs to `stablehlo.*` ops, allowing Python
  code to be compiled to XLA IR.

---

## 8. Glossary of Components

| Component            | Responsibility                                                   |
|:---------------------|:-----------------------------------------------------------------|
| **ASTEngine**        | Orchestrator. Routes code to Rewriter or Compiler pipeline.      |
| **SemanticsManager** | Database. Loads JSON specs/snapshots and manages lookup indexes. |
| **PivotRewriter**    | Legacy shim wrapping the Rewriter Pipeline for testing.          |
| **ApiPass**          | Transformer. Swaps function calls based on Hub definitions.      |
| **ImportFixer**      | Refiner. Manages `import` statements and aliasing.               |
| **Lifter**           | Frontend. Converts Assembly/Text to `LogicalGraph`.              |
| **Synthesizer**      | Backend. Converts `LogicalGraph` to target code.                 |
| **ConsensusEngine**  | Discovery. Finds common denominators in diverse APIs.            |
| **HarnessGenerator** | Verification. Creates physical validation scripts.               |
