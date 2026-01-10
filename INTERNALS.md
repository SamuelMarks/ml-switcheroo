Internal Architecture & Theoretical Mechanics
=============================================

**Document Version**: 0.0.1

**Scope**: Core Engine, Compiler Pipeline, Ontology Maintenance, and Extension Architecture

## 1. Abstract

`ml-switcheroo` is a deterministic, specification-driven source-to-source compiler engineered to solve the $O(N^2)$
interoperability challenge in deep learning infrastructure. It creates a semantic bridge between high-level frameworks (
PyTorch, JAX, TensorFlow, Keras, MLX, Flax NNX, PaxML), low-level representations (MLIR, StableHLO, NVIDIA SASS, AMD
RDNA), and visual domain-specific languages (TikZ, HTML).

The system's fundamental axiom is **Bidirectional Isomorphism**: every supported language is treated potentially as both
a *Source* and a *Target*. Whether the input is a Python AST defining a neural network or a stream of RDNA assembly
instructions, the engine lifts the code into a centralized **Abstract Standard (The Hub)** before projecting it into the
desired destination dialect (The Spoke).

This document details the internal subsystems, justifying the architectural bifurcation between the **High-Fidelity
Rewriter** (for structured languages) and the **Graph Compiler** (for linear/unstructured languages), and elucidating
the automated mechanisms of knowledge acquisition that keep the internal ontology synchronized with the evolving ML
ecosystem using LLM-assisted loops and upstream standards ingestion.

---

## 2. The Grand Unified Architecture

The `ASTEngine` serves as the central dispatch controller. It classifies the input and output languages based on their
structural properties and routes them through one of two isomorphic pipelines.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Google Sans', 'fontSize': '14px', 'lineColor': '#20344b'}}}%%
graph TD
    %% --- DESIGN CONSTRAINTS ---
    classDef header font-family:'Google Sans Medium',fill:#20344b,color:#ffffff,stroke:#20344b;
    classDef process fill:#4285f4,stroke:#20344b,stroke-width:1px,color:#ffffff,font-family:'Google Sans',rx:4px;
    classDef decision fill:#f9ab00,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Google Sans Medium',rx:20px;
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono',stroke-dasharray: 2 2;
    classDef storage fill:#ea4335,stroke:#20344b,stroke-width:1px,color:#ffffff,font-family:'Google Sans',rx:4px;
    classDef asm fill:#20344b,stroke:#20344b,stroke-width:1px,color:#ffffff,font-family:'Roboto Mono',rx:2px;

    SRC("Input Artifact<br/>(Source)"):::artifact
    
    subgraph ENGINE [" ⚡ ASTEngine Orchestration "]
        direction TB
        DISPATCH{"Language Topology?"}:::decision
        
        subgraph PATH_A [" 🟢 Path A: Rewriter (CST Preservation) "]
            direction TB
            INGEST_CST("<b>Ingest / Parse</b><br/>(Source &rarr; CST)"):::process
            TRANSFORM("<b>Pivot Transformation</b><br/>(CST &harr; Hub &harr; CST)"):::process
            EMIT_CST("<b>Emission</b><br/>(CST &rarr; Target)"):::process
            
            INGEST_CST --> TRANSFORM --> EMIT_CST
        end

        subgraph PATH_B [" 🔵 Path B: Compiler (Graph Synthesis) "]
            direction TB
            LIFT("<b>Lifter</b><br/>(Source &rarr; IR)"):::process
            IR("<b>LogicalGraph</b><br/>(Topological DAG)"):::artifact
            SYNTH("<b>Synthesizer</b><br/>(IR &rarr; Target)"):::process
            
            LIFT --> IR --> SYNTH
        end
    end

    KB[("<b>Semantics Manager</b><br/>(The Abstract Standard)")]:::storage

    TGT("Output Artifact<br/>(Target)"):::artifact

    %% --- EDGES ---
    SRC --> DISPATCH
    
    DISPATCH -->|" Structured (Python, MLIR, StableHLO) "| PATH_A
    DISPATCH -->|" Unstructured (ASM, TikZ, HTML) "| PATH_B

    TRANSFORM <-->|" Query Specs / Inject Plugins "| KB
    SYNTH <-->|" Query Macros "| KB

    EMIT_CST --> TGT
    SYNTH --> TGT
```

### 2.1. Path A: The High-Fidelity Rewriter

**Ecosystem**: Pure Python (PyTorch, JAX, Flax, TF, Keras, MLX, PaxML) and Textual IRs (MLIR, StableHLO).

This path treats code as a mutable abstract document. The primary objective is **preservation**. When transcoding
high-level code, comments, whitespace, and variable names must be retained.

* **Mechanism**: The `PivotRewriter` walks the tree, identifies semantic nodes (Calls, Classes), maps them to the
  Abstract Hub, and mutates them in-place to match the Target Spoke configuration.
* **Intermediate Representations**: MLIR and StableHLO are treated as textual formats in this path to allow
  high-fidelity refactoring (structure preservation) without lossy graph compilation.

### 2.2. Path B: The Graph Compiler

**Ecosystem**: Hardware ISAs (SASS, RDNA) and Visual DSLs (HTML Grid, TikZ).

This path treats code as a reconstructible logic flow. Assembly languages are linear streams of instructions; they lack
the hierarchical structure (classes, functions) required by Path A.

* **Mechanism**: A `Lifter` parses source text (e.g., SASS Mnemonics) into a `LogicalGraph` Intermediate
  Representation (IR). The `Synthesizer` then reconstructs valid target code (e.g., TikZ diagrams or Python
  Re-implementation) from the topology.

---

## 3. The Knowledge Base (The Hub)

The cornerstone of the system is the **Internal Abstract Standard**, an ontology defining the Platonic ideal of
mathematical operations. It resides in `src/ml_switcheroo/semantics/`.

The Hub defines *what* an operation is (Identity, Signature, Constraints), while the Spokes (Framework Adapters) define
*how* it is implemented.

### 3.1. Construction via The LLM Feedback Loop

The maintenance of the Knowledge Base is semi-automated, utilizing Large Language Models (LLMs) to bridge the gap
between library APIs and the formal Schema.

1. **Introspection (`ml_switcheroo suggest`)**: The system performs runtime reflection on installed libraries.
    * Command: `ml_switcheroo suggest 'jax.numpy.*'` or `ml_switcheroo suggest 'mlx.nn.layers.*'`
    * Action: It scans the namespace, extracts signatures and docstrings, and generates a structured prompt.
2. **LLM Processing**: This prompt is fed to an LLM, which is tasked with mapping the discovered APIs to the Operation
   Definition Language (ODL). It generates a YAML file reconciling the signatures of PyTorch, JAX, TensorFlow, MLX, and
   Flax NNX into a single abstract definition.
3. **Hydration (`ml_switcheroo define`)**: The resulting YAML files are injected into the system.
    * Command: `ml_switcheroo define my_new_ops.yaml`
    * Action: This hydrates the JSON storage (`semantics/*.json` for specifications, `snapshots/*.json` for mappings),
      effectively "teaching" the compiler new math.

### 3.2. Construction via Standards Ingestion

To ground the Abstract Standard in reality, `ml-switcheroo` ingests upstream specifications directly.

* **ONNX (Neural)**: The `OnnxSpecImporter` parses the official `Operators.md` from the ONNX repository, populating
  `k_neural_net.json` with layer definitions common to all frameworks.
* **Array API (Math)**: The `ArrayApiSpecImporter` parses the Python Consortium's standard stubs, populating
  `k_array_api.json` with mathematically rigorous tensor operations.

---

## 4. Path A: The Rewriter Internals

This component executes the bidirectional translation for structured languages using `LibCST`.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Google Sans', 'fontSize': '14px'}}}%%
classDiagram
    direction TD

    class RewriterContext {
        +SemanticsManager semantics
        +SymbolTable symbols
        +List~Scope~ scope_stack
    }

    class RewriterStage {
        +visit()
        +leave()
    }

    class StructureStage {
        +leave_ClassDef()
        +leave_FunctionDef()
    }
    class Note_Struct["<b>Topology Rewriter</b><br/>Inheritance & Signatures"]

    class ApiStage {
        +leave_Call()
        +leave_Attribute()
    }
    class Note_Api["<b>Logic Rewriter</b><br/>Function dispatch"]

    RewriterStage <|-- StructureStage
    RewriterStage <|-- ApiStage
    RewriterStage *-- RewriterContext
%% Styles
    style RewriterContext fill: #4285f4, color: #fff, stroke: #20344b
    style RewriterStage fill: #fff, color: #20344b, stroke: #20344b
    style StructureStage fill: #5cdb6d, color: #20344b, stroke: #20344b
    style ApiStage fill: #57caff, color: #20344b, stroke: #20344b
    style Note_Api fill: #20344b, color: white, stroke: none, font-family: 'Roboto Mono', font-size: 10px
    style Note_Struct fill: #20344b, color: white, stroke: none, font-family: 'Roboto Mono', font-size: 10px
```

### 4.1. Adding New Python Frameworks

Any framework is both a source and a target. Adding support requires implementing the `FrameworkAdapter` protocol in
`src/ml_switcheroo/frameworks/`.

1. **Registry**: Add a file (e.g., `tinygrad.py`) decorated with `@register_framework`.
2. **Structural Traits**: Define base classes (e.g., `tinygrad.Tensor`) and lifecycle methods.
3. **Discovery**: Define `search_modules` to opt-in to the `suggest` loop.
4. **Auto-Wiring**: The system automatically generates the inverse mapping. If `TinyGrad` maps to Hub Operation `Add`,
   the engine inherently knows how to translate `Add` back to `TinyGrad`.

### 4.2. Handling MLIR & StableHLO

These are treated as structured text. The `MlirParser` ingests MLIR into a custom CST. The `PivotRewriter` treats
`stablehlo.abs` identical to `jax.numpy.abs`, allowing seamless refactoring of IR code using the same semantic database
as Python code.

---

## 5. Path B: The Compiler Internals

This path handles linear instruction streams, enabling "Decompilation" (ASM $\to$ Python) and Visualization (
Python $\to$ Diagram).

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Google Sans', 'fontSize': '14px'}}}%%
graph TD
    classDef ir fill:#f9ab00,color:#20344b,stroke:#20344b;
    classDef step fill:#4285f4,color:#ffffff,stroke:#20344b,rx:5px;
    classDef artifact fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono';

    SRC_ASM("Source (Assembly/TikZ)"):::artifact
    
    subgraph LIFTER_BLOCK [" 1. Lifting (Frontends) "]
        REGEX("<b>Lexer</b><br/>(Regex Patterns)"):::step
        LIFT("<b>Lifter</b><br/>(CFG/DFG Analysis)"):::step
        
        REGEX --> LIFT
    end
    
    GRAPH("<b>LogicalGraph (DAG)</b>"):::ir
    
    subgraph SYNTH_BLOCK [" 2. Synthesis (Backends) "]
        MACRO("<b>Macro Expander</b><br/>(e.g. loops &rarr; instructions)"):::step
        EMIT("<b>Emitter</b><br/>(Text Generation)"):::step
        MACRO --> EMIT
    end

    TGT_ASM("Target (Assembly/TikZ)"):::artifact

    SRC_ASM --> LIFTER_BLOCK
    LIFT --> GRAPH
    GRAPH --> SYNTH_BLOCK
    EMIT --> TGT_ASM
```

### 5.1. Adding New ISAs (ASM)

To add a new ISA (e.g., a custom accelerator):

1. **Frontend**: Implement a `Lifter` that parses the textual assembly into `LogicalNodes`. It must reverse-engineer
   loops to infer high-level semantics (e.g., detecting a 3x3 convolution loop).
2. **Backend**: Implement a `Synthesizer` with a `RegisterAllocator` to map symbolic variables to physical registers.

### 5.2. Visual DSLs as Targets & Sources

* **Target (Rendering)**: The `HtmlBackend` calculates topological rank from the `LogicalGraph` to output CSS Grid
  coordinates.
* **Source (OCR/Diagrams)**: The `TikzParser` reads LaTeX diagram code, lifting nodes and edges into the `LogicalGraph`.
  This allows converting a paper's architecture diagram directly into executable PyTorch code.

---

## 6. Verification Loop

The system ensures fidelity through the `HarnessGenerator`.

1. **Constraint Satisfaction**: It reads the ODL constraints (Rank, Dtype) from the JSON Hub.
2. **Fuzzing**: It uses Hypothesis to generate random tensors valid for *both* Source and Target frameworks.
3. **Cross-Check**: It executes both the Source implementation and the Transpiled Target, asserting
   `np.allclose(result_src, result_tgt)`.

This closes the loop: **Introspection (Suggest) $\to$ Definition (Hub) $\to$ Implementation (Spoke) $\to$ Verification (
Fuzzer).**
