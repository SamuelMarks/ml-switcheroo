Maintenance
===========

**ml-switcheroo** is a data-driven transpiler. Its intelligence relies on a distributed **Knowledge Base** separating
*Abstract Specifications* (The Hub) from *Framework Implementations* (The Spokes).

Maintenance primarily involves synchronizing this knowledge base with the ecosystem of Machine Learning libraries and
upstream standards.

This guide covers the full lifecycle: **Ingestion**, **Discovery**, **Mapping**, **Verification**, and **Release**.

--- 

## 🔄 The Maintenance Lifecycle

Data flows from external authoritative sources (Standards Bodies, Library APIs) into our semantic storage tiers, and
finally into verification reports.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef default font-family:'Google Sans Normal',color:#20344b,stroke:#20344b;
    classDef external fill:#ea4335,stroke:#20344b,color:#ffffff,rx:5px,font-family:'Google Sans Medium';
    classDef hub fill:#f9ab00,stroke:#20344b,color:#20344b,rx:5px,font-family:'Google Sans Medium';
    classDef spoke fill:#ffd427,stroke:#f9ab00,stroke-dasharray:5,5,color:#20344b,rx:5px,font-family:'Google Sans Medium';
    classDef action fill:#4285f4,stroke:#20344b,color:#ffffff,rx:5px,font-family:'Google Sans Medium';

    subgraph Sources ["1. Upstream Sources"]
        direction TB
        STD_A("Array API Standard<br/>(Python Consortium)"):::external
        STD_B("ONNX Operators<br/>(Linux Foundation)"):::external
        LIBS("Installed Libraries<br/>(Torch, JAX, TF)"):::external
    end

    subgraph Ingestion ["2. Ingestion & Discovery"]
        IMPORT("import-spec"):::action
        SCAFFOLD("scaffold / sync"):::action
        CONSENSUS("sync-standards<br/>(Consensus Engine)"):::action
        HARVEST("harvest<br/>(Learn from Tests)"):::action
    end

    subgraph Storage ["3. Knowledge Base"]
        direction TB
        HUB[("<b>The Hub (Specs)</b><br/>semantics/*.json<br/><i>Definitions & Types</i>")]:::hub
        SPOKE[("<b>The Spokes (Maps)</b><br/>snapshots/*_mappings.json<br/><i>API Links & Plugins</i>")]:::spoke
    end
    
    subgraph Verify ["4. Verification"]
        CI("CI Fuzzer &<br/>Gen-Tests"):::action
        LOCK("Verified Lockfile<br/>README Matrix"):::spoke
    end

    STD_A --> IMPORT
    STD_B --> IMPORT
    LIBS --> SCAFFOLD
    LIBS --> CONSENSUS

    IMPORT --> HUB
    CONSENSUS --> HUB
    HARVEST --> SPOKE
    SCAFFOLD --> SPOKE
    
    HUB --> CI
    SPOKE --> CI
    CI --> LOCK
```

--- 

## ⚡ Quick Start: The Bootstrap Script

The entire Knowledge Base can be hydrated from scratch using the bootstrap utility. This script sequentially runs
ingestion, consensus discovery, scaffolding, ghost snapshotting, and synchronization for all supported frameworks.

**Run this when:**

* You have added a new framework adapter.
* You want to update mappings for newer versions of PyTorch/JAX/TF.
* You want to reset the semantic definitions to their upstream defaults.

```bash
# Full hydration cycle (Warning: Overwrites existing JSONs) 
./scripts/bootstrap.sh
```

--- 

## 🛠️ Phase 1: Ingestion (The Hub)

We maintain three tiers of "Abstract Standards" in `src/ml_switcheroo/semantics/` defining **WHAT** an operation is.

### Tier A: Math (Array API)

Derived from the Python Data API Consortium.

```bash
# 1. Clone the standard stubs
git clone -b 2024.12 --depth=1 https://github.com/data-apis/array-api _tmp/array-api

# 2. Import definitions to k_array_api.json
ml_switcheroo import-spec ./_tmp/array-api/src/array_api_stubs/_2024_12
```

### Tier B: Neural (ONNX)

Derived from the Open Neural Network Exchange (ONNX) operator set.

```bash
# 1. Fetch Operators docs
git clone --depth=1 -b v1.20.0 https://github.com/onnx/onnx _tmp/onnx

# 2. Parse Markdown to k_neural_net.json
ml_switcheroo import-spec ./_tmp/onnx/docs/Operators.md
```

### Discovery (Consensus Engine)

For operations not covered by official bodies (e.g., Optimizers, proprietary Layers), we use the **Consensus Engine**.
It scans all installed frameworks, clusters compatible API signatures (e.g., `Torch.Adam` vs `Flax.Adam`), and proposes
a unified standard.

```bash
# Scan installed libs and generate k_discovered.json
ml_switcheroo sync-standards --categories layer activation loss optimizer
```

--- 

## 🔗 Phase 2: Mapping (The Spokes)

Once the Hub (Specs) is populated, we link specific frameworks to it defining **HOW** operations are implemented. These
mappings live in `src/ml_switcheroo/snapshots/`.

### Mapping a Framework (`sync`)

The `sync` command introspects a library (e.g., `torch`) and matches its API surface against the known Spec.

```bash
# Link PyTorch implementation to the Standards
ml_switcheroo sync torch

# Link JAX implementation
ml_switcheroo sync jax
```

### Heuristic Scaffolding (`scaffold`)

For frameworks with non-standard naming conventions (e.g., `tensorflow`), use the `scaffold` command. It utilizes regex
patterns defined in the Framework Adapter's `discovery_heuristics` property to fuzzy-match APIs.

```bash
# Scan and populate mappings via regex heuristics
ml_switcheroo scaffold --frameworks tensorflow mlx
```

### Semantic Harvesting (`harvest`)

The most robust way to maintain mappings is to "Learn from Humans." If you write a manual test case fixing a translation
error, the Harvester can extract the rule back into the JSONs.

1. **Write/Fix a test** in `tests/examples/`:
   ```python
   def test_custom_add(): 
       # You manually fixed arguments: alpha -> scale
       jax.numpy.add(x, y, scale=0.5) 
   ```
2. **Run the extractor**:
   ```bash
   ml_switcheroo harvest tests/examples/test_custom_add.py --target jax
   ```

--- 

## 👻 Phase 3: Ghost Mode Support

ml-switcheroo can run in browser environments (WebAssembly) where heavy libraries like PyTorch cannot be installed. To
support this, we must capture raw API signatures.

### Capturing Snapshots

This command dumps the raw introspection data (signatures, docstrings, class hierarchies) of installed libraries into
JSON files. This data allows the `GhostInspector` to simulate the presence of the library during transpilation.

```bash
# Generates files like snapshots/torch_v2.1.0.json
ml_switcheroo snapshot --out-dir src/ml_switcheroo/snapshots
```

*Note: `bootstrap.sh` runs this automatically.*

--- 

## ✅ Phase 4: Verification (CI Loop)

We validate the mathematical correctness of mappings using a robotic fuzzer. It generates random inputs based on Type
Hints in the Spec, executes the operation in both Source and Target frameworks, and asserts equivalence.

### Running the Fuzzer

```bash
# 1. Install all backends
pip install ".[test]" 
pip install torch jax flax tensorflow mlx numpy

# 2. Run Verification Suite
ml_switcheroo ci
```

### Physical Test Generation

To ensure regression testing without running the full fuzzer every time, generate physical Python test files:

```bash
ml_switcheroo gen-tests --out tests/generated/test_tier_a_math.py
```

### Updating Compatibility Matrix

If the CI pass changes the support status of any operation, update the `README.md` table:

```bash
ml_switcheroo ci --update-readme
```

--- 

## 📚 Documentation & Web Demo

The project documentation (Sphinx) includes a client-side WebAssembly (WASM) demo powered by Pyodide.

### Building Docs & Wheel

The documentation build script automatically packages the current source into a `.whl` and injects it into the static
site assets.

```bash
python scripts/build_docs.py
```

--- 

## 🗃️ Glossary of Artifacts

The Knowledge Base is composed of specific JSON files with distinct roles.

| Artifact Path | Classification | Role & Purpose | Maintenance Strategy | 
| :--- | :--- | :--- | :--- | 
| `semantics/k_array_api.json` | **Hub (Spec)** | **Tier A (Math):** Basic array operations (abs, sum) derived from the Python Data API Consortium. | **Import** via `import-spec`. | 
| `semantics/k_neural_net.json` | **Hub (Spec)** | **Tier B (Neural):** stateful layers (Conv2d, LSTM) derived from ONNX Operators. | **Import** via `import-spec`. | 
| `semantics/k_framework_extras.json` | **Hub (Spec)** | **Tier C (Extras):** Utilities, IO, Devices. Often manually curated or scaffolded. | **Harvest** or **Wizard**. | 
| `semantics/k_discovered.json` | **Hub (Spec)** | **Consensus:** Ops discovered by overlapping API surfaces (Optimizers/Activations). | **Generate** via `sync-standards`. | 
| `snapshots/{fw}_v*_map.json` | **Spoke (Overlay)** | **Mapping Overlay:** Defines how a specific framework implements the specs. Contains API paths and Plugin hooks. | **Sync**, **Scaffold**, or **Harvest**. | 
| `snapshots/{fw}_v*.json` | **Ghost Snapshot** | **Raw API Dump:** Serialized signatures of the library. Used by `GhostInspector` in WASM. | **Capture** via `snapshot`. | 
