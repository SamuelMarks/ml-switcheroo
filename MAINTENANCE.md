Maintenance
===========

ml-switcheroo is a data-driven system. Maintenance primarily involves keeping the **Semantic Knowledge Base** (`src/ml_switcheroo/semantics/*.json`) synchronized with upstream Standards (ONNX, Array API) and downstream Implementations (PyTorch, JAX, etc.).

This guide covers the 4 stages of the maintenance lifecycle: **Ingestion**, **Discovery**, **Verification**, and **Release**.

---

## 1. Ingestion (Upstream Standards)

When Spec Bodies (maintained by the Python Consortium or Linux Foundation) update their definitions, we import them to establish the "Abstract Standard".

### Updating the Math Standard (Array API)

1. Clone/Download the latest stubs from [data-apis/array-api](https://github.com/data-apis/array-api).
2. Run the importer pointing to the stub directory.

```bash
$ git clone -b 2024.12 --depth=1 https://github.com/data-apis/array-api
$ ml_switcheroo import-spec ./array-api/src/array_api_stubs/_2024_12

# Output:
# ℹ️  Detected Array API Stubs Directory
# ℹ️  Parsing 19 stub files...
# ℹ️  Merging with existing 182 entries...
# ✅ Saved 182 operations to src/ml_switcheroo/semantics/k_array_api.json
```

### Updating the Neural Standard (ONNX)

1. Fetch the `Operators.md` from the [ONNX repository](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
2. Run the markdown importer.

```bash
$ git clone --depth=1 -b v1.20.0 https://github.com/onnx/onnx

$ ml_switcheroo import-spec ./onnx/docs/Operators.md

# Output:
# ℹ️  Detected ONNX Markdown Spec: Operators.md
# ℹ️  Parsing ONNX Spec: Operators.md...
# ℹ️  Merging with existing 5 entries...
# ✅ Saved 202 operations to src/ml_switcheroo/semantics/k_neural_net.json
```

---

## 2. Operation-connect (Mapping ML array & math APIs)

This step links the Abstract Standards loaded in Step 1 to concrete implementations found in installed libraries. It writes to `src/ml_switcheroo/snapshots/{fw}_mappings.json`.

```bash
$ uv pip install keras jax tensorflow torch

$ for lib in keras jax tensorflow torch; do
    ml_switcheroo sync "$lib"
  done

# Output:
# ℹ️  Syncing keras against Array API Standard...
# ✅ Linked 95 operations for keras (Skipped 26 mismatches).
# ℹ️  Syncing jax against Array API Standard...
# ✅ Linked 146 operations for jax (Skipped 26 mismatches).
# ℹ️  Syncing tensorflow against Array API Standard...
# ✅ Linked 89 operations for tensorflow (Skipped 10 mismatches).
# ℹ️  Syncing torch against Array API Standard...
# ✅ Linked 149 operations for torch (Skipped 5 mismatches).
```

### Specialized Frameworks (e.g. PaxML)

Note: PaxML often requires specific environment constraints (e.g., Linux x86_64, CPython 3.10).

```bash
$ uv pip install paxml jaxlib==0.4.26
$ ml_switcheroo sync paxml

# Output:
# ℹ️  Syncing paxml against Standard...
# ✅ Linked 0 operations for paxml (Skipped 3 mismatches)
```

**Note:** The `sync` command combines findings from all Semantic Tiers (Math, Neural, Extras) into a single overlay file per framework.

---

## 3. Discovery (Mapping Frameworks)

Once standards are defined, we check for APIs that exist in frameworks but *missed* the strict matching in Step 2.

### A. Batch Scaffolding (Automated)

When adding a fresh framework or updating a major version (e.g., PyTorch 2.x -> 3.x), use the Scaffolder. It uses the `discovery_heuristics` defined in `FrameworkAdapter` classes to fuzzy-match APIs.

```bash
# Scan installed libraries and propose mappings
ml_switcheroo scaffold --frameworks torch jax
```

### B. Interactive Mapping (The Wizard)

For APIs that don't match heuristics (e.g., `torch.rfft` vs `jax.numpy.fft.rfft`), use the interactive wizard. This handles argument renaming (`dim` -> `axis`) and plugin assignment.

```bash
ml_switcheroo wizard torch
```

### C. Semantic Harvesting (Human-in-the-Loop)

The most robust way to maintain mappings is to "Learn from Humans." If you write a manual test case fixing a translation error, the Harvester can extract the rule back into the JSONs.

1. Write/Fix a test in `tests/examples/`:
   ```python
   def test_custom_add():
       # You manually fixed arguments: alpha -> scale
       jax.numpy.add(x, y, scale=0.5)
   ```
2. Run the extractor:
   ```bash
   ml_switcheroo harvest tests/examples/test_custom_add.py --target jax
   ```

---

## 4. Verification (CI Loop)

We validate mappings using two methods: **Robotic Fuzzing** (using Types from Specs) and **Physical Test Files**.

### Running the Equivalence Runner

This runs the `InputFuzzer` against every entry in the Knowledge Base (Semantics + Snapshots).

```bash
# Run full suite
ml_switcheroo ci

# Generate a "Lockfile" of verified operations
ml_switcheroo ci --json-report verified_ops.json
```

*Note: The `verified_ops.json` can be referenced in `pyproject.toml` to prevent the engine from generating code for unverified operations.*

### Regenerating Physical Tests

Ideally, we generate physical python test files (`tests/generated/`) to commit to the repo. This ensures regression testing even without running the full fuzzer.

```bash
ml_switcheroo gen-tests --out tests/generated/test_tier_a_math.py
```

---

## 5. Documentation & Web Demo

Maintenance of the documentation site and the WASM demo.

### Compatibility Matrix

To update the `README.md` table with the latest verification status from the CI loop:

```bash
ml_switcheroo ci --update-readme
```

### Migration Guides

Generate text-based comparison documents for users.

```bash
ml_switcheroo gen-docs --source torch --target jax --out docs/MIGRATION_GUIDE.md
```

### Building the WASM Demo

The documentation includes a client-side transpiler running via Pyodide. To update it:

1. **Build the Wheel**: The doc builder needs a `whl` of `ml-switcheroo` to inject into the static site.
2. **Run Sphinx**:
   ```bash
   python scripts/build_docs.py
   ```
   *This script automatically builds the package wheel, copies it to `docs/_static`, and compiles the HTML.*

---

## Glossary of Artifacts

| File | Purpose | Maintenance Strategy |
| :--- | :--- | :--- |
| `semantics/k_array_api.json` | Math Operations | **Import** via `import-spec` from Array API Stubs. |
| `semantics/k_neural_net.json` | Layers & Stateful Ops | **Import** via `import-spec` from ONNX. |
| `semantics/k_framework_extras.json` | IO, Devices, Utils | **Harvest** from manual code or **Wizard**. |
| `snapshots/{fw}_mappings.json` | Framework Implementation Overlays | **Sync** command, **Scaffold**, or **Wizard**. |
