Extending with DSL
==================

The **Operation Definition Language (ODL)** is a declarative YAML schema used to teach `ml-switcheroo` new mathematical concepts. It serves as the "DNA" of the transpiler, defining:

1.  **Semantic Interface**: Arguments, Types, Shapes, and Constraints.
2.  **Implementation logic**: How to map the operation to specific backends (Torch, JAX, TF, etc.).
3.  **Verification Data**: Hints for the automated fuzzer to prove correctness.

ODL allows you to inject logic into the **Knowledge Base** without writing Python AST transformation code.

---

## 🏗️ The ODL Lifecycle

Data flows from the declarative YAML file into the distributed Knowledge Base (Hub & Spoke), triggering the automatic generation of validation tests.

```mermaid
graph TD
    %% --- STYLE DEFINITIONS ---
    classDef default font-family:'Google Sans',color:#20344b,stroke:#20344b;
    classDef file fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef process fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px;
    classDef output fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef generated fill:#57caff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px;

    %% --- NODES ---
    YAML("<b>ODL YAML</b><br/>my_op.yaml"):::file
    
    subgraph CLI [" CLI: ml_switcheroo define "]
        direction TB
        PARSER("<b>ODL Parser</b><br/>Validates Schema"):::process
        INJECTOR("<b>Injectors</b><br/>Updates AST & JSONs"):::process
        
        PARSER --> INJECTOR
    end
    
    subgraph KB [" Knowledge Base "]
        direction TB
        HUB[("<b>The Hub</b><br/>standards.py<br/><i>Abstract Def</i>")]:::hub
        SPOKE[("<b>The Spokes</b><br/>frameworks/*.json<br/><i>Variants</i>")]:::hub
    end
    
    TEST_GEN("<b>Test Generator</b><br/>Builds PyTest Harness"):::process
    
    ARTIFACTS("<b>Generated Code</b><br/>tests/generated/test_my_op.py<br/>plugins/my_op_plugin.py"):::output

    %% --- EDGES ---
    YAML --> PARSER
    INJECTOR -->|" 1a. Writes spec "| HUB
    INJECTOR -->|" 1b. Writes maps "| SPOKE
    
    HUB --> TEST_GEN
    SPOKE --> TEST_GEN
    
    TEST_GEN -->|" 2. Creates "| ARTIFACTS
```

---

## 🤖 LLM-Assisted Workflow (The Fast Cycle)

Writing YAML manually is slow. `ml-switcheroo` includes a suite of CLI tools designed to put an LLM "in the loop" for rapid API coverage.

### 1. Find Missing Ops (`audit`)

Identify API calls in your codebase that are not yet mapped. Usage of `--json` allows piping to automation scripts.

```bash
# Check coverage
ml_switcheroo audit ./my_project --roots torch

# Output JSON for tooling
ml_switcheroo audit ./my_project --roots torch --json | jq '.[].api'
```

### 2. Generate Context (`suggest`)

The `suggest` command introspects the installed source library (live!) and generates a pre-filled prompt for an LLM. This prompt includes:

*   **Signatures & Docstrings**: Extracted via runtime introspection.
*   **ODL Schema**: The rigid ODL JSON schema required by the parser.
*   **One-Shot Example**: A valid baseline mapping.

```bash
# Generate prompt for a specific API
ml_switcheroo suggest torch.nn.functional.grid_sample > prompt.txt

# Copy prompt.txt to ChatGPT / Claude / Local LLM to get valid YAML.
```

### 3. Validate Safety (`dry-run`)

LLMs can hallucinate. Always preview the changes before injecting code.

```bash
# View Unified Diff of standards.py and framework adapters
ml_switcheroo define generated_op.yaml --dry-run
```

### 4. Inject & Verify

```bash
# Apply the definition and generate tests
ml_switcheroo define generated_op.yaml
```

### 5. Schema Export (`schema`)

If building custom agents, you can export the raw JSON schema for function calling.

```bash
ml_switcheroo schema > odl_schema.json
```

---

## 📚 The Schema at a Glance

A complete ODL definition looks like this:

```yaml
operation: "LogSoftmax"
description: "Applies the LogSoftmax function to an n-dimensional input Tensor."
op_type: "function" # function | context | decorator

# 1. Standard Arguments (The Abstract Signature)
std_args:
  - name: "input"
    type: "Tensor"
    rank: 4                 # Constraint: Must be 4D (e.g. NCHW)
    dtype: "float32"        # Constraint: Input must be float check
    shape_spec: "[B, C, ...]" # Symbolic shape hint for Fuzzer

  - name: "dim"
    type: "int"
    default: "-1"           # Default value if missing in source
    min: -2
    max: 3

# 2. Return Verification
return_type: "Tensor"
output_shape_calc: "lambda input, dim: input.shape" # Verifies output shape matches input

# 3. Framework Implementations
variants:
  torch:
    api: "torch.nn.functional.log_softmax"

  jax:
    api: "jax.nn.log_softmax"
    args:
      dim: "axis"           # Rename 'dim' -> 'axis'
    min_version: "0.4.0"    # Version constraints
    required_imports:
      - "import jax"
```

To apply this file:

```bash
# Preview
ml_switcheroo define my_op.yaml --dry-run
# Apply
ml_switcheroo define my_op.yaml
```

---

## 🧬 Feature Reference

### 1. Argument Normalization

The core job of ODL is pivoting arguments from **Source Names** to **Standard Names**, and then to **Target Names**.

```yaml
std_args: [ "x", "axis", "keepdims" ]
variants:
  torch:
    api: "torch.sum"
    args:
      axis: "dim"          # Map Spec 'axis' -> Torch 'dim'
      keepdims: "keepdim"
  jax:
    api: "jnp.sum"
    # JAX matches standard names, no mapping needed
```

### 2. Rich Parameter Constraints (Fuzzer Control)

You can attach metadata to `std_args` to constrain the inputs generated during verification (CI) or strict mode checking.

| Field | Description | Example |
| :--- | :--- | :--- |
| `type` | Python Type Hint string. | `"int"`, `"Tensor"`, `"List[int]"` |
| `default` | Default value (transpiled if arg missing). | `"1e-5"`, `"True"` |
| `rank` | Required tensor rank (number of dims). | `4` |
| `dtype` | Required data type. | `"float32"`, `"int64"`, `"bool"` |
| `shape_spec` | Symbolic shape string. | `"[B, T, D]"`, `"[N, N]"` |
| `min` / `max` | Numeric bounds for scalar generation. | `min: 0`, `max: 1` |
| `options` | Allowed values (Enumeration). | `["sum", "mean", "none"]` |

**Example: Convolution Weights**

```yaml
std_args:
  - name: "weight"
    type: "Tensor"
    rank: 4
    shape_spec: "[Out, In, K, K]" # Enforce square kernel in fuzzer
```

### 3. Conditional Dispatch (Rules)

Sometimes a single API mapping isn't enough. You can use **Dispatch Rules** to switch the target API based on the *value* or *type* of an argument at runtime.

**Supported Operators:** `eq`, `neq`, `gt`, `lt`, `in`, `not_in`, `is_type`.

```yaml
operation: "Resize"
std_args: [ "image", "mode" ]
variants:
  jax:
    api: "jax.image.resize" # Default
    dispatch_rules:
      # If mode == 'nearest', use specific function
      - if_arg: "mode"
        op: "eq"
        val: "nearest"
        use_api: "jax.image.resize_nearest"

      # If input is a List, use batch processor
      - if_arg: "image"
        op: "is_type"
        val: "list"
        use_api: "jax.image.resize_batch"
```

### 4. Argument Value Mapping (Enum Translation)

Map string literals or integers between frameworks.

```yaml
operation: "Reduce"
std_args: [ "x", "reduction" ]
variants:
  torch:
    api: "torch.reduce"
    # Logic: Source 'mean' -> Target 'avg'
    arg_values:
      reduction:
        mean: "'avg'"
        sum: "'add'"
```

### 5. Output Adaptation

Handle differences in return signatures.

*   **Selection:** If source returns a Tuple `(val, idx)` but target returns only `val`.
*   **Casting:** If target usually returns `float32` but spec requires `int64`.

```yaml
variants:
  jax:
    api: "jnp.max_indices"
    # Select index 0 from result tuple
    output_select_index: 0
    # Cast result to int64
    output_cast: "jnp.int64"
```

### 6. Tensor Layout Permutation

Automatically inject `transpose` / `permute` calls to align memory layouts (e.g. NCHW vs NHWC).

```yaml
operation: "Conv2d"
variants:
  jax:
    api: "jax.lax.conv"
    # Syntax: SOURCE_LAYOUT -> TARGET_LAYOUT
    layout_map:
      input: "NCHW->NHWC"
      weight: "OIHW->HWIO"
      return: "NHWC->NCHW"
```

### 7. Argument Packing & Variadics

Convert `func(*args)` to `func(list=[...])`.

```yaml
std_args:
  - name: "tensors"
    is_variadic: true # Accepts *tensors
variants:
  keras:
    api: "keras.layers.Add"
    # Packs *tensors into a list and passes to 'inputs' argument (implicit pos 0)
    pack_as: "List"
```

### 8. Constraint Injection via Metadata

Mark operations with specific flags to trigger built-in engine plugins without writing custom code.

```yaml
operation: "Add_"
is_inplace: true   # Triggers 'unroll_inplace_ops' plugin automatically
variants:
  torch: { api: "torch.add_" }
```

---

## 🔌 Advanced Configuration

### Version Constraints

Prevent invalid code generation if the target environment is too old or too new.

```yaml
variants:
  jax:
    api: "jax.scipy.special.logits"
    min_version: "0.4.0"
    max_version: "0.5.0"
```

### Dependency Management

Inject imports required by your mapping. The `ImportFixer` will place these at the top of the file and deduplicate them.

```yaml
variants:
  numpy:
    api: "np.sigmoid"
    # Can be simple strings or structured objects
    required_imports:
      - "import numpy as np"
      - module: "scipy.special"
        alias: "sp"
```

### Plugin Scaffolding

If ODL is not expressive enough, define a stub for a Python plugin loop. The CLI will generate the file `src/ml_switcheroo/plugins/{name}.py` for you to fill in.

```yaml
operation: "ComplexOp"
variants:
  jax:
    requires_plugin: "my_complex_logic"

# Define the stub to generate
scaffold_plugins:
  - name: "my_complex_logic"
    type: "call_transform"
    doc: "Handles complex logic for JAX."
    # Optional: Pre-compile rules into the python code
    rules:
      - if_arg: "x"
        op: "eq"
        val: 0
        use_api: "jax.zeros_like"
```

---

## 🧪 Verification Logic

The `gen-tests` command uses the metadata in your ODL to create physical test files.

*   `test_rtol` / `test_atol`: Set numerical tolerance for equivalence checks.
*   `nondeterministic`: Set to `true` to relax checks for RNG ops.
*   `verification_mode`: Set to `"exact"` for strict integer/boolean matching, or `"approx"` (default) for floating point tolerances.
*   `output_shape_calc`: A Python lambda string to verify output shape rigorously.
    ```yaml
    # Checks that output shape is input shape with last dim removed
    output_shape_calc: "lambda input, dim: input.shape[:-1]"
    ```
