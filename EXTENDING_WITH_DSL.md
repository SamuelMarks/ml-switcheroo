Extending with ODL
==================

The **Operation Definition Language (ODL)** is a declarative YAML schema used to teach `ml-switcheroo` new mathematical
operations without writing Python AST transformation code or manually updating:

- `src/ml_switcheroo/semantics/standards_internal.py` and
- `src/ml_switcheroo/frameworks/*.py` (for torch, mlx, tensorflow, jax, etc.)

ODL serves as the "DNA" of the transpiler, defining **Inputs** (Arguments, Types, Shapes), **Behavior** (Math logic,
Side effects), and **Implementation** (How to map it to PyTorch, JAX, TensorFlow, etc.).

---

## 🔄 The Workflow: Audit & Define

The standard loop for adding missing operations is **Audit $\to$ Define $\to$ Inject**.

### 1. Audit your codebase

Run the `audit` command on your source project to find APIs that are not yet in the Knowledge Base.

```bash
# Check 'my_model.py' for unmapped PyTorch calls
ml_switcheroo audit my_model.py --roots torch
```

**Output:**

```text
❌ Missing Operations
Framework    API Name          Suggestion
torch        torch.erf         Run 'scaffold' or 'wizard'
torch        torch.reciprocal  Run 'scaffold' or 'wizard'
```

### 2. Describe the Operation (YAML)

Create a file named `math_ops.yaml` and describe the missing operations using ODL.

```yaml
# math_ops.yaml
operation: "Erf"
description: "Computes the error function of each element."
std_args:
  - name: "input"
    type: "Tensor"
variants:
  torch:
    api: "torch.erf"
  jax:
    api: "jax.lax.erf"
    required_imports: [ "import jax" ]
```

### 3. Inject into Knowledge Base

Run the `define` command. This parses the YAML, validates it against the schema, and injects the Python code directly
into the `ml-switcheroo` source tree.

```bash
ml_switcheroo define math_ops.yaml
```

**Output:**

```text
✅ Updated Hub: src/ml_switcheroo/semantics/standards_internal.py
✅ Updated Spoke (torch): src/ml_switcheroo/frameworks/torch.py
✅ Updated Spoke (jax): src/ml_switcheroo/frameworks/jax.py
```

---

## 🧬 ODL Feature Gallery

The DSL supports complex transpilation logic including layout permutation, argument packing, macros, and conditional
dispatch.

### 1. Basic Argument Renaming

Map argument names between frameworks (e.g., PyTorch `dim` vs JAX `axis`).

```yaml
operation: "Sum"
description: "Sum of array elements over a given axis."
std_args:
  - name: "x"
    type: "Tensor"
  - name: "axis"
    type: "int"
    default: "None"
variants:
  torch:
    api: "torch.sum"
    args:
      axis: "dim"  # Map Spec 'axis' -> Torch 'dim'
  jax:
    api: "jnp.sum"
    # JAX uses 'axis' natively, no mapping needed
```

### 2. Argument Injection & Casting

Inject fixed parameters required by the target framework but missing in the source.

```yaml
operation: "LayerNorm"
description: "Applies Layer Normalization."
std_args: [ "input", "normalized_shape", "eps" ]
variants:
  torch:
    api: "torch.nn.LayerNorm"
  jax:
    api: "flax.nnx.LayerNorm"
    args:
      normalized_shape: "num_features"
      eps: "epsilon"
    # Inject arguments not present in the Abstract Standard
    inject_args:
      use_fast_variance: true
    # Cast specific inputs to avoid type errors
    casts:
      epsilon: "float32"
```

### 3. Variadic Packing (Star-Args)

Convert variable positional arguments into a tuple (common in `permute` vs `transpose`).

```yaml
operation: "Permute"
description: "Permutes tensor dimensions."
std_args:
  - name: "input"
    type: "Tensor"
  # Mark 'dims' as variadic (*dims)
  - name: "dims"
    type: "int"
    is_variadic: true
variants:
  torch:
    api: "torch.permute"
    # Torch layout: permute(input, *dims)
  jax:
    api: "jax.numpy.transpose"
    # JAX layout: transpose(a, axes=(...))
    # This instructs the engine to pack *dims into a tuple and pass it to 'axes='
    pack_to_tuple: "axes"
```

### 4. Tensor Layout Permutation

Automatically inject `transpose` calls to align memory layouts (e.g., NCHW $\leftrightarrow$ NHWC).

```yaml
operation: "Conv2d"
description: "2D Convolution."
std_args: [ "input", "weight" ]
variants:
  torch:
    api: "torch.nn.functional.conv2d"
    # Torch assumes NCHW
  jax:
    api: "jax.lax.conv"
    args:
      input: "lhs"
      weight: "rhs"
    # JAX LAX expects NHWC. The Engine will wrap inputs/outputs automatically.
    # Notation matches Einstein summation or explicit dimension chars.
    layout_map:
      input: "NCHW->NHWC"
      weight: "OIHW->HWIO"
      return: "NHWC->NCHW"
```

### 5. Macros (Composite Operations)

Define operations as inline Python expressions rather than function calls. Useful for activation functions or simple
math combos.

```yaml
operation: "Swish"
std_args: [ "x" ]
variants:
  torch:
    api: "torch.nn.functional.silu"
  jax:
    # Defines an inline template. {x} is replaced by the argument variable.
    macro_template: "{x} * jax.nn.sigmoid({x})"
    required_imports: [ "import jax" ]
```

### 6. Infix Operators

Convert function calls to Python operators (e.g., `add(a, b)` $\to$ `a + b`).

```yaml
operation: "Add"
std_args: [ "a", "b" ]
variants:
  torch:
    api: "torch.add"
  numpy:
    # Transformation: np.add(a, b) -> a + b
    transformation_type: "infix"
    operator: "+"
```

### 7. Output Adaptation

Handle API mismatches in return values (e.g., Tuple vs Scalar).

```yaml
operation: "Max"
std_args: [ "x" ]
variants:
  torch:
    api: "torch.max"
    # Torch returns (values, indices). We only want values.
    # Wraps result: (torch.max(x))[0]
    output_select_index: 0
  jax:
    api: "jnp.max"
    # Wraps result: (lambda x: x.astype(jnp.float32))(...)
    output_cast: "jnp.float32"
```

### 8. Conditional Dispatch (Rules)

Dynamically switch the target API based on the *value* of an argument.

```yaml
operation: "Resize"
std_args: [ "image", "size", "mode" ]
variants:
  jax:
    api: "jax.image.resize" # Default
    dispatch_rules:
      # If mode == 'nearest', switch API
      - if_arg: "mode"
        op: "eq"
        val: "nearest"
        use_api: "jax.image.resize_nearest"
      # If mode in ['bilinear', 'bicubic'], switch API
      - if_arg: "mode"
        op: "in"
        val: [ "bilinear", "bicubic" ]
        use_api: "jax.image.resize_bilinear"
```

---

## 🧪 Verification Constraints (Fuzzer)

You can guide the automated Fuzzer (CI) by adding constraints to `std_args`. This ensures generated tests use valid
inputs.

```yaml
operation: "LogSoftmax"
std_args:
  - name: "input"
    type: "Tensor"
    # Generate 4D tensors
    rank: 4
    # Enforce float32
    dtype: "float32"
    # Symbolic Shape: Start with Batch, End with Channels
    shape_spec: "[B, ..., C]"

  - name: "dim"
    type: "int"
    # Constrain random integer generation
    min: -1
    max: 3

  - name: "reduction"
    type: "str"
    # Restrict to enum values
    options: [ "mean", "sum", "none" ]

# Verify output shape matches input shape
output_shape_calc: "lambda input, dim, reduction: input.shape"
```

---

## 🔌 Plugin Scaffolding

If ODL is not expressive enough (e.g., requires complex AST restructing), you can ask the `define` command to generate a
Python plugin file for you.

```yaml
operation: "StrangeOp"
std_args: [ "x" ]
variants:
  jax:
    # Link to a plugin hook
    requires_plugin: "my_strange_logic"

# Define the plugin stubs to generate
scaffold_plugins:
  - name: "my_strange_logic"
    type: "call_transform"
    doc: "Handles strange logic for JAX."
    # Optional: Pre-compile some rules into the python code
    rules:
      - if_arg: "x"
        op: "eq"
        val: 0
        use_api: "jax.zeros_like"
```

Running `ml_switcheroo define strange.yaml` will create `src/ml_switcheroo/plugins/my_strange_logic.py` with the
boilerplate code ready for editing.
