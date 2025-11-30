Extending
=========

Welcome! ml-switcheroo is designed as a modular platform. We strictly separate **Core Logic** (AST parsing via `LibCST`)
from **Knowledge** (API Mappings). This architecture allows you to add support for new operators, obscure libraries, or
even entirely new backends (like Apple MLX or TinyGrad) without rewriting the compiler engine.

This guide covers:

1. **Semantic Taxonomy:** Where to store mappings.
2. **Automated Discovery:** Using the CLI to write JSON for you.
3. **Writing Plugins:** Creating AST transformations for complex behaviors.
4. **Extending Validation:** Adding new backends to the Fuzzer.

---

## 1. The Semantic Taxonomy (Where does data go?)

Before writing code, understand where the data lives. We categorize mappings into three distinct JSON "Tiers" in
`src/ml_switcheroo/semantics/`.

### Tier A: The Math Standard (`k_array_api.json`)

* **Scope:** Raw tensor manipulations (`sum`, `matmul`, `reshape`, `cos`, `abs`).
* **Source of Truth:** [Python Array API Standard](https://data-apis.org/array-api/latest/).
* **Rule:** Standard argument names are strict (e.g., `x`, `axis`, `keepdims`).

### Tier B: The Neural Standard (`k_neural_net.json`)

* **Scope:** Deep Learning Layers and stateful objects (`Conv2d`, `Linear`, `MultiHeadAttention`).
* **Source of Truth:** [ONNX Operators](https://onnx.ai/onnx/operators/).
* **Rule:** If it has learnable weights (`self.weight`) or training state (`self.running_mean`), it belongs here.

### Tier C: The Extras Bin (`k_framework_extras.json`)

* **Scope:** Framework-specific utilities that fall outside general standards.
* **Examples:** Gradient transforms (`no_grad`), Data Loaders (`DataLoader`), Distributions (`Bernoulli`), System (
  `manual_seed`).
* **Rule:** Use this for "glue code" that requires framework-specific shims.

---

## 2. Workflow: Automated Discovery

**Stop!** Do not write JSON by hand unless absolutely necessary. Use the CLI tools to generate standardized entries.

### A. The Interactive Wizard (Recommended)

This tool guides you through categorizing functions and renaming arguments.

```bash
# Start the wizard for the 'torch' package
ml_switcheroo wizard torch
```

1. It scans the library for APIs missing from our Knowledge Base.
2. It asks you to bucket them (`[M]ath`, `[N]eural`, `[E]xtras`).
3. It asks you to map implementation arguments (e.g., `input`, `dim`) to standard arguments (`x`, `axis`).
4. It saves the valid JSON to the correct file.

### B. The Semantic Harvester (Advanced)

If you have written a manual test case that passes, ml-switcheroo can reverse-engineer the mapping rules from your test
code.

1. Write a passing test in `tests/examples/my_custom_op.py`.
2. Run the harvester:
   ```bash
   ml_switcheroo harvest tests/examples/my_custom_op.py --target jax
   ```
3. The tool detects `jax.numpy.op(a=x, b=y)` calls and automatically updates `k_array_api.json` to map `x->a` and
   `y->b`.

---

## 3. Advanced: Writing AST Plugins

JSON mappings handle 1:1 name swaps (`torch.abs` -> `jnp.abs`) and argument renaming (`dim` -> `axis`). However,
real-world deep learning code often requires structural changes.

**Plugins** allow you to write Python functions that manipulate the Abstract Syntax Tree (AST) directly using `LibCST`.

### The Hook Registry

Plugins are located in `src/ml_switcheroo/plugins/`. You register a transformation using the `@register_hook` decorator.

```python
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


@register_hook("my_custom_logic")
def transform_node(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    # ... modification logic ...
    return node
```

### The `HookContext`

The `ctx` object passed to your hook provides powerful helpers to modify the surrounding scope:

* `ctx.inject_signature_arg(name, annotation)`: Adds a new argument to the function definition (e.g., adding `rng` to
  `forward`).
* `ctx.inject_preamble(statement)`: Inserts code at the very top of the function body.
* `ctx.lookup_api("add")`: Finds the target API path for "add" based on the current configuration.
* `ctx.raw_config(key)`: Reads settings from `pyproject.toml`.

### Example 1: Argument Decomposition (`decompose_alpha`)

**Problem:** `torch.add(x, y, alpha=2)` exists, but JAX `add` has no `alpha`. We must rewrite it to `add(x, y * 2)`.

```python
# src/ml_switcheroo/plugins/decompositions.py

@register_hook("decompose_alpha")
def transform_alpha_add(node: cst.Call, ctx: HookContext) -> cst.Call:
    # 1. Check for alpha argument
    alpha_arg = None
    clean_args = []
    for arg in node.args:
        if arg.keyword and arg.keyword.value == "alpha":
            alpha_arg = arg
        else:
            clean_args.append(arg)

    if not alpha_arg:
        return node

        # 2. Create Multiplication AST Node: (y * alpha)
    # We assume the second argument is the one being scaled
    target_val = clean_args[1].value
    mult_expr = cst.BinaryOperation(
        left=target_val,
        operator=cst.Multiply(),
        right=alpha_arg.value
    )

    # 3. Update args list with the new expression
    clean_args[1] = clean_args[1].with_changes(value=mult_expr)

    # 4. Return new call (Function renaming happens automatically via Semantics)
    return node.with_changes(args=clean_args)
```

**Linking the Plugin:**
In `k_array_api.json`:

```json
{
  "add": {
    "variants": {
      "torch": {
        "api": "torch.add"
      },
      "jax": {
        "api": "jax.numpy.add",
        "requires_plugin": "decompose_alpha"
      }
    }
  }
}
```

### Example 2: State Injection (`inject_prng`)

**Problem:** PyTorch relies on global RNG. JAX requires stateless explicit key passing.
**Goal:** Transform `dropout(x)` inside `forward(x)` into `bernoulli(key, x)` inside `forward(rng, x)`.

```python
# src/ml_switcheroo/plugins/rng_threading.py
import libcst as cst

from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("inject_prng")
def inject_prng_threading(node: cst.Call, ctx: HookContext) -> cst.Call:
    if ctx.target_fw != "jax":
        return node

    # 1. Modify Function Signature
    # Changes `def forward(self, x)` to `def forward(self, rng, x)`
    ctx.inject_signature_arg("rng", annotation="jax.Array")

    # 2. Add Key Splitting Preamble
    # Inserts `rng, key = jax.random.split(rng)` at start of function
    ctx.inject_preamble("rng, key = jax.random.split(rng)")

    # 3. Modify the Call Site
    # Adds `key=key` to the function call arguments
    key_arg = cst.Arg(keyword=cst.Name("key"), value=cst.Name("key"))
    new_args = list(node.args) + [key_arg]

    return node.with_changes(args=new_args)
```

---

## 4. Extending Validation (New Frameworks)

ml-switcheroo includes a fuzzing engine that verifies translations by running inputs through both frameworks. If you want
to add support for a new backend (e.g., **TinyGrad** or a custom internal library), you must write a `FrameworkAdapter`.

### Step 1: Create the Adapter in `ml_switcheroo.testing.adapters`

An adapter simply tells the fuzzer how to convert a NumPy array into the target framework's tensor format.

```python
# src/ml_switcheroo/testing/adapters.py
from typing import Any

class TinyGradAdapter:
    @staticmethod
    def convert(data: Any) -> Any:
        try:
            from tinygrad.tensor import Tensor
            return Tensor(data)
        except ImportError:
            return data
```

### Step 2: Register the Adapter

Updates `_ADAPTER_REGISTRY` in the same file.

```python
from ml_switcheroo.testing.adapters import register_adapter

register_adapter("tinygrad", TinyGradAdapter)
```

### Step 3: Run Verification

Now you can validate mappings against this new backend.

```bash
# Update PyProject.toml or use flags
ml_switcheroo ci --target tinygrad
```

---

## 5. Configuration

Plugins can read arbitrary configuration values from `pyproject.toml`.

**User's `pyproject.toml`:**

```toml
[tool.ml_switcheroo]
# ... standard settings ...

[tool.ml_switcheroo.plugin_settings]
my_plugin_threshold = 0.85
enable_experimental_rewrites = true
```

**Accessing within a Plugin:**

```python
from ml_switcheroo.core.hooks import register_hook

@register_hook("my_plugin")
def my_hook(node, ctx):
    threshold = ctx.raw_config("my_plugin_threshold", default=0.5)
    if ctx.raw_config("enable_experimental_rewrites"):
        return apply_complex_logic(node, threshold)
    return node
```
