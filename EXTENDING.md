Extending
=========

**ml-switcheroo** is built on a modular, data-driven architecture. Supporting new operations or entirely new frameworks
involves populating the **Knowledge Base** rather than modifying the core engine logic.

There are three ways to extend the system, ordered by complexity:

1. **ODL (YAML)**: Define operations declaratively using the Operation Definition Language. (Recommended)
2. **Adapter API**: Write Python classes for full framework support.
3. **Plugin Hooks**: Write AST transformation logic for complex architectural mismatches.

---

## 🏗️ Architecture Overview

The extension system works by injecting definitions into the Knowledge Base (The Hub) and linking them to specific
framework implementations (The Spokes).

```mermaid
graph TD
%% Styles
    classDef hub fill: #f9ab00, stroke: #20344b, stroke-width: 2px, color: #20344b, font-family: 'Google Sans', rx: 5px
    classDef adapter fill: #4285f4, stroke: #20344b, stroke-width: 2px, stroke-dasharray: 0, color: #ffffff, font-family: 'Google Sans', rx: 5px
    classDef plugin fill: #34a853, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans', rx: 5px
    classDef tool fill: #ea4335, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans', rx: 5px

    subgraph EXT [Your Extension]
        direction TB
        ADAPTER("<b>Framework Adapter</b><br/>src/frameworks/*.py<br/><i>Definitions</i>"):::adapter
        PLUGIN("<b>Plugin Hooks</b><br/>src/plugins/*.py<br/><i>Complex Logic</i>"):::plugin
    end

    subgraph CORE [Core System]
        direction TB
        HUB("<b>Semantic Hub</b><br/>standards_internal.py<br/><i>Abstract Operations</i>"):::hub
    end

    subgraph TOOLS [Automation Tools]
        direction TB
        DEFINE("<b>CLI Command</b><br/>ml_switcheroo define<br/><i>Code Injection</i>"):::tool
        YAML("<b>ODL YAML</b><br/>Operation Definition<br/><i>Declarative Spec</i>"):::tool
    end

%% Wiring
    YAML --> DEFINE
    DEFINE -->|" 1. Inject Spec "| HUB
    DEFINE -->|" 2. Inject Mapping "| ADAPTER
    DEFINE -->|" 3. Scaffold File "| PLUGIN
    ADAPTER -->|" Registration "| HUB
    PLUGIN -->|" Transformation "| HUB
```

---

## 🚀 1. The Operation Definition Language (ODL)

The quickest way to add support for missing operations is via the `define` command. You write a YAML file describing the
operation, and the tool injects the necessary Python code into the system.

### Workflow

1. Create a file `my_op.yaml`.
2. Run `ml_switcheroo define my_op.yaml`.

This command automatically:

1. Updates the Abstract Standard (Hub) in `standards_internal.py`.
2. Updates Framework Mappings (Spokes) in `frameworks/*.py`.
3. Scaffolds Plugin files if complex logic is requested.

### ODL Schema Reference

```yaml
# 1. Abstract Definition (The Hub)
operation: "LogSoftmax"
description: "Applies the LogSoftmax function to an n-dimensional input Tensor."
std_args:
  - name: "input"
    type: "Tensor"
  - name: "dim"
    type: "int"
    default: "-1"

# 2. Variants (The Spokes)
variants:
  # Framework keys must match registered adapters (torch, jax, tensorflow, etc.)
  torch:
    api: "torch.nn.functional.log_softmax" # Direct mapping

  jax:
    api: "jax.nn.log_softmax"
    # Argument renaming: Standard 'dim' -> Framework 'axis'
    args:
      dim: "axis"

  # Advanced Feature: Auto-Inference
  # If you don't know the exact API path, use "infer" to let the engine search existing libraries.
  numpy:
    api: "infer"

  # Advanced Feature: Output Adaptation
  # Use Python lambdas to reshape return values (e.g., taking the first element of a tuple)
  # Useful for APIs that return (val, indices) when you only need val.
  tensorflow:
    api: "tf.nn.log_softmax"
    output_adapter: "lambda x: x[0]"

  # Advanced Feature: Infix Operators
  # Map a function call to a math operator 
  # Example: add(a, b) -> a + b
  mlx:
    api: "mx.add"
    transformation_type: "infix" # or "inline_lambda"
    operator: "+"
```

### Plugin Scaffolding via ODL

If an operation requires logic too complex for simple mapping (e.g., conditional dispatch based on argument values), you
can define **Rules** in the YAML. The CLI will generate a Python plugin with `if/else` logic pre-written.

```yaml
# Inside your YAML file:
scaffold_plugins:
  - name: "resize_dispatcher"
    type: "call_transform"
    doc: "Dispatches to different resize APIs based on interpolation mode."
    rules:
      - if_arg: "mode"
        is: "nearest"
        use_api: "jax.image.resize_nearest"
      - if_arg: "mode"
        is: "bilinear"
        use_api: "jax.image.resize_bilinear"
```

---

## 🔌 2. Adding a Framework Adapter

To support a new library (e.g., `tinygrad` or `my_lib`), you create a Python class that acts as the translation
interface.

**Location:** `src/ml_switcheroo/frameworks/my_lib.py`

```python
from typing import Dict, Tuple
from ml_switcheroo.frameworks.base import register_framework, FrameworkAdapter, StandardMap
from ml_switcheroo.semantics.schema import StructuralTraits


@register_framework("my_lib")
class MyLibAdapter:
    display_name = "My Library"

    # Optional: Inherit behavior (e.g. 'flax_nnx' inherits 'jax' math capabilities)
    inherits_from = None

    # Discovery configuration
    ui_priority = 100

    # --- 1. Import Logic ---
    @property
    def import_alias(self) -> Tuple[str, str]:
        # How is the library imported? alias is usually what users type (e.g. 'np', 'tf')
        return ("my_lib", "ml")

    @property
    def import_namespaces(self) -> Dict[str, Dict[str, str]]:
        # Remap source namespaces. 
        # Rules: If input uses 'torch.nn', we inject 'import my_lib.layers as nn'
        return {
            "torch.nn": {"root": "my_lib", "sub": "layers", "alias": "nn"},
        }

    # --- 2. Static Mappings (The "Definitions") ---
    # This property allows Ghost Mode to work without the library installed.
    @property
    def definitions(self) -> Dict[str, StandardMap]:
        return {
            # Simple 1:1 Mapping
            "Abs": StandardMap(api="ml.abs"),

            # Argument Renaming
            "Linear": StandardMap(
                api="ml.layers.Dense",
                args={"in_features": "input_dim", "out_features": "units"}
            ),

            # Linking to a Plugin (Logic located in src/plugins/)
            "permute_dims": StandardMap(
                api="ml.transpose",
                requires_plugin="pack_varargs"
            )
        }

    # --- 3. Structural Traits ---
    # Configure how Classes/Functions are rewritten without custom code
    @property
    def structural_traits(self) -> StructuralTraits:
        return StructuralTraits(
            module_base="ml.Module",  # Base class for layers
            forward_method="call",  # Inference method name
            requires_super_init=True,  # Inject super().__init__()?
            strip_magic_args=["rngs"],  # Remove args not used by this FW
            lifecycle_strip_methods=["to", "cpu"],  # Methods to silently remove
            impurity_methods=["add_"]  # Methods flagged as side-effects
        )
```

---

## 🧠 3. Plugin System (Custom Code)

For operations that require manipulating the AST structure (e.g., packing args, unwrapping state, injecting context
managers), you use the **Hook System**.

Create a python file in `src/ml_switcheroo/plugins/`. It will be automatically discovered.

### Auto-Wired Plugins

You can register a hook and definition its semantic mapping in one place using the `auto_wire` parameter. This
architecture maintains locality of behavior and avoids editing huge JSON files manually.

```python
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext


# Define the operation spec and logic in one go
@register_hook(
    trigger="custom_reshape_logic",
    auto_wire={
        "ops": {
            "Reshape": {
                "std_args": ["x", "shape"],
                "variants": {
                    "torch": {"api": "torch.reshape"},
                    "jax": {"api": "jnp.reshape", "requires_plugin": "custom_reshape_logic"}
                }
            }
        }
    }
)
def transform_reshape(node: cst.Call, ctx: HookContext) -> cst.Call:
    # 1. Inspect 'node' (The Call AST)
    # 2. Use 'ctx' to look up config or API mappings
    target_api = ctx.lookup_api("Reshape")  # returns "jnp.reshape"

    # 3. Perform transformation logic (e.g. check args and modify them)
    # This example grabs the function name, ignores args logic for brevity
    new_func = cst.Name("reshaped_manually")

    return node.with_changes(func=new_func)
```

### The Hook Context (`ctx`)

The context object passed to your function provides helper methods for robust plugin writing:

* `ctx.target_fw`: The active target framework key (string).
* `ctx.lookup_api(op_name)`: Resolve the API string for the current target via the Semantics Manager.
* `ctx.inject_signature_arg(name)`: Add an argument to the enclosing function definition (e.g., inject `rng` into
  `def forward(...)`).
* `ctx.inject_preamble(code)`: Add code to the start of the function body.
* `ctx.config(key)`: Read plugin settings from `pyproject.toml` or CLI args.
