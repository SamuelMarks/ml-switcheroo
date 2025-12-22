Extending
=========

**ml-switcheroo** is built on a modular, data-driven architecture.

There are three ways to extend the system, ordered by complexity:

1.
    - **ODL (YAML)**: Define operations declaratively. **[See: [EXTENDING_WITH_DSL](EXTENDING_WITH_DSL.md)]** 
    - Or update these Python files (no YAML DSL required):
      - `src/ml_switcheroo/semantics/standards_internal.py`
      - `src/ml_switcheroo/frameworks/*.py` (for torch, mlx, tensorflow, jax, etc.)
3. **Adapter API**: Write Python classes for full framework support (e.g. adding `TinyGrad`).
3. **Plugin Hooks**: Write AST transformation logic for complex architectural mismatches.

This document covers **2** and **3**. For adding mathematical operations, please refer to the DSL guide.

---

## 🏗️ Architecture Overview

The extension system works by injecting definitions into the Knowledge Base (The Hub) and linking them to specific
framework implementations (The Spokes).

```mermaid
graph TD
%% Styles based on ARCHITECTURE.md theme
    classDef hub fill: #f9ab00, stroke: #20344b, stroke-width: 2px, color: #20344b, font-family: 'Google Sans', rx: 5px
    classDef adapter fill: #4285f4, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans', rx: 5px
    classDef plugin fill: #34a853, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans', rx: 5px
    classDef tool fill: #ea4335, stroke: #20344b, stroke-width: 2px, color: #ffffff, font-family: 'Google Sans', rx: 5px
    classDef input fill: #ffffff, stroke: #20344b, stroke-width: 1px, color: #20344b, font-family: 'Roboto Mono'

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
        YAML("<b>ODL YAML</b><br/>Operation Definition<br/><i>Declarative Spec</i>"):::input
    end

%% Wiring
    YAML --> DEFINE
    DEFINE -->|" 1. Inject Spec "| HUB
    DEFINE -->|" 2. Inject Mapping "| ADAPTER
    DEFINE -->|" 3. Scaffold File "| PLUGIN
    ADAPTER -->|" Registration "| HUB
    PLUGIN -.->|" AST Transformation "| HUB
```

---

## 🔌 2. Adding a Framework Adapter

To support a new library (e.g., `tinygrad`, `custom_engine`), you create a Python class that acts as the translation
interface. It converts the library's specific idioms into traits understood by the core engine.

**Location:** `src/ml_switcheroo/frameworks/my_lib.py`

```python
from typing import Dict, Tuple, List
from ml_switcheroo.frameworks.base import register_framework, FrameworkAdapter, StandardMap
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.enums import SemanticTier


@register_framework("my_lib")
class MyLibAdapter:
    display_name = "My Library"

    # Optional: Inherit implementation behavior (e.g., 'flax_nnx' inherits 'jax' math)
    inherits_from = None

    # Discovery configuration
    ui_priority = 100

    # --- 1. Import Logic ---
    @property
    def import_alias(self) -> Tuple[str, str]:
        # How is the library imported? (Package Name, Common Alias)
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
    # It populates the Spokes.
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
            ),

            # Inline Transformation
            "Add": StandardMap(
                api="ml.add",
                transformation_type="infix",
                operator="+"
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

    # --- 4. Tiers ---
    @property
    def supported_tiers(self) -> List[SemanticTier]:
        return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]
```

---

## 🧠 3. Plugin System (Custom Code)

For operations that require manipulating the AST structure (e.g., packing args, unwrapping state, injecting context
managers), you use the **Hook System**.

Create a python file in `src/ml_switcheroo/plugins/`. It will be automatically discovered.

### Auto-Wired Plugins

You can register a hook and define its semantic mapping ("Hub entry") in one place using the `auto_wire` parameter. This
architecture maintains locality of behavior and avoids editing multiple JSON files manually.

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

    # 3. Perform transformation logic (e.g., check args and modify them)
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
