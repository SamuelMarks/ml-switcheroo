Extending
=========

**ml-switcheroo** is built on a modular, data-driven architecture.

There are three ways to extend the system, ordered by complexity:

1.
    - **ODL (Operation Definition Language)**: Declaratively define operations using YAML or `StandardMap` objects. This handles 90% of cases (renaming, reordering, packing args, macros). **[See: [EXTENDING_WITH_DSL](EXTENDING_WITH_DSL.md)]**
    - Or update these Python files (no YAML DSL required):
      - `src/ml_switcheroo/semantics/standards_internal.py`
      - `src/ml_switcheroo/frameworks/*.py` (for torch, mlx, tensorflow, jax, etc.)
2.  **Adapter API**: Write Python classes to support entirely new frameworks (e.g. adding `TinyGrad` or `MindSpore`).
3.  **Plugin Hooks**: Write AST transformation logic for complex architectural mismatches that ODL cannot handle (e.g., state injection, context manager rewriting).

This document covers **2** and **3**.

---

## 🏗️ Architecture Overview

The extension system works by injecting definitions into the Knowledge Base (The Hub) and linking them to specific framework implementations (The Spokes).

```mermaid
graph TD
    %% Styles based on ARCHITECTURE.md theme
    classDef hub fill:#f9ab00,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:'Google Sans',rx:5px;
    classDef adapter fill:#4285f4,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef plugin fill:#34a853,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef tool fill:#ea4335,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:'Google Sans',rx:5px;
    classDef input fill:#ffffff,stroke:#20344b,stroke-width:1px,color:#20344b,font-family:'Roboto Mono';

    subgraph EXT [Your Extension]
        direction TB
        ADAPTER("<b>Framework Adapter</b><br/>src/frameworks/*.py<br/><i>Definitions & Traits</i>"):::adapter
        PLUGIN("<b>Plugin Hooks</b><br/>src/plugins/*.py<br/><i>AST Logic</i>"):::plugin
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

To support a new library (e.g., `tinygrad`, `custom_engine`), you create a Python class that acts as the translation interface. It converts the library's specific idioms into traits understood by the core engine.

**Location:** `src/ml_switcheroo/frameworks/my_lib.py`

```python
from typing import Dict, Tuple, List, Set, Any
from ml_switcheroo.frameworks.base import register_framework, FrameworkAdapter, StandardMap, ImportConfig
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits
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
    def import_namespaces(self) -> Dict[str, ImportConfig]:
        # Declare namespaces for the Import Fixer
        return {
            "my_lib": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="ml"),
            "my_lib.layers": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="layers"),
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

            # DSL Feature: Argument Packing (Variadic -> Tuple)
            # Replaces legacy 'pack_varargs' plugin
            "permute_dims": StandardMap(
                api="ml.transpose", 
                pack_to_tuple="axes" 
            ),

            # DSL Feature: Inline Macro
            "SiLU": StandardMap(
                macro_template="{x} * ml.sigmoid({x})" 
            ),
            
            # Linking to a Custom Plugin (Logic located in src/plugins/)
            "SpecialOp": StandardMap(
                requires_plugin="my_custom_logic"
            )
        }

    # --- 3. Structural Traits ---
    # Configure how Classes/Functions are rewritten without custom code
    @property
    def structural_traits(self) -> StructuralTraits:
        return StructuralTraits(
            module_base="ml.Module",           # Base class for layers
            forward_method="call",             # Inference method name
            requires_super_init=True,          # Inject super().__init__()?
            inject_magic_args=[],              # No special context args
            lifecycle_strip_methods=["gpu"],   # Methods to silently remove
            impurity_methods=["add_"]          # Methods flagged as side-effects
        )

    # --- 4. Plugin Traits ---
    # Configure how generic plugins interact with this framework
    @property
    def plugin_traits(self) -> PluginTraits:
        return PluginTraits(
            has_numpy_compatible_arrays=True,    # Supports .astype() casting?
            requires_explicit_rng=False,         # Requires JAX-style keys?
            requires_functional_state=False      # Requires BatchNorm unrolling?
        )
        
    @property
    def supported_tiers(self) -> List[SemanticTier]:
        return [SemanticTier.ARRAY_API, SemanticTier.NEURAL]
```

---

## 🧠 3. Plugin System (Custom Code)

For operations that require manipulating the AST structure (e.g. injecting imports, wrapping contexts, unwrapping state), you use the **Hook System**.

Create a python file in `src/ml_switcheroo/plugins/`. It will be automatically discovered.

### Anatomy of a Plugin

Plugins are functions decorated with `@register_hook`. They receive the current AST node and a Context object.

```python
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("my_custom_logic")
def transform_special_op(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    """
    Example: Transforms `special_op(x)` into `context_wrapper(x)`
    """
    # 1. Inspect Context
    # Check framework capabilities or configuration
    if not ctx.plugin_traits.has_numpy_compatible_arrays:
        return node
        
    # Look up API path dynamically (Decoupling)
    target_api = ctx.lookup_api("SpecialOp") or "default.op"
    
    # 2. Inject Dependencies (Preamble)
    if not ctx.metadata.get("my_helper_injected"):
        ctx.inject_preamble("import my_helper_lib")
        ctx.metadata["my_helper_injected"] = True
        
    # 3. Modify AST
    # Change function name
    parts = target_api.split(".")
    new_func = cst.Name(parts[0])
    for p in parts[1:]:
        new_func = cst.Attribute(value=new_func, attr=cst.Name(p))
        
    return node.with_changes(func=new_func)
```

### The Hook Context (`ctx`)

The context object passed to your function provides helper methods for robust plugin writing without hardcoding frameowrk strings:

*   `ctx.target_fw`: The active target framework key (string).
*   `ctx.plugin_traits`: A `PluginTraits` object describing the target (e.g., `requires_explicit_rng`). Prefer checking this over `target_fw`.
*   `ctx.lookup_api(op_name)`: Resolve the API string for the current target via the Semantics Manager.
*   `ctx.inject_signature_arg(name)`: Add an argument to the enclosing function definition (e.g., inject `rng` into `def forward(...)`).
*   `ctx.inject_preamble(code)`: Add code to the start of the function body or module header.
*   `ctx.current_variant`: Access the `FrameworkVariant` definition from ODL to read custom metadata (e.g. `args` map).

### Auto-Wired Plugins

You can register a hook and inject its semantic mapping ("Hub entry") in one place using the `auto_wire` parameter. This architecture maintains locality of behavior.

```python
@register_hook(
    trigger="custom_reshape",
    auto_wire={
        "ops": {
            "Reshape": {
                "std_args": ["x", "shape"],
                "variants": {
                    "torch": {"api": "torch.reshape"},
                    "jax": {"requires_plugin": "custom_reshape"}
                }
            }
        }
    }
)
def transform_reshape(node: cst.Call, ctx: HookContext) -> cst.Call:
    # Logic here...
    return node
```
