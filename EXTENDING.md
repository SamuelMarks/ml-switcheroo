Extending ml-switcheroo
=======================

ml-switcheroo is built on a modular "Zero-Edit" architecture. You can add support for new Machine Learning frameworks (like Keras, TinyGrad, or generic NumPy wrappers), new backends, or custom patterns without modifying the core engine logic.

This guide covers:

1.  **Adding a New Framework**: Using the `FrameworkAdapter` protocol.
2.  **Mapping APIs**: Populating the Semantic Knowledge Base.
3.  **Complex Logic**: Handling Data Loaders and Custom Patterns via Plugins.
4.  **Verification**: Validating your new backend against the standard.

---

## 1. Adding a New Framework or Backend

To support a new library (e.g., `my_framework`), you must define an **Adapter**. This tells the system how to import the library, how to construct tensors for testing, and how its classes (Layers/Modules) are structured.

### Step 1: Create the Adapter File

Create a new file in `src/ml_switcheroo/frameworks/` (e.g., `my_framework.py`).

Implement a class decorated with `@register_framework` that satisfies the `FrameworkAdapter` protocol.

```python
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .base import register_framework, StructuralTraits

@register_framework("my_framework")  # Unique key used in CLI (e.g. --target my_framework)
class MyFrameworkAdapter:
    # --- UI & Metadata ---
    display_name: str = "My Framework"
    ui_priority: int = 100  # Order in the Compatibility Matrix

    # --- Discovery Configuration ---
    # Modules to scan when running `ml_switcheroo sync my_framework`
    @property
    def search_modules(self) -> List[str]:
        return ["my_framework", "my_framework.nn", "my_framework.ops"]

    # Default import alias handling (e.g. 'import my_framework as mf')
    @property
    def import_alias(self) -> Tuple[str, str]:
        return ("my_framework", "mf")

    # Regex patterns to help the Scaffolder categorize APIs automatically
    @property
    def discovery_heuristics(self) -> Dict[str, List[str]]:
        return {
            "neural": [r"\.layers\.", r"Module$"],
            "extras": [r"\.data\.", r"\.io\."]
        }

    # --- Structural Traits (The "Zero-Edit" Rewriter Config) ---
    # Defines how Classes and Functions should be transformed.
    @property
    def structural_traits(self) -> StructuralTraits:
        return StructuralTraits(
            module_base="my_framework.Module",  # Base class for layers
            forward_method="call",              # Method name for inference (forward/call)
            requires_super_init=True,           # Does __init__ require super().__init__()?
            init_method_name="__init__",        # Constructor name (usually __init__, sometimes setup)
            # Arguments to inject into signatures (e.g. rngs for JAX)
            inject_magic_args=[],
            # Methods to strip from chains during conversion (e.g. .cuda(), .detach())
            lifecycle_strip_methods=["to_gpu", "detach"],
            # In-place methods that violate purity
            impurity_methods=["add_", "copy_"] 
        )

    @property
    def rng_seed_methods(self) -> List[str]:
        return ["seed", "set_seed"]

    # --- Test Harness Generation Support ---
    
    @classmethod
    def get_import_stmts(cls) -> str:
        return "import my_framework as mf"

    @classmethod
    def get_creation_syntax(cls, var_name: str) -> str:
        # Code to convert numpy array `var_name` to tensor
        return f"mf.tensor({var_name})"

    @classmethod
    def get_numpy_conversion_syntax(cls, var_name: str) -> str:
        # Code to convert tensor back to numpy
        return f"{var_name}.numpy()"

    # --- Device & IO Support ---
    
    def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
        return f"mf.device('{device_type}')"

    def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
        if op == "save":
            return f"mf.save({object_arg}, {file_arg})"
        return f"mf.load({file_arg})"
        
    def get_serialization_imports(self) -> List[str]:
        return ["import my_framework as mf"]

    # --- Runtime Conversion (for Fuzzer) ---
    def convert(self, data: Any) -> Any:
        # Runtime logic to convert data to this framework (used during `ml_switcheroo ci`)
        try:
            import my_framework
            return my_framework.tensor(data)
        except ImportError:
            return data
```

See `src/ml_switcheroo/frameworks/base.py` for the full protocol definition.

### Step 2: Register Dependencies

If your new adapter requires third-party libraries (like `tinygrad` or `keras`), ensure they are installed in your environment. The adapter file itself is lazy-loaded, but methods like `convert` will run inside the verification harness.

---

## 2. Mapping Semantic Operations

After creating the adapter, the system knows *how* to write code for your framework, but it doesn't know *which* functions map to the standard operations (e.g., `abs`, `conv2d`).

You must populate the Knowledge Base (`src/ml_switcheroo/semantics/*.json`).

### Automated Discovery (Recommended)

1.  **Sync**: Scans your installed library for functions that match standard names (e.g., if you have `my_framework.add` and the standard has `add`).
    ```bash
    ml_switcheroo sync my_framework
    ```

2.  **Scaffold**: Uses the regex heuristics in your adapter to find and categorize new APIs that aren't in the standard yet.
    ```bash
    ml_switcheroo scaffold --frameworks my_framework
    ```

### Interactive Mapping (The Wizard)

For APIs that don't match standard names (e.g., `my_framework.reduction_sum` vs `sum`), use the wizard:

```bash
ml_switcheroo wizard my_framework
```

This will prompt you to categorize functions and map arguments (e.g., map `input_tensor` to standard `x`).

---

## 3. Extending Support (Data Loaders & Custom Patterns)

Some features, like Data Loaders or Distributed Contexts, don't map 1:1 between frameworks. You can handle these using **Plugins**.

### Step 1: Write a Plugin Hook

Create a file in `src/ml_switcheroo/plugins/` (e.g., `custom_patterns.py`).

```python
import libcst as cst
from ml_switcheroo.core.hooks import register_hook, HookContext

@register_hook("convert_my_dataloader")
def transform_dataloader(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    """
    Transforms torch.utils.data.DataLoader(...) into my_framework.Dataset(...)
    """
    # 1. Check target
    if ctx.target_fw != "my_framework":
        return node

    # 2. Inject helper imports if needed
    ctx.inject_preamble("from my_framework import Dataset")

    # 3. Modify valid arguments
    new_args = []
    for arg in node.args:
        if arg.keyword and arg.keyword.value == "batch_size":
            new_args.append(arg)
        # Filter out incompatible args like 'num_workers' if not supported

    # 4. Rewrite function name
    new_func = cst.Name("Dataset")
    
    return node.with_changes(func=new_func, args=new_args)
```

### Step 2: Link the Plugin in Semantics

Add an entry to `src/ml_switcheroo/semantics/k_framework_extras.json`.

```json
"DataLoader": {
  "description": "Abstract Data Loader",
  "std_args": ["dataset", "batch_size"],
  "variants": {
    "torch": { "api": "torch.utils.data.DataLoader" },
    "my_framework": {
      "api": "my_framework.Dataset",
      "requires_plugin": "convert_my_dataloader"
    }
  }
}
```

Now, when converting *to* `my_framework`, any `DataLoader` call will trigger your plugin.

---

## 4. Verification

Once your Adapter and Semantics are in place, you can verify the integration using the built-in fuzzer.

```bash
# Run the CI suite targeting your new framework
ml_switcheroo ci --target my_framework
```

This process:
1.  Reads the Semantics Knowledge Base.
2.  Generates Python test files using the template strings from your Adapter (`get_creation_syntax`, etc.).
3.  Executes the tests, fuzzing inputs and comparing the output of `my_framework` against the inputs/reference.

If tests pass, your extension is fully integrated!

