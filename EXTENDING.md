Extending
=========

ml-switcheroo is built on a modular "Zero-Edit" architecture. You can add support for new Machine Learning frameworks (like `tinygrad`, `keras`, or `mlx`) without modifying the core engine logic. 

**Artifacts required for a new framework:**
1.  **Adapter**: Python class in `src/ml_switcheroo/frameworks/`.
2.  **Snapshot**: JSON mapping in `src/ml_switcheroo/snapshots/`.

--- 

## 1. Adding a New Framework Adapter

To support a new library (e.g., `my_framework`), create a file `src/ml_switcheroo/frameworks/my_framework.py`.

```python
from typing import List, Tuple, Dict, Any
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.schema import StructuralTraits

@register_framework("my_framework")  # Unique key used in CLI
class MyFrameworkAdapter: 
    # --- Metadata --- 
    display_name: str = "My Framework" 
    ui_priority: int = 100 

    # --- Discovery --- 
    @property
    def search_modules(self) -> List[str]: 
        return ["my_framework", "my_framework.nn"] 

    @property
    def import_alias(self) -> Tuple[str, str]: 
        # Default alias behavior: import my_framework as mf
        return ("my_framework", "mf") 

    # --- Structural Traits (Zero-Edit Rewriter Config) --- 
    # Defines how Classes and Functions are transformed. 
    @property
    def structural_traits(self) -> StructuralTraits: 
        return StructuralTraits( 
            module_base="my_framework.Module",  # Base class for layers
            forward_method="call",              # Method name: forward vs call
            requires_super_init=True,           # Needs super().__init__()?
            lifecycle_strip_methods=["gpu"],    # Remove .gpu() calls?
        ) 

    # --- Verification & Test Gen Support --- 
    @classmethod
    def get_import_stmts(cls) -> str: 
        return "import my_framework as mf" 

    @classmethod
    def get_creation_syntax(cls, var_name: str) -> str: 
        # Code to turn numpy array 'var_name' into tensor
        return f"mf.tensor({var_name})" 

    @classmethod
    def get_numpy_conversion_syntax(cls, var_name: str) -> str: 
        # Code to turn tensor back to numpy
        return f"{var_name}.numpy()" 

    # ... See base.py for full protocol including convert()
```

--- 

## 2. Mapping APIs (The Overlay)

You typically do **not** edit the core spec files (`semantics/*.json`). Instead, you define how your framework implements those standards in a Snapshot Overlay.

Create `src/ml_switcheroo/snapshots/my_framework_mappings.json`:

```json
{
  "__framework__": "my_framework",
  "mappings": {
    "Abs": { 
      "api": "my_framework.abs" 
    },
    "Add": { 
        "api": "my_framework.add" 
    },
    "Linear": { 
        "api": "my_framework.nn.Linear",
        "args": { 
            "in_features": "in_ch", 
            "out_features": "out_ch" 
        } 
    },
    "DataLoader": { 
        "api": "my_framework.data.Loader",
        "requires_plugin": "convert_dataloader"
    }
  },
  "templates": {
      "import": "import my_framework as mf",
      "convert_input": "mf.tensor({np_var})",
      "to_numpy": "{res_var}.numpy()"
  }
}
```

### Automated Discovery

Instead of writing JSON manually, you can auto-generate the file by scanning your installed library:

```bash
# 1. Install your library
pip install my_framework

# 2. Run the Scaffolder
ml_switcheroo scaffold --frameworks my_framework
```

This will find functions that match standard names (e.g. `abs`, `sum`) and populate `snapshots/my_framework_mappings.json`.

--- 

## 3. Verification

Once your Adapter and Snapshot are present, verify integration:

```bash
# 1. Run fuzzer targeting your framework
ml_switcheroo ci --target my_framework
```

This command:
1.  Reads the abstract specs (e.g. `Abs` requires numeric input).
2.  Generates a Python script using your Adapter's templates.
3.  Executes `my_framework.abs(data)` and compares the result against NumPy/Torch reference.
