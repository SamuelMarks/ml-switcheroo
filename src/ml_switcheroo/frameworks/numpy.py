"""
Numpy Framework Adapter.

This module provides the implementation definitions for the NumPy API.
It maps abstract operations for Math and Extras to ``numpy.*`` functions
and defines type mappings for data-driven casting logic.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
)


@register_framework("numpy")
class NumpyAdapter:
  """
  Adapter for generic NumPy.

  Provides support for:
  1.  **Math Tiers**: Basic array operations (abs, mean, sum).
  2.  **Type Mapping**: Abstract Dtypes to ``numpy.float32``, etc.
  3.  **IO**: Save/Load operations.
  """

  display_name: str = "NumPy"
  inherits_from: Optional[str] = None
  ui_priority: int = 20

  @property
  def search_modules(self) -> List[str]:
    """Returns list of numpy submodules to scan."""
    return ["numpy", "numpy.linalg", "numpy.fft"]

  @property
  def unsafe_submodules(self) -> Set[str]:
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Returns ('numpy', 'np')."""
    return ("numpy", "np")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Remaps imports to 'np' alias."""
    return {"numpy": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="np")}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns for IO and Randomness."""
    return {"extras": [r"\\.random\\\\.", r"save", r"load"]}

  @property
  def test_config(self) -> Dict[str, str]:
    """Test templates for NumPy."""
    return {
      "import": "import numpy as np",
      "convert_input": "{np_var}",  # Identity (NumPy is default)
      "to_numpy": "{res_var}",  # Identity
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    return []

  def get_harness_init_code(self) -> str:
    return ""

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    NumPy supports Arrays (Math) and Extras (IO).
    It does NOT support Neural layers structurally.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Returns default structural traits (no class rewriting)."""
    return StructuralTraits(
      auto_strip_magic_args=True  # NumPy doesn't support random keys or context args
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Plugin capabilities.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for NumPy mappings.
    Includes Math, Extras, Types, and Casting logic.
    """
    return {
      # --- Math / Array ---
      "Abs": StandardMap(api="np.abs"),
      "Mean": StandardMap(api="np.mean"),
      "Sum": StandardMap(api="np.sum"),
      "Add": StandardMap(api="np.add", transformation_type="infix", operator="+"),
      "Sub": StandardMap(api="np.subtract", transformation_type="infix", operator="-"),
      "Mul": StandardMap(api="np.multiply", transformation_type="infix", operator="*"),
      "Div": StandardMap(api="np.divide", transformation_type="infix", operator="/"),
      "max": StandardMap(api="np.max"),
      "min": StandardMap(api="np.min"),
      "exp": StandardMap(api="np.exp"),
      "log": StandardMap(api="np.log"),
      "square": StandardMap(api="np.square"),
      "sqrt": StandardMap(api="np.sqrt"),
      # --- Extras ---
      "randn": StandardMap(api="numpy.random.randn"),
      "permute_dims": StandardMap(api="numpy.transpose", pack_to_tuple="axes"),
      # --- Types ---
      "Float32": StandardMap(api="numpy.float32"),
      "Float64": StandardMap(api="numpy.float64"),
      "Float16": StandardMap(api="numpy.float16"),
      "Int64": StandardMap(api="numpy.int64"),
      "Int32": StandardMap(api="numpy.int32"),
      "Int16": StandardMap(api="numpy.int16"),
      "UInt8": StandardMap(api="numpy.uint8"),
      "Bool": StandardMap(api="numpy.bool_"),
      # --- Casting ---
      # Maps generic casting hooks to .astype(), relying on the plugin to
      # inject the correct type object looked up from the types above.
      "CastFloat": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastDouble": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastHalf": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastLong": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastInt": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastShort": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastByte": StandardMap(api="astype", requires_plugin="type_methods"),
      "CastBool": StandardMap(api="astype", requires_plugin="type_methods"),
      "SiLU": StandardMap(macro_template="{x} * (1 / (1 + np.exp(-{x})))", required_imports=["import numpy as np"]),
      "TensorType": StandardMap(api="numpy.ndarray"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    """Returns seed methods."""
    return ["seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """NumPy doesn't implement Layers/Losses structurally."""
    return []

  # --- Syntax Generation ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Returns CPU syntax ignoring device requests (NumPy is CPU-only)."""
    return "'cpu'"

  def get_device_check_syntax(self) -> str:
    """
    NumPy does not support GPUs.
    """
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """No-op for NumPy."""
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """Returns imports for IO."""
    return ["import numpy as np"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Returns np.save/load syntax."""
    if op == "save" and object_arg:
      return f"np.save(file={file_arg}, arr={object_arg})"
    elif op == "load":
      return f"np.load(file={file_arg})"
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No dynamic wiring needed for NumPy."""
    pass

  # --- Verification ---

  def convert(self, data: Any) -> Any:
    """Attempts to convert input data to a NumPy array."""
    if isinstance(data, (list, tuple)):
      return type(data)(self.convert(x) for x in data)
    if isinstance(data, dict):
      return {k: self.convert(v) for k, v in data.items()}

    if hasattr(data, "detach"):
      try:
        return data.detach().cpu().numpy()
      except Exception:
        pass
    if hasattr(data, "numpy"):
      try:
        return data.numpy()
      except Exception:
        pass
    if hasattr(data, "__array__"):
      try:
        return np.array(data)
      except Exception:
        pass
    return data

  @classmethod
  def get_example_code(cls) -> str:
    """Returns the primary example code used for instant demos."""
    return cls().get_tiered_examples()["tier1_math"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns NumPy idiomatic examples.
    """
    return {
      "tier1_math": """import numpy as np

def linear_algebra_ops(a, b): 
    # Tier 1: Standard Numeric Computing
    # Matrix Multiplication
    dot = np.matmul(a, b) 

    # Element-wise operations
    diff = np.abs(a - b) 

    # Aggregation
    norm = np.linalg.norm(diff) 
    return dot, norm
""",
      "tier2_neural": """import numpy as np

# Tier 2: Neural Networks (Out of Scope for NumPy) 
# NumPy does not offer a built-in neural layer API. 
# While possible to write one from scratch, it is not
# supported by the ml-switcheroo transpiler out-of-the-box. 
""",
      "tier3_extras": """import numpy as np

def serialize_data(arr, filename): 
    # Tier 3: IO Persistence
    # Use standard binary format (.npy) 
    np.save(file=filename, arr=arr) 

    # Reload
    loaded = np.load(file=filename) 
    return loaded
""",
    }
