import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import register_framework, StructuralTraits, StandardCategory


@register_framework("numpy")
class NumpyAdapter:
  """Adapter for generic NumPy."""

  display_name: str = "NumPy"
  inherits_from: Optional[str] = None
  ui_priority: int = 20

  @property
  def search_modules(self) -> List[str]:
    return ["numpy", "numpy.linalg", "numpy.fft"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("numpy", "np")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"extras": [r"\\.random\\\\.", r"save", r"load"]}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    # NumPy supports Arrays (Math) and Extras (IO).
    # It does NOT support Neural layers structurally.
    return [SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits()

  @property
  def definitions(self) -> Dict[str, Any]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    # NumPy doesn't implement Layers/Losses structurally
    return []

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

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return "'cpu'"

  def get_serialization_imports(self) -> List[str]:
    return ["import numpy as np"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"np.save(file={file_arg}, arr={object_arg})"
    elif op == "load":
      return f"np.load(file={file_arg})"
    return ""

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies wiring fixes for NumPy.
    Corrects auto-generated mappings that might incorrectly point to non-existent APIs.
    """
    mappings = snapshot.setdefault("mappings", {})

    # Fix: `permute_dims` often fuzzy-matches to non-existent `numpy.permute_dims`
    # Map it explicitly to `numpy.transpose` with the packing plugin.
    mappings["permute_dims"] = {"api": "numpy.transpose", "requires_plugin": "pack_varargs"}

  # --- Verification ---

  def convert(self, data):
    if isinstance(data, (list, tuple)):
      return type(data)(self.convert(x) for x in data)
    if isinstance(data, dict):
      return {k: self.convert(v) for k, v in data.items()}

    if hasattr(data, "detach"):
      try:
        return data.detach().cpu().numpy()
      except:
        pass
    if hasattr(data, "numpy"):
      try:
        return data.numpy()
      except:
        pass
    if hasattr(data, "__array__"):
      try:
        return np.array(data)
      except:
        pass
    return data
