"""
Numpy Framework Adapter.

This module provides the implementation definitions for the NumPy API.
It maps abstract operations for Math and Extras to ``numpy.*`` functions
and defines type mappings for data-driven casting logic.

Capabilities:
1.  **Math**: `np.abs`, `np.mean`.
2.  **IO**: `np.save`, `np.load` for persistence.
3.  **Weight Migration**: Handling dict-based `.npz` archives via `get_weight_*` hooks.
"""

import numpy as np
import textwrap
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
from ml_switcheroo.frameworks.loader import load_definitions


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
    """
    Returns list of numpy submodules to scan.

    Returns:
        List[str]: Modules.
    """
    return ["numpy", "numpy.linalg", "numpy.fft"]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns unsafe modules.

    Returns:
        Set[str]: Empty.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Returns import tuple.

    Returns:
        Tuple[str, str]: ("numpy", "np").
    """
    return ("numpy", "np")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Remaps imports to 'np' alias.

    Returns:
        Dict[str, ImportConfig]: Mappings.
    """
    return {"numpy": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="np")}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for IO and Randomness.

    Returns:
        Dict[str, List[str]]: Patterns.
    """
    return {"extras": [r"\\.random\\\\.", r"save", r"load"]}

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Test templates for NumPy.

    Returns:
        Dict[str, str]: Templates.
    """
    return {
      "import": "import numpy as np",
      "convert_input": "{np_var}",  # Identity (NumPy is default)
      "to_numpy": "{res_var}",  # Identity
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Imports for harness.

    Returns:
        List[str]: Empty.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Init code.

    Returns:
        str: Empty.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns identity code for NumPy arrays.

    Returns:
        str: Code string.
    """
    return "if isinstance(obj, np.ndarray): return obj"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    NumPy supports Arrays (Math) and Extras (IO).
    It does NOT support Neural layers structurally.

    Returns:
        List[SemanticTier]: Supported tiers.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    Returns list of magic args.

    Returns:
        List[str]: Empty.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Returns default structural traits (no class rewriting).

    Returns:
        StructuralTraits: Traits.
    """
    return StructuralTraits(
      auto_strip_magic_args=True  # NumPy doesn't support random keys or context args
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Plugin capabilities.

    Returns:
        PluginTraits: Capabilities.
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
    Loaded dynamically from `frameworks/definitions/numpy.json`.

    Returns:
        Dict[str, StandardMap]: Definitions.
    """
    return load_definitions("numpy")

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns seed methods.

    Returns:
        List[str]: Methods list.
    """
    return ["seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    NumPy doesn't implement Layers/Losses structurally.

    Args:
        category: Category.

    Returns:
        List[GhostRef]: Empty list.
    """
    return []

  # --- Syntax Generation ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns CPU syntax ignoring device requests (NumPy is CPU-only).

    Args:
        device_type: Device.
        device_index: Index.

    Returns:
        str: "'cpu'".
    """
    return "'cpu'"

  def get_device_check_syntax(self) -> str:
    """
    NumPy does not support GPUs.

    Returns:
        str: "False".
    """
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    No-op for NumPy.

    Args:
        rng_var: RNG variable.
        key_var: Key variable.

    Returns:
        str: "pass".
    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """
    Returns imports for IO.

    Returns:
        List[str]: Imports.
    """
    return ["import numpy as np"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Returns np.save/load syntax.

    Args:
        op: 'save' or 'load'.
        file_arg: Path.
        object_arg: Obj.

    Returns:
        str: Code.
    """
    if op == "save" and object_arg:
      return f"np.save(file={file_arg}, arr={object_arg})"
    elif op == "load":
      return f"np.load(file={file_arg})"
    return ""

  # --- Weight Handling ---

  def get_weight_conversion_imports(self) -> List[str]:
    return ["import numpy as np"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Loads .npz files into a dictionary."""
    return textwrap.dedent(
      f""" 
            loaded = np.load({path_var}, allow_pickle=True) 
            # If NpzFile wrapper, convert to dict
            if hasattr(loaded, 'files'): 
                raw_state = {{k: loaded[k] for k in loaded.files}} 
            elif isinstance(loaded.item(), dict): 
                # Handle 0-d array wrapping a dict (common in np.save) 
                raw_state = loaded.item() 
            else: 
                raw_state = {{'data': loaded}} 
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return f"{tensor_var}"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Saves dictionary to compressed .npz."""
    return f"np.savez_compressed({path_var}, **{state_var})"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """No dynamic wiring needed for NumPy."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Generates NumPy documentation URL.

    Args:
        api_name: API Path.

    Returns:
        Optional[str]: URL.
    """
    return f"https://numpy.org/doc/stable/reference/generated/{api_name}.html"

  # --- Verification ---

  def convert(self, data: Any) -> Any:
    """
    Attempts to convert input data to a NumPy array.

    Args:
        data (Any): Input.

    Returns:
        Any: Numpy array or original.
    """
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

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns NumPy idiomatic examples.

    Returns:
        Dict[str, str]: Examples.
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
