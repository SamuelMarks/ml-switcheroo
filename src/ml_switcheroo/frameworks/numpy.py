import numpy as np
from typing import List, Tuple, Optional, Dict
from .base import register_framework, StructuralTraits


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
    return {"extras": [r"\.random\.", r"save", r"load"]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits()

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["seed"]

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

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import numpy as np"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return var_name

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return var_name

  @classmethod
  def get_example_code(cls) -> str:
    return "import numpy as np\n\nval = np.abs([-1, 2])"
