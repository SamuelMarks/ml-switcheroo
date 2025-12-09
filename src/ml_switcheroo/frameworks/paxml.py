import numpy as np
from typing import List, Tuple, Optional, Dict
from .base import register_framework, StructuralTraits


@register_framework("paxml")
class PaxmlAdapter:
  """Adapter for Google PaxML (Praxis)."""

  display_name: str = "PaxML / Praxis"
  inherits_from: str = "jax"
  ui_priority: int = 60

  @property
  def search_modules(self) -> List[str]:
    return ["praxis", "praxis.layers", "praxis.base_layer"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("praxis.layers", "pl")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.praxis\."], "extras": []}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="praxis.base_layer.BaseLayer",
      forward_method="__call__",
      init_method_name="setup",
      requires_super_init=False,
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    # Use JAX logic
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

  def get_serialization_imports(self) -> List[str]:
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    # Use JAX/Orbax logic
    if op == "save" and object_arg:
      return f"orbax.checkpoint.PyTreeCheckpointer().save(directory={file_arg}, item={object_arg})"
    elif op == "load":
      return f"orbax.checkpoint.PyTreeCheckpointer().restore({file_arg})"
    return ""

    # --- Verification ---

  def convert(self, data):
    try:
      import jax.numpy as jnp
    except ImportError:
      return data
    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return jnp.array(data)
    return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import jax.numpy as jnp\nimport praxis"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"jnp.array({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"np.array({var_name})"

  @classmethod
  def get_example_code(cls) -> str:
    return (
      "import praxis.layers as pl\n"
      "from praxis import base_layer\n\n"
      "class Simple(base_layer.BaseLayer):\n"
      "    def setup(self):\n"
      "        self.l = pl.Linear(10, 10)\n\n"
      "    def __call__(self, x):\n"
      "        return self.l(x)"
    )
