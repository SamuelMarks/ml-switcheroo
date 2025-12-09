import numpy as np
from typing import List, Tuple, Optional, Dict
from .base import register_framework, StructuralTraits


@register_framework("jax")
class JaxAdapter:
  """Adapter for JAX / Flax."""

  display_name: str = "JAX / Flax"
  inherits_from: Optional[str] = None
  ui_priority: int = 10

  @property
  def search_modules(self) -> List[str]:
    return ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("jax.numpy", "jnp")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.linen", r"Module$"], "extras": [r"random\.", r"pmap", r"vmap", r"jit"]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="flax.nnx.Module",
      forward_method="__call__",
      inject_magic_args=[("rngs", "flax.nnx.Rngs")],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      # New: Define Args that MUST be static for JIT compatibility
      jit_static_args=["axis", "axes", "dim", "dims", "keepdim", "keepdims", "ord", "mode", "dtype"],
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    clean_type = device_type.strip("'\"").lower()
    backend = "gpu" if clean_type in ("cuda", "mps", "gpu") else clean_type

    is_literal = device_type.startswith(("'", '"'))
    type_code = f"'{backend}'" if is_literal else device_type

    idx_code = device_index if device_index is not None else "0"
    return f"jax.devices({type_code})[{idx_code}]"

  def get_serialization_imports(self) -> List[str]:
    return ["import orbax.checkpoint"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
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
    return "import jax\nimport jax.numpy as jnp"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"jnp.array({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"np.array({var_name})"

  @classmethod
  def get_example_code(cls) -> str:
    return (
      "from flax import nnx\n\n"
      "class Simple(nnx.Module):\n"
      "    def __init__(self, rngs: nnx.Rngs):\n"
      "        self.l = nnx.Linear(10, 10, rngs=rngs)\n\n"
      "    def __call__(self, x):\n"
      "        return self.l(x)"
    )
