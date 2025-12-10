"""
Keras (v3) Framework Adapter.

Supports Keras 3.x which unifies JAX, TensorFlow, and PyTorch backends.
It maps `keras.ops` to the Array API (Math) and `keras.layers` to Neural Ops.
"""

from typing import List, Tuple, Dict, Optional, Any
from .base import register_framework, StructuralTraits


@register_framework("keras")
class KerasAdapter:
  """Adapter for Keras 3+."""

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25  # After JAX/Torch, before TF

  # --- Discovery Configuration ---
  # Used by `ml_switcheroo sync keras` to find math/layer implementations
  @property
  def search_modules(self) -> List[str]:
    return ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("keras", "keras")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\.layers\.", r"Layer$", r"Model$"],
      "array": [r"\.ops\.", r"\.math\."],
      "extras": [r"\.callbacks\.", r"\.saving\."],
    }

  # --- Structural Traits ---
  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      init_method_name="__init__",
      inject_magic_args=[],
      lifecycle_strip_methods=[],
      impurity_methods=["fit", "compile"],
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["utils.set_random_seed"]

  # --- Test Harness Syntax ---

  def convert(self, data: Any) -> Any:
    try:
      import keras

      return keras.ops.convert_to_tensor(data)
    except (ImportError, AttributeError):
      return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import keras\nfrom keras import ops"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"ops.convert_to_tensor({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"ops.convert_to_numpy({var_name})"

  # --- IO ---

  def get_serialization_imports(self) -> List[str]:
    return ["import keras"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"{object_arg}.save({file_arg})"
    elif op == "load":
      return f"keras.saving.load_model({file_arg})"
    return ""

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    # Placeholder for context manager syntax
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  @classmethod
  def get_example_code(cls) -> str:
    return (
      "import keras\n"
      "from keras import ops\n\n"
      "class M(keras.Layer):\n"
      "    def __init__(self):\n"
      "        super().__init__()\n"
      "        self.d = keras.layers.Dense(10)\n\n"
      "    def call(self, x):\n"
      "        return self.d(x)"
    )
