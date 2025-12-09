from typing import List, Tuple, Optional, Dict
from .base import register_framework, StructuralTraits


@register_framework("tensorflow")
class TensorFlowAdapter:
  """Adapter for TensorFlow."""

  display_name: str = "TensorFlow"
  inherits_from: Optional[str] = None
  ui_priority: int = 30

  @property
  def search_modules(self) -> List[str]:
    return ["tensorflow", "tensorflow.math", "tensorflow.linalg", "tensorflow.signal"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("tensorflow", "tf")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {"neural": [r"\.keras\.", r"Layer$"], "extras": [r"\.io\.", r"\.data\."]}

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="keras.Layer", forward_method="call", requires_super_init=True, strip_magic_args=["rngs"]
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["set_seed", "random.set_seed"]

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    clean_type = device_type.strip("'\"").lower()
    tf_type = "CPU"
    if clean_type in ("cuda", "gpu", "mps"):
      tf_type = "GPU"

    idx_str = "0"
    if device_index:
      if device_index.isdigit():
        idx_str = device_index
      else:
        return f"tf.device(f'{tf_type}:{{str({device_index})}}')"

    return f"tf.device('{tf_type}:{idx_str}')"

  def get_serialization_imports(self) -> List[str]:
    return ["import tensorflow as tf"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"tf.io.write_file({file_arg}, {object_arg})"
    elif op == "load":
      return f"tf.io.read_file({file_arg})"
    return ""

    # --- Verification ---

  def convert(self, data):
    try:
      import tensorflow as tf

      return tf.convert_to_tensor(data)
    except (ImportError, ValueError, TypeError, Exception):
      return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import tensorflow as tf"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"tf.convert_to_tensor({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"{var_name}.numpy()"

  @classmethod
  def get_example_code(cls) -> str:
    return "import tensorflow as tf\n\nval = tf.abs(tf.constant([-1.0, 2.0]))"
