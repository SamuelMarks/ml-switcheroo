"""
TensorFlow Framework Adapter.

This module implements the adapter for TensorFlow (Core & Keras), providing
mappings for math operations, neural layers, and IO.
"""

import sys
from typing import List, Tuple, Optional, Dict, Any, Set
from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardCategory,
  StandardMap,
  ImportConfig,
)

try:
  import tensorflow as tf
except ImportError:
  pass


@register_framework("tensorflow")
class TensorFlowAdapter:
  """
  Adapter for TensorFlow (Core & Keras).
  """

  display_name: str = "TensorFlow"
  inherits_from: Optional[str] = None
  ui_priority: int = 30

  def __init__(self) -> None:
    pass

  # --- Metadata ---

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Prevent scanning of C-Extensions."""
    return {
      "pywrap_tensorflow",
      "python",
      "core",
      "compiler",
      "contrib",
      "examples",
      "tools",
    }

  @property
  def search_modules(self) -> List[str]:
    return [
      "tensorflow",
      "tensorflow.math",
      "tensorflow.linalg",
      "tensorflow.signal",
      "keras.layers",
      "keras.ops",
    ]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("tensorflow", "tf")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Self-declaration of namespaces."""
    return {
      "tensorflow": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="tf"),
      "tensorflow.data": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="tf.data"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\\.keras\\.", r"Layer$"],
      "extras": [r"\\.io\\.", r"\\.data\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    return {
      "import": "import tensorflow as tf",
      "convert_input": "tf.convert_to_tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    }

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      strip_magic_args=["rngs"],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    return PluginTraits(
      has_numpy_compatible_arrays=True,  # Supports .astype via TF/Keras ops
      requires_explicit_rng=False,  # TF handles RNG statefully (usually)
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Static definitions for TensorFlow."""
    return {
      "Abs": StandardMap(api="tf.abs"),
      "Mean": StandardMap(api="tf.math.reduce_mean"),
      "Sum": StandardMap(api="tf.math.reduce_sum"),
      "exp": StandardMap(api="tf.math.exp"),
      "log": StandardMap(api="tf.math.log"),
      "square": StandardMap(api="tf.math.square"),
      "Add": StandardMap(api="tf.math.add"),
      "Sub": StandardMap(api="tf.math.subtract"),
      "Mul": StandardMap(api="tf.math.multiply"),
      "Div": StandardMap(api="tf.math.divide"),
      # Explicitly define pack_to_tuple as 'perm' for TF logic
      "permute_dims": StandardMap(api="tf.transpose", pack_to_tuple="perm"),
      "randn": StandardMap(api="tf.random.normal"),
      "ArgMax": StandardMap(api="tf.math.argmax"),
      "ArgMin": StandardMap(api="tf.math.argmin"),
      "DataLoader": StandardMap(api="tf.data.Dataset", requires_plugin="tf_data_loader"),
      # Types
      "Float32": StandardMap(api="tf.float32"),
      "Float64": StandardMap(api="tf.float64"),
      "Float16": StandardMap(api="tf.float16"),
      "Int64": StandardMap(api="tf.int64"),
      "Int32": StandardMap(api="tf.int32"),
      "Int16": StandardMap(api="tf.int16"),
      "UInt8": StandardMap(api="tf.uint8"),
      "Bool": StandardMap(api="tf.bool"),
      # Casting
      "CastFloat": StandardMap(api="tf.cast", inject_args={"dtype": "tf.float32"}),
      "CastDouble": StandardMap(api="tf.cast", inject_args={"dtype": "tf.float64"}),
      "CastHalf": StandardMap(api="tf.cast", inject_args={"dtype": "tf.float16"}),
      "CastLong": StandardMap(api="tf.cast", inject_args={"dtype": "tf.int64"}),
      "CastInt": StandardMap(api="tf.cast", inject_args={"dtype": "tf.int32"}),
      "CastShort": StandardMap(api="tf.cast", inject_args={"dtype": "tf.int16"}),
      "CastByte": StandardMap(api="tf.cast", inject_args={"dtype": "tf.uint8"}),
      "CastBool": StandardMap(api="tf.cast", inject_args={"dtype": "tf.bool"}),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["set_seed", "random.set_seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    results = []
    try:
      import tensorflow as tf

      if category == StandardCategory.ACTIVATION:
        target_names = {
          "relu",
          "sigmoid",
          "tanh",
          "softmax",
          "leaky_relu",
          "elu",
          "selu",
        }
        for name in target_names:
          if hasattr(tf.nn, name):
            results.append(GhostInspector.inspect(getattr(tf.nn, name), f"tf.nn.{name}"))

      elif category == StandardCategory.LAYER:
        is_keras_available = hasattr(tf, "keras") and hasattr(tf.keras, "layers")
        if is_keras_available:
          import inspect

          for name, obj in inspect.getmembers(tf.keras.layers):
            if inspect.isclass(obj) and "Layer" in name and not name.startswith("_"):
              results.append(GhostInspector.inspect(obj, f"tf.keras.layers.{name}"))

    except ImportError:
      pass
    return results

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": """import tensorflow as tf

def math_ops(x, y): 
    # Tier 1: Core TensorFlow Math
    a = tf.abs(x) 
    b = tf.math.add(a, y) 
    return tf.math.reduce_mean(b) 
""",
      "tier2_neural": """import tensorflow as tf

class Model(tf.Module): 
    # Tier 2: Low-level TF Module (Not Keras) 
    def __init__(self, in_features, out_features): 
        super().__init__() 
        self.w = tf.Variable(tf.random.normal([in_features, out_features])) 
        self.b = tf.Variable(tf.zeros([out_features])) 

    def __call__(self, x): 
        return tf.matmul(x, self.w) + self.b
""",
      "tier3_extras": """import tensorflow as tf

def data_pipeline(tensors, batch_size=32): 
    # Tier 3: tf.data Input Pipeline
    dataset = tf.data.Dataset.from_tensor_slices(tensors) 
    loader = dataset.shuffle(1024).batch(batch_size) 
    return loader
""",
    }

  # --- Syntax Generators ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generates TF device context scope string."""
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

  def get_device_check_syntax(self) -> str:
    """Return syntax to check if GPU is available."""
    return "len(tf.config.list_physical_devices('GPU')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns no-op (pass) as TF uses internal state or generator objects
    not needing explicit splitting in the JAX style.
    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    return ["import tensorflow as tf"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    if op == "save" and object_arg:
      return f"tf.io.write_file({file_arg}, {object_arg})"
    elif op == "load":
      return f"tf.io.read_file({file_arg})"
    return ""

  def convert(self, data: Any) -> Any:
    try:
      import tensorflow as tf

      return tf.convert_to_tensor(data)
    except (ImportError, ValueError, TypeError, Exception):
      return data
