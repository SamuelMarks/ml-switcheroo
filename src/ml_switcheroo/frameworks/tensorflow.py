"""
TensorFlow Framework Adapter.

This module provides the adapter for TensorFlow, supporting both core TF operations
and legacy Keras integration if present within the TF namespace.

Capabilities:
1.  **Core Math**: Maps `Mean`, `Abs` to `tf.math`.
2.  **Plugin Wiring**: Handles `transpose` (permute) via the `pack_varargs` plugin.
3.  **Discovery**: Scans `tensorflow.nn` for activations and layers.
"""

import sys
from typing import List, Tuple, Optional, Dict
from ml_switcheroo.core.ghost import GhostRef, GhostInspector
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import register_framework, StructuralTraits, StandardCategory, StandardMap

try:
  import tensorflow as tf
except ImportError:
  pass


@register_framework("tensorflow")
class TensorFlowAdapter:
  """
  Adapter for TensorFlow (Core & Keras).

  Supports translation of Low-level TF modules and math operations.
  """

  display_name: str = "TensorFlow"
  inherits_from: Optional[str] = None
  ui_priority: int = 30

  def __init__(self):
    pass

  @property
  def search_modules(self) -> List[str]:
    """
    Modules to scan during Scaffolding.
    Broadens search to include signal, linalg, and embedded keras layers.
    """
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
    """Default import alias: import tensorflow as tf."""
    return ("tensorflow", "tf")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns for categorizing discovered APIs."""
    return {
      "neural": [r"\\.keras\\.", r"Layer$"],
      "extras": [r"\\.io\\.", r"\\.data\\."],
    }

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines structural rewriting rules for TF Classes.
    Often overlaps with Keras traits (call vs forward).
    """
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      strip_magic_args=["rngs"],
    )

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for core operations.

    Ensures 'Mean' uses 'reduce_mean' and 'permute_dims' uses the plugin.
    """
    return {
      "Mean": StandardMap(api="tf.math.reduce_mean"),
      "permute_dims": StandardMap(api="tf.transpose", requires_plugin="pack_varargs"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["set_seed", "random.set_seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Implements runtime introspection for the Consensus Engine.
    Scans `tf.nn` for activations/math and `tf.keras` for layers.
    """
    results = []
    try:
      import tensorflow as tf

      if category == StandardCategory.ACTIVATION:
        # Scan tf.nn for common activation functions
        # Note: `tf.nn` includes many ops, we filter by known activation names
        # to avoid pollution, or scan check signature.
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
        # Scan tf.keras.layers if available (TF < 2.16 might have embedded Keras)
        if hasattr(tf, "keras") and hasattr(tf.keras, "layers"):
          import inspect

          for name, obj in inspect.getmembers(tf.keras.layers):
            if inspect.isclass(obj) and "Layer" in name and not name.startswith("_"):
              results.append(GhostInspector.inspect(obj, f"tf.keras.layers.{name}"))

    except ImportError:
      pass
    return results

  # --- Verification ---

  @classmethod
  def get_example_code(cls) -> str:
    """Returns snippet for the web demo."""
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns TensorFlow Core idiomatic examples.

    Tiers:
    1. Math: Uses `tf.math` operations.
    2. Neural: Uses `tf.Module` (distinct from Keras Layers) to show low-level state.
    3. Extras: Uses `tf.data` pipelines.
    """
    return {
      "tier1_math": """import tensorflow as tf

def math_ops(x, y): 
    # Tier 1: Core TensorFlow Math
    # Maps to tf.math or root alias tf.* 
    a = tf.abs(x) 
    b = tf.math.add(a, y) 

    # Reduction
    return tf.math.reduce_mean(b) 
""",
      "tier2_neural": """import tensorflow as tf

class Model(tf.Module): 
    # Tier 2: Low-level TF Module (Not Keras) 
    # Demonstrates manual variable management
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
    # Efficient asynchronous data loading mechanism
    dataset = tf.data.Dataset.from_tensor_slices(tensors) 

    # Shuffle and Batch
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size) 

    return dataset
""",
    }

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Generates `tf.device` string.
    Resolves 'cuda'/'gpu' to 'GPU', 'cpu' to 'CPU'.
    """
    clean_type = device_type.strip("'\"").lower()
    tf_type = "CPU"
    if clean_type in ("cuda", "gpu", "mps"):
      tf_type = "GPU"

    idx_str = "0"
    if device_index:
      if device_index.isdigit():
        idx_str = device_index
      else:
        # Dynamic index handling
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

  def convert(self, data):
    """Converts inputs to TF Tensors."""
    try:
      import tensorflow as tf

      return tf.convert_to_tensor(data)
    except (ImportError, ValueError, TypeError, Exception):
      return data
