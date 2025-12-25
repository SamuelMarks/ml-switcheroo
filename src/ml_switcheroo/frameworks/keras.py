"""
Keras (v3) Framework Adapter.

This module provides the adapter for Keras 3+, enabling translation between
Keras and other frameworks (JAX, Torch, TensorFlow).

It implements:
1.  **Layers**: Mapping `keras.layers` (e.g. Dense, Conv2D).
2.  **Ops**: Mapping `keras.ops` (backend-agnostic math).
3.  **String Types**: Mapping casting operations to string dtypes (`dtype="float32"`).
4.  **Discovery**: Runtime introspection of Keras modules.
"""

import inspect
import logging
from typing import List, Tuple, Dict, Optional, Any, Set

try:
  import keras
  import keras.ops
  import keras.layers
  import keras.losses
  import keras.optimizers
  import keras.activations
  import keras.random
except ImportError:
  keras = None

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  StandardCategory,
  StandardMap,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector
from ml_switcheroo.enums import SemanticTier


@register_framework("keras")
class KerasAdapter:
  """
  Adapter for Keras 3+ (Multi-backend).
  """

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25  # After JAX/Torch, before TF

  def __init__(self):
    """
    Initializes the adapter.
    Detects installation status to toggle interaction mode.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if keras is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("keras")
      if not self._snapshot_data:
        logging.warning("Keras not installed and no snapshot found. Scanning unavailable.")

  # --- Discovery Configuration ---

  @property
  def search_modules(self) -> List[str]:
    """Modules to scan during discovery."""
    return ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Canonical import alias."""
    return ("keras", "keras")

  @property
  def import_namespaces(self) -> Dict[str, Dict[str, str]]:
    """
    Namespace remapping rules.
    Includes numpy alias to ensure `numpy.float32` -> `np.float32`.
    """
    return {
      "keras": {"root": "keras", "sub": None, "alias": "keras"},
      "numpy": {"root": "numpy", "sub": None, "alias": "np"},
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns for categorizing APIs."""
    return {
      "neural": [r"\\.layers\\.", r"Layer$", r"Model$"],
      "array": [r"\\.ops\\.", r"\\.math\\."],
      "extras": [r"\\.callbacks\\.", r"\\.saving\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """Templates for generating tests."""
    return {
      "import": "import keras\nfrom keras import ops",
      "convert_input": "keras.ops.convert_to_tensor({np_var})",
      "to_numpy": "keras.ops.convert_to_numpy({res_var})",
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Supported capability tiers."""
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  # --- Structural Traits ---

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines Keras structural patterns.
    """
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      init_method_name="__init__",
      inject_magic_args=[],
      strip_magic_args=["rngs"],
      lifecycle_strip_methods=[],
      impurity_methods=["fit", "compile"],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static definitions for Keras mappings.

    Implements **Casting Strategy**:
    Unlike JAX/NumPy (objects) or Torch (attributes), Keras APIs expect string dtypes
    or backend-agnostic types (which often default to strings in the high-level API).
    Standalone types map to NumPy types for compatibility, but Cast operations
    inject string literals.
    """
    return {
      # --- Math ---
      "Abs": StandardMap(api="keras.ops.abs"),
      "Mean": StandardMap(api="keras.ops.mean"),
      "Add": StandardMap(api="keras.ops.add", args={"x": "x1", "y": "x2"}),
      "Sub": StandardMap(api="keras.ops.subtract", args={"x": "x1", "y": "x2"}),
      "Mul": StandardMap(api="keras.ops.multiply", args={"x": "x1", "y": "x2"}),
      "Div": StandardMap(api="keras.ops.divide", args={"x": "x1", "y": "x2"}),
      # --- Layers ---
      "Linear": StandardMap(api="keras.layers.Dense", args={"out_features": "units"}),
      "Flatten": StandardMap(api="keras.layers.Flatten"),
      "Reshape": StandardMap(api="keras.ops.reshape", args={"shape": "newshape"}),
      "ArgMax": StandardMap(api="keras.ops.argmax", args={"dim": "axis"}),
      "ArgMin": StandardMap(api="keras.ops.argmin", args={"dim": "axis"}),
      "MultiheadAttention": StandardMap(
        api="keras.layers.MultiHeadAttention",
        args={"embed_dim": "key_dim"},
        requires_plugin="repack_attention_call",
      ),
      "Embedding": StandardMap(
        api="keras.layers.Embedding", args={"num_embeddings": "input_dim", "embedding_dim": "output_dim"}
      ),
      "Sequential": StandardMap(api="keras.Sequential", requires_plugin="keras_sequential_pack"),
      "LayerNorm": StandardMap(
        api="keras.layers.LayerNormalization", args={"eps": "epsilon", "normalized_shape": "axis"}
      ),
      "GELU": StandardMap(
        api="keras.layers.Activation",
        args={"input": "gelu"},
        transformation_type="inline_lambda",
        operator="keras.activations.gelu",
      ),
      "log_softmax": StandardMap(api="keras.activations.log_softmax"),
      # --- Vision ---
      "Resize": StandardMap(api="keras.layers.Resizing", args={"size": "height"}),
      "Normalize": StandardMap(api="keras.layers.Normalization", args={"std": "variance", "mean": "mean"}),
      "ToTensor": StandardMap(api="keras.ops.convert_to_tensor"),
      "CenterCrop": StandardMap(api="keras.layers.CenterCrop", args={"size": "height"}),
      "RandomCrop": StandardMap(api="keras.layers.RandomCrop"),
      "RandomHorizontalFlip": StandardMap(api="keras.layers.RandomFlip", args={"p": "mode"}),
      "RandomVerticalFlip": StandardMap(api="keras.layers.RandomFlip"),
      "Grayscale": StandardMap(api="lambda x: x", transformation_type="inline_lambda"),
      "Variable": StandardMap(api="keras.Variable", args={"value": "initializer"}),
      # --- Types ---
      # Mapping standalone types to NumPy ensures interoperability in Keras constants
      # required_imports helps ImportFixer know to pull in numpy
      "Float32": StandardMap(api="numpy.float32", required_imports=["import numpy"]),
      "Float64": StandardMap(api="numpy.float64", required_imports=["import numpy"]),
      "Float16": StandardMap(api="numpy.float16", required_imports=["import numpy"]),
      "Int64": StandardMap(api="numpy.int64", required_imports=["import numpy"]),
      "Int32": StandardMap(api="numpy.int32", required_imports=["import numpy"]),
      "Int16": StandardMap(api="numpy.int16", required_imports=["import numpy"]),
      "UInt8": StandardMap(api="numpy.uint8", required_imports=["import numpy"]),
      "Bool": StandardMap(api="bool"),
      # --- Casting ---
      # Functional casting using `keras.ops.cast(x, dtype='string')`
      # We inject the dtype as a string argument.
      "CastFloat": StandardMap(api="keras.ops.cast", inject_args={"dtype": "float32"}),
      "CastDouble": StandardMap(api="keras.ops.cast", inject_args={"dtype": "float64"}),
      "CastHalf": StandardMap(api="keras.ops.cast", inject_args={"dtype": "float16"}),
      "CastLong": StandardMap(api="keras.ops.cast", inject_args={"dtype": "int64"}),
      "CastInt": StandardMap(api="keras.ops.cast", inject_args={"dtype": "int32"}),
      "CastShort": StandardMap(api="keras.ops.cast", inject_args={"dtype": "int16"}),
      "CastByte": StandardMap(api="keras.ops.cast", inject_args={"dtype": "uint8"}),
      "CastBool": StandardMap(api="keras.ops.cast", inject_args={"dtype": "bool"}),
      "CastChar": StandardMap(api="keras.ops.cast", inject_args={"dtype": "int8"}),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    """Global seed methods."""
    return ["utils.set_random_seed"]

  # --- Ghost Protocol Implementation ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Dynamically collects API for Consensus.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []

    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    results = []

    if category == StandardCategory.LOSS:
      results.extend(self._scan_module(keras.losses, "keras.losses", kind="class", block_list={"Loss", "Container"}))
    elif category == StandardCategory.OPTIMIZER:
      results.extend(
        self._scan_module(keras.optimizers, "keras.optimizers", kind="class", block_list={"Optimizer", "TFOptimizer"})
      )
    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_module(keras.activations, "keras.activations", kind="function"))
    elif category == StandardCategory.LAYER:
      results.extend(self._scan_module(keras.layers, "keras.layers", kind="class", block_list={"Layer"}))

    return results

  def _scan_module(
    self, module: Any, prefix: str, kind: str = "class", block_list: Optional[Set[str]] = None
  ) -> List[GhostRef]:
    """Helper to scan Keras modules."""
    if not module:
      return []
    block_list = block_list or set()
    found = []

    for name, obj in inspect.getmembers(module):
      if name.startswith("_"):
        continue
      if name in block_list:
        continue

      if kind == "class" and inspect.isclass(obj):
        is_keras_object = hasattr(obj, "get_config") or hasattr(obj, "from_config")
        if is_keras_object:
          ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
          found.append(ref)

      elif kind == "function" and inspect.isfunction(obj):
        ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
        found.append(ref)

    return found

  # --- Verification Support ---

  def convert(self, data: Any) -> Any:
    """Converts input object to Keras Tensor."""
    try:
      import keras

      return keras.ops.convert_to_tensor(data)
    except (ImportError, AttributeError):
      return data

  # --- IO Serialization ---

  def get_serialization_imports(self) -> List[str]:
    """Imports for save/load."""
    return ["import keras"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Syntax for save/load."""
    if op == "save" and object_arg:
      return f"{object_arg}.save({file_arg})"
    elif op == "load":
      return f"keras.saving.load_model({file_arg})"
    return ""

  # --- Device Management ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generates device scope syntax."""
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  # --- Manual Wiring ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies manual wiring not covered by definitions."""
    pass

  @classmethod
  def get_example_code(cls) -> str:
    """Returns standard example."""
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns Keras examples."""
    return {
      "tier1_math": """import keras
from keras import ops

def math_ops(x, y):
  # Tier 1: Using keras.ops for backend-agnostic math
  a = ops.abs(x)
  b = ops.add(a, y)
  return ops.mean(b)
""",
      "tier2_neural": """import keras
from keras import layers

def build_model(input_shape):
  # Tier 2: Functional Model API
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(32, 3, activation="relu")(inputs)
  x = layers.Flatten()(x)
  outputs = layers.Dense(10)(x)
  return keras.Model(inputs, outputs)
""",
      "tier3_extras": """import keras
from keras import random

def generate_noise(shape):
  seed_gen = random.SeedGenerator(42)
  return random.normal(shape, seed=seed_gen)
""",
    }
