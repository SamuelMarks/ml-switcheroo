"""
Keras (v3) Framework Adapter.

Supports Keras 3.x which unifies JAX, TensorFlow, and PyTorch backends.
It maps `keras.ops` to the Array API (Math) and `keras.layers` to Neural Ops.

Implements **Ghost Protocol** (collect_api) to support dynamic discovery
in environments where the heavy Keras library might not be installed.
"""

import inspect
import sys
import logging
from typing import List, Tuple, Dict, Optional, Any, Set

# Conditional import for hybrid environment support
try:
  import keras
  import keras.ops
  import keras.losses
  import keras.optimizers
  import keras.activations
  import keras.random
except ImportError:
  keras = None

from .base import (
  register_framework,
  StructuralTraits,
  StandardCategory,
  StandardMap,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector


@register_framework("keras")
class KerasAdapter:
  """Adapter for Keras 3+."""

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25  # After JAX/Torch, before TF

  def __init__(self):
    """
    Initializes the adapter.
    Detects if 'keras' library is present. If not, enters Ghost Mode.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if keras is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("keras")
      if not self._snapshot_data:
        logging.warning("Keras not installed and no snapshot found. Scanning unavailable.")

    @classmethod
    def get_example_code(cls) -> str:
      return cls().get_tiered_examples()["tier2_neural"]

    def get_tiered_examples(self) -> Dict[str, str]:
      return {
        "tier1_math": """import keras
from keras import ops

def math_ops(x, y):
  # Tier 1: Using keras.ops for backend-agnostic math
  a = ops.abs(x)
  b = ops.add(a, y)
  return ops.mean(b)""",
        "tier2_neural": """import keras
from keras import layers

def build_model(input_shape):
  # Tier 2: Functional Model API
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(32, 3, activation="relu")(inputs)
  x = layers.Flatten()(x)
  outputs = layers.Dense(10)(x)
  return keras.Model(inputs, outputs)""",
        "tier3_extras": """import keras
import keras.random

def logic(x):
  # Tier 3: Framework Extras (Random)
  seed = keras.random.SeedGenerator(42)
  return keras.random.normal(x.shape, seed=seed)""",
      }

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
      "neural": [r"\\.layers\\.", r"Layer$", r"Model$"],
      "array": [r"\\.ops\\.", r"\\.math\\."],
      "extras": [r"\\.callbacks\\.", r"\\.saving\\."],
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
  def definitions(self) -> Dict[str, StandardMap]:
    return {}

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["utils.set_random_seed"]

  # --- Ghost Protocol Implementation ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """Collects API signatures (Live or Ghost)."""
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []

    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """Scans living Keras objects."""
    results = []

    # Keras naming conventions (e.g. `MeanSquaredError`, `Adam`) do not strictly use suffixes.
    # We rely on structural traits (get_config) and exclude base classes explicitly.
    if category == StandardCategory.LOSS:
      results.extend(self._scan_module(keras.losses, "keras.losses", kind="class", block_list={"Loss", "Container"}))
    elif category == StandardCategory.OPTIMIZER:
      results.extend(
        self._scan_module(keras.optimizers, "keras.optimizers", kind="class", block_list={"Optimizer", "TFOptimizer"})
      )
    elif category == StandardCategory.ACTIVATION:
      # Activations in Keras are generally functions in keras.activations
      results.extend(self._scan_module(keras.activations, "keras.activations", kind="function"))

    return results

  def _scan_module(
    self, module: Any, prefix: str, kind: str = "class", block_list: Optional[Set[str]] = None
  ) -> List[GhostRef]:
    """
    Generic scanner for Keras modules.
    Verifies that classes are likely serializable Keras objects (contain `get_config`).
    """
    if not module:
      return []

    block_list = block_list or set()
    found = []

    for name, obj in inspect.getmembers(module):
      # Skip private/internal
      if name.startswith("_"):
        continue

      if name in block_list:
        continue

      # Class Scanning Logic
      if kind == "class" and inspect.isclass(obj):
        # Keras 3 check: All standard components have `get_config` or `from_config`
        # This is the most robust signal that this is a valid serializable Keras Entity.
        is_keras_object = hasattr(obj, "get_config") or hasattr(obj, "from_config")

        if is_keras_object:
          ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
          found.append(ref)

      # Function Scanning Logic (Activations)
      elif kind == "function" and inspect.isfunction(obj):
        ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
        found.append(ref)

    return found

  # --- Test Harness Syntax ---

  def convert(self, data: Any) -> Any:
    try:
      import keras

      return keras.ops.convert_to_tensor(data)
    except (ImportError, AttributeError):
      return data

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
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Injects Keras specific manual wiring hooks."""
    mappings = snapshot.setdefault("mappings", {})

    # Wiring for Tier C Vision Operations
    # Maps abstract Vision Ops to Keras Preprocessing Layers
    mappings["Resize"] = {"api": "keras.layers.Resizing", "args": {"size": "height"}}
    mappings["CenterCrop"] = {"api": "keras.layers.CenterCrop", "args": {"size": "height"}}
    mappings["Normalize"] = {"api": "keras.layers.Normalization", "args": {"std": "variance", "mean": "mean"}}
    mappings["RandomCrop"] = {"api": "keras.layers.RandomCrop"}
    mappings["RandomHorizontalFlip"] = {"api": "keras.layers.RandomFlip", "args": {"p": "mode"}}
    mappings["RandomVerticalFlip"] = {"api": "keras.layers.RandomFlip"}
    # Use ZeroPadding2D as nearest equivalent for general Pad
    mappings["Pad"] = {"api": "keras.layers.ZeroPadding2D"}
    # ToTensor logic is usually implicit in Keras or uses ops.convert_to_tensor
    mappings["ToTensor"] = {"api": "keras.ops.convert_to_tensor"}
    # Grayscale is often just a no-op identity in model definition or handled by preprocessing
    # Mapping to a lambda pass-through for architectural compliance
    mappings["Grayscale"] = {"api": "lambda x: x", "transformation_type": "inline_lambda"}
