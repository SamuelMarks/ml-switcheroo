"""
Keras (v3) Framework Adapter.

This module provides the adapter for Keras 3+, enabling translation between
Keras and other frameworks (JAX, Torch, TensorFlow).

Capabilities:
1.  **Math Operations**: Maps `keras.ops` functions to Array API standards.
2.  **Neural Layers**: Maps `keras.layers` to Neural Standards.
3.  **Discovery**: Implements `collect_api` to dynamically find losses,
    optimizers, and activations by inspecting the installed Keras package.
4.  **Plugins**: Wires `keras_sequential_pack` for Sequential container rewriting.
"""

import inspect
import logging
from typing import List, Tuple, Dict, Optional, Any, Set

# Conditional import for hybrid environment support
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

  Supports translation of Functional API models, Layer subclasses, and
  backend-agnostic math operations (`keras.ops`).
  """

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25  # After JAX/Torch, before TF

  def __init__(self):
    """
    Initializes the adapter.

    Detects if the 'keras' library is present in the environment.
    If present, sets mode to `InitMode.LIVE` for runtime scanning.
    If absent, sets mode to `InitMode.GHOST` and loads cached snapshots.
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
    """
    Returns the primary example code snippet used for instant demos.
    """
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns a dictionary of Keras 3.0 idiomatic examples categorized by Tier.

    Returns:
        Dict[str, str]: Map of 'tierX_name' to Python source code strings.
    """
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

    # --- Discovery Configuration ---

  @property
  def search_modules(self) -> List[str]:
    """
    Returns the list of module paths to scan during bulk scaffolding.
    """
    return ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Returns the default import structure for generated code.
    Format: (module_path, alias) -> `import keras as keras`.
    """
    return ("keras", "keras")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Returns regex patterns used to categorize found APIs into Semantic Tiers.
    """
    return {
      "neural": [r"\\.layers\\.", r"Layer$", r"Model$"],
      "array": [r"\\.ops\\.", r"\\.math\\."],
      "extras": [r"\\.callbacks\\.", r"\\.saving\\."],
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns list of Semantic Tiers supported by this framework.
    Keras supports all major tiers including Neural Layers.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  # --- Structural Traits ---

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines configuration for class and function rewriting logic.

    Keras Traits:
    - Base Class: `keras.Layer` (or Model)
    - Forward Method: `call`
    - Constructor: Requires `super().__init__()`
    - Impurities: `fit`, `compile` (Stateful/Side-effect heavy methods)
    """
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
    """
    Returns static definitions to ensure core operations work without discovery.
    Includes argument mappings for key layers (e.g. `Linear` -> `Dense`).
    """
    return {
      "Abs": StandardMap(api="keras.ops.abs"),
      "Mean": StandardMap(api="keras.ops.mean"),
      "Linear": StandardMap(api="keras.layers.Dense", args={"out_features": "units"}),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns list of methods that set global random seeds.
    """
    return ["utils.set_random_seed"]

  # --- Ghost Protocol Implementation ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures for the requested category.

    If in GHOST mode, hydrates from cached snapshots.
    If in LIVE mode, performs runtime introspection of the Keras package.

    Args:
        category (StandardCategory): The API category to scan (Loss, Layer, etc).

    Returns:
        List[GhostRef]: A list of discovered API references.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """Hydrates GhostRefs from loaded snapshot JSON data."""
    if not self._snapshot_data:
      return []

    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """Performs live introspection of Keras submodules."""
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
    """
    Generic helper to scan a Python module for Keras objects.

    Filters based on `kind` (class vs function) and verifies Keras-specific traits
    (serialization methods) to avoid collecting utility classes.
    """
    if not module:
      return []
    block_list = block_list or set()
    found = []

    for name, obj in inspect.getmembers(module):
      if name.startswith("_"):
        continue
      if name in block_list:
        continue

      # Class Scanning Logic
      if kind == "class" and inspect.isclass(obj):
        # Keras 3 check: All standard components have config methods.
        # This explicitly filters out utility classes or mixins.
        is_keras_object = hasattr(obj, "get_config") or hasattr(obj, "from_config")
        if is_keras_object:
          ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
          found.append(ref)

      # Function Scanning Logic
      elif kind == "function" and inspect.isfunction(obj):
        ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
        found.append(ref)

    return found

  # --- Verification Support ---

  def convert(self, data: Any) -> Any:
    """
    Converts input data (NumPy/List) to a Keras tensor.
    Used by the Fuzzer for equivalence testing.
    """
    try:
      import keras

      return keras.ops.convert_to_tensor(data)
    except (ImportError, AttributeError):
      return data

  # --- IO Serialization ---

  def get_serialization_imports(self) -> List[str]:
    """Returns import statements required for serialization."""
    return ["import keras"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Returns Python code for saving/loading models."""
    if op == "save" and object_arg:
      return f"{object_arg}.save({file_arg})"
    elif op == "load":
      return f"keras.saving.load_model({file_arg})"
    return ""

  # --- Device Management ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Returns Keras 3 device scope syntax (e.g. `keras.name_scope('gpu')`).
    Note: Keras 3 handles devices via backend config largely, but scopes allow some control.
    """
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  # --- Manual Wiring (The Last Mile) ---

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Injects manual wiring rules into the Snapshot.

    This handles cases where automated discovery methods cannot fully capture
    the necessary transformation logic (e.g., container packing, complex renames).

    Updates:
        - Wires `Sequential` to `keras_sequential_pack` plugin.
        - Wires Vision ops (`Resize`, `Normalize`) to Keras Preprocessing Layers.
        - Wires IO operations.

    Args:
        snapshot (Dict): The snapshot dictionary to mutate in-place.
    """
    mappings = snapshot.setdefault("mappings", {})

    # Sequential: Requires packing args into list via plugin
    mappings["Sequential"] = {"api": "keras.Sequential", "requires_plugin": "keras_sequential_pack"}

    # Wiring for Tier C Vision Operations
    mappings["Resize"] = {"api": "keras.layers.Resizing", "args": {"size": "height"}}
    mappings["CenterCrop"] = {"api": "keras.layers.CenterCrop", "args": {"size": "height"}}
    mappings["Normalize"] = {"api": "keras.layers.Normalization", "args": {"std": "variance", "mean": "mean"}}
    mappings["RandomCrop"] = {"api": "keras.layers.RandomCrop"}
    mappings["RandomHorizontalFlip"] = {"api": "keras.layers.RandomFlip", "args": {"p": "mode"}}
    mappings["RandomVerticalFlip"] = {"api": "keras.layers.RandomFlip"}
    mappings["Pad"] = {"api": "keras.layers.ZeroPadding2D"}
    mappings["ToTensor"] = {"api": "keras.ops.convert_to_tensor"}
    mappings["Grayscale"] = {"api": "lambda x: x", "transformation_type": "inline_lambda"}
