"""
Keras (v3) Framework Adapter.

This module provides the adapter for Keras 3+, enabling translation between
Keras and other frameworks (JAX, Torch, TensorFlow).

Capabilities:
1.  **Math Operations**: Maps `keras.ops` functions to Array API standards.
2.  **Neural Layers**: Maps `keras.layers` to Neural Standards.
3.  **Discovery**: Implements `collect_api` to dynamically find losses,
    optimizers, and activations by inspecting the installed Keras package.
"""

import inspect
import sys
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
    """Returns the primary example code used for instant demos."""
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns Keras 3.0 idiomatic examples.

    Tiers:
    1. Math: Uses `keras.ops` for backend-agnostic tensor math.
    2. Neural: Uses the Functional API (`Input` -> `Model`).
    3. Extras: Uses `SeedGenerator` for controlled randomness.
    """
    return {
      "tier1_math": """import keras
from keras import ops

def math_ops(x, y): 
    # Tier 1: Using keras.ops for backend-agnostic math
    # These map effectively to jax.numpy, torch, or tf via Keras 3 logic
    a = ops.abs(x) 
    b = ops.add(a, y) 
    return ops.mean(b) 
""",
      "tier2_neural": """import keras
from keras import layers

def build_model(input_shape): 
    # Tier 2: Functional Model API
    # Maps input shape -> Conv -> Flatten -> Dense -> Model
    inputs = keras.Input(shape=input_shape) 
    x = layers.Conv2D(32, 3, activation="relu")(inputs) 
    x = layers.Flatten()(x) 
    outputs = layers.Dense(10)(x) 
    
    return keras.Model(inputs, outputs) 
""",
      "tier3_extras": """import keras
from keras import random

def generate_noise(shape): 
    # Tier 3: Framework Extras (Random) 
    # Keras 3 uses explicitly explicit SeedGenerators for stateful RNG behavior
    # across stateless backends like JAX. 
    seed_gen = random.SeedGenerator(42) 
    return random.normal(shape, seed=seed_gen) 
""",
    }

  # --- Discovery Configuration ---
  @property
  def search_modules(self) -> List[str]:
    """Modules to scan when running bulk scaffolding."""
    return ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Returns default alias tuple: (module, alias)."""
    return ("keras", "keras")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns to categorize discovered APIs."""
    return {
      "neural": [r"\\.layers\\.", r"Layer$", r"Model$"],
      "array": [r"\\.ops\\.", r"\\.math\\."],
      "extras": [r"\\.callbacks\\.", r"\\.saving\\."],
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Keras supports all major tiers including Neural Layers."""
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  # --- Structural Traits ---
  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines behavior for class and function rewriting.
    Keras uses `call` for forward pass and requires `super().__init__()`.
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
    Static definitions to ensure core ops work without discovery.
    Includes argument name mapping for Dense layers (units vs out_features).
    """
    return {
      "Abs": StandardMap(api="keras.ops.abs"),
      "Mean": StandardMap(api="keras.ops.mean"),
      "Linear": StandardMap(api="keras.layers.Dense", args={"out_features": "units"}),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    """Methods that set global random seed."""
    return ["utils.set_random_seed"]

  # --- Ghost Protocol Implementation ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures (Live or Ghost).

    If Keras is installed, it scans `keras.layers`, `keras.losses`, etc.
    If not installed, it attempts to load from a cached snapshot.

    Args:
        category (StandardCategory): The API category to scan.

    Returns:
        List[GhostRef]: Found API signatures.
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
    """Scans living Keras objects using runtime introspection."""
    results = []

    if category == StandardCategory.LOSS:
      # Classes in keras.losses
      results.extend(
        self._scan_module(
          keras.losses,
          "keras.losses",
          kind="class",
          block_list={"Loss", "Container"},
        )
      )

    elif category == StandardCategory.OPTIMIZER:
      # Classes in keras.optimizers
      results.extend(
        self._scan_module(
          keras.optimizers,
          "keras.optimizers",
          kind="class",
          block_list={"Optimizer", "TFOptimizer"},
        )
      )

    elif category == StandardCategory.ACTIVATION:
      # Activations in Keras are generally functions in keras.activations
      results.extend(self._scan_module(keras.activations, "keras.activations", kind="function"))

    elif category == StandardCategory.LAYER:
      # Layers in keras.layers
      results.extend(self._scan_module(keras.layers, "keras.layers", kind="class", block_list={"Layer"}))

    return results

  def _scan_module(
    self,
    module: Any,
    prefix: str,
    kind: str = "class",
    block_list: Optional[Set[str]] = None,
  ) -> List[GhostRef]:
    """
    Generic scanner for Keras modules.

    Verifies that classes are valid serializable Keras objects by checking
    for `get_config` or `from_config` methods.

    Args:
        module: The python module object.
        prefix: The module path string (e.g. 'keras.layers').
        kind: 'class' or 'function'.
        block_list: Set of names to ignore.

    Returns:
        List[GhostRef]: Extracted API signatures.
    """
    if not module:
      return []

    block_list = block_list or set()
    found = []

    for name, obj in inspect.getmembers(module):
      # Skip private/internal members
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

      # Function Scanning Logic (Activations/Ops)
      elif kind == "function" and inspect.isfunction(obj):
        ref = GhostInspector.inspect(obj, f"{prefix}.{name}")
        found.append(ref)

    return found

  # --- Test Harness Syntax ---

  def convert(self, data: Any) -> Any:
    """Converts input data to Keras tensor."""
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
    """Keras 3 device scopes."""
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Injects Keras specific manual wiring hooks."""
    mappings = snapshot.setdefault("mappings", {})

    # Wiring for Tier C Vision Operations
    # Maps abstract Vision Ops to Keras Preprocessing Layers
    mappings["Resize"] = {
      "api": "keras.layers.Resizing",
      "args": {"size": "height"},
    }
    mappings["CenterCrop"] = {
      "api": "keras.layers.CenterCrop",
      "args": {"size": "height"},
    }
    mappings["Normalize"] = {
      "api": "keras.layers.Normalization",
      "args": {"std": "variance", "mean": "mean"},
    }
    mappings["RandomCrop"] = {"api": "keras.layers.RandomCrop"}
    mappings["RandomHorizontalFlip"] = {
      "api": "keras.layers.RandomFlip",
      "args": {"p": "mode"},
    }
    mappings["RandomVerticalFlip"] = {"api": "keras.layers.RandomFlip"}
    mappings["Pad"] = {"api": "keras.layers.ZeroPadding2D"}
    mappings["ToTensor"] = {"api": "keras.ops.convert_to_tensor"}
    # Grayscale is often identity in model definition or handled by preprocessing pipeline
    mappings["Grayscale"] = {
      "api": "lambda x: x",
      "transformation_type": "inline_lambda",
    }
