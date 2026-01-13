"""
Keras (v3) Framework Adapter.

This module implements the adapter for the Keras 3 framework, supporting
multi-backend translation (JAX/Torch/TensorFlow).

It handles:
1.  **Math**: Mapping `keras.ops.*` (backend-agnostic math).
2.  **Layers**: Mapping `keras.layers.*`.
3.  **Discovery**: Runtime introspection of the Keras API surface.
4.  **Ghost Mode**: Silent fallback when Keras is not installed.
5.  **Weight Migration**: Loading/saving .h5 or .keras files via h5py.
"""

import inspect
import logging
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

try:
  import keras
  import keras.activations
  import keras.layers
  import keras.losses
  import keras.ops
  import keras.optimizers
  import keras.random
except ImportError:
  keras = None

from ml_switcheroo.core.ghost import GhostInspector, GhostRef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
  ImportConfig,
  InitMode,
  StandardCategory,
  StandardMap,
  StructuralTraits,
  load_snapshot_for_adapter,
  register_framework,
)
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("keras")
class KerasAdapter:
  """
  Adapter for Keras v3 (Multi-backend).

  Provides definitions for Keras Core Ops, Layers, and Models.
  """

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25

  def __init__(self) -> None:
    """
    Initializes the adapter.

    Detects if Keras is installed. If not, attempts to load a static snapshot
    for Ghost Mode operation. Logs at DEBUG level if missing to avoid CLI noise.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

    if keras is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("keras")
      if not self._snapshot_data:
        # Downgraded from WARNING to DEBUG to prevent CLI spam on bootstrap
        logging.debug("Keras not installed and no snapshot found. Adapter disabled.")

  @property
  def search_modules(self) -> List[str]:
    """
    Returns list of search modules.

    If in Ghost Mode, returns empty list to prevent Scaffolder
    from attempting to import missing modules.

    Returns:
        List[str]: Module names.
    """
    if self._mode == InitMode.GHOST:
      return []
    return ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Submodules to exclude from recursive scans.

    Returns:
        Set[str]: Empty set for Keras properly scoped imports.
    """
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Default import alias.

    Returns:
        Tuple[str, str]: ("keras", "keras").
    """
    return ("keras", "keras")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Namespace mapping for import fixer.

    Returns:
        Dict[str, ImportConfig]: Configuration for import injection.
    """
    return {
      "keras": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="keras"),
      "keras.ops": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="ops"),
      "keras.layers": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="layers"),
      "numpy": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="np"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns for scaffold categorization.

    Returns:
        Dict[str, List[str]]: Patterns mapping tiers to paths.
    """
    return {
      "neural": [r"\\.layers\\.", r"Layer$", r"Model$"],
      "array": [r"\\.ops\\.", r"\\.math\\."],
      "extras": [r"\\.callbacks\\.", r"\\.saving\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Templates for test code generation.

    Returns:
        Dict[str, str]: Test Harness values.
    """
    return {
      "import": "import keras\nfrom keras import ops",
      "convert_input": "keras.ops.convert_to_tensor({np_var})",
      "to_numpy": "keras.ops.convert_to_numpy({res_var})",
    }

  @property
  def harness_imports(self) -> List[str]:
    """
    Imports for verification harness.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Init code for verification harness.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns code to convert Keras tensors to NumPy.

    Returns:
        str: Conversion logic checking for `numpy` property.
    """
    return "if hasattr(obj, 'numpy'): return obj.numpy()"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Supported semantic tiers.

    Returns:
        List[SemanticTier]: [ARRAY_API, NEURAL, EXTRAS].
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """
    List of framework-specific magic arguments.

    Returns:
        List[str]: Empty list as Keras handles state implicitly.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Structural transformation rules.

    Returns:
        StructuralTraits: Configuration object.
    """
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      init_method_name="__init__",
      inject_magic_args=[],
      auto_strip_magic_args=True,
      lifecycle_strip_methods=[],
      impurity_methods=["fit", "compile"],
    )

  @property
  def plugin_traits(self) -> Any:
    """
    Plugin behavior flags.

    Returns:
        PluginTraits: Object defining capabilities.
    """
    from ml_switcheroo.semantics.schema import PluginTraits

    return PluginTraits(
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Static mappings for Keras.
    Loaded dynamically from `frameworks/definitions/keras.json`.

    Returns:
        Dict[str, StandardMap]: Definitions.
    """
    defs = load_definitions("keras")
    if "ReLU" not in defs:
      defs["ReLU"] = StandardMap(api="keras.layers.ReLU")
    return defs

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Global seed methods.

    Returns:
        List[str]: Names like "set_random_seed".
    """
    return ["utils.set_random_seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API definitions via introspection or snapshot.

    Args:
        category (StandardCategory): Category to scan.

    Returns:
        List[GhostRef]: Found items.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """
    Loads from snapshot data.

    Args:
        category (StandardCategory): Category to retrieve.

    Returns:
        List[GhostRef]: Hydrated references.
    """
    if not self._snapshot_data:
      return []

    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """
    Scans live modules.

    Args:
        category (StandardCategory): Category to scan.

    Returns:
        List[GhostRef]: Found items.
    """
    results = []
    if category == StandardCategory.LOSS:
      results.extend(self._scan_module(keras.losses, "keras.losses", kind="class", block_list={"Loss", "Container"}))
    elif category == StandardCategory.OPTIMIZER:
      results.extend(
        self._scan_module(
          keras.optimizers,
          "keras.optimizers",
          kind="class",
          block_list={"Optimizer", "TFOptimizer"},
        )
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
    Reflectively scans a Keras module.

    Args:
        module (Any): The module object.
        prefix (str): Prefix for API path.
        kind (str): Expected kind ("class" or "function").
        block_list (Optional[Set[str]]): Names to exclude.

    Returns:
        List[GhostRef]: Found items.
    """
    if not module:
      return []
    block_list = block_list or set()
    found = []

    try:
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
    except Exception:
      pass

    return found

  def convert(self, data: Any) -> Any:
    """
    Converts input data to Keras Tensor.

    Args:
        data (Any): Input data.

    Returns:
        Any: Keras Tensor or original data.
    """
    try:
      import keras

      return keras.ops.convert_to_tensor(data)
    except (ImportError, AttributeError):
      return data

  def get_serialization_imports(self) -> List[str]:
    """
    Imports for saving/loading.

    Returns:
        List[str]: Imports.
    """
    return ["import keras"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Syntax for saving/loading models.

    Args:
        op (str): 'save' or 'load'.
        file_arg (str): Path string.
        object_arg (Optional[str]): Object name.

    Returns:
        str: Generated python code.
    """
    if op == "save" and object_arg:
      return f"{object_arg}.save({file_arg})"
    elif op == "load":
      return f"keras.saving.load_model({file_arg})"
    return ""

  # --- Weight Handling Logic ---

  def get_weight_conversion_imports(self) -> List[str]:
    """
    Returns imports required for the generated weight migration script.

    Returns:
        List[str]: List of import statements.
    """
    return ["import keras", "import numpy as np", "import h5py"]

  def get_weight_load_code(self, path_var: str) -> str:
    """
    Returns python code to load a checkpoint.
    Stub implemented as Keras models contain structure + weights, making raw dict handling tricky.
    """
    return textwrap.dedent(
      f""" 
            try: 
                # Keras weights are usually saved with .weights.h5 or as full model
                # This stub attempts to load if it's a full model file, extracting weights
                model = keras.models.load_model({path_var}, compile=False) 
                raw_state = {{w.name: w.numpy() for w in model.weights}} 
            except Exception as e: 
                print(f"Warning: Failed to load Keras model ({{e}}). Assuming raw h5 weights file.") 
                # Fallback to h5py if available for raw weights
                try: 
                    import h5py
                    f = h5py.File({path_var}, 'r') 
                    raw_state = {{}} 
                    def visit_func(name, node): 
                        if isinstance(node, h5py.Dataset): 
                            raw_state[name] = node[()] 
                    f.visititems(visit_func) 
                except ImportError: 
                    print("h5py not installed, cannot load raw weights.") 
                    raw_state = {{}} 
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """
    Returns python expression string that converts `tensor_var` from Keras tensor to numpy array.
    """
    return f"{tensor_var}.numpy() if hasattr({tensor_var}, 'numpy') else np.array({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """
    Returns Python code to save the dictionary `state_var` (mapping flat keys to numpy arrays)
    to `path_var` using h5py.

    Args:
        state_var (str): Variable name of the state dictionary.
        path_var (str): Variable name of the output path.

    Returns:
        str: Generated Python code block.
    """
    return textwrap.dedent(
      f""" 
            print(f"Saving generic HDF5 weights to {{ {path_var} }} using h5py...") 
            with h5py.File({path_var}, "w") as f: 
                for key, val in {state_var}.items(): 
                    # Save flat keys as datasets
                    # We use '/' replacement to create groups if key implies hierarchy, 
                    # or just flat keys if preferred. 
                    # Keras variable names usually allowed in HDF5 keys. 
                    f.create_dataset(str(key), data=val) 
            print("Done.") 
            """
    )

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Syntax for device scoping.

    Args:
        device_type (str): 'cuda', 'cpu'.
        device_index (Optional[str]): Index.

    Returns:
        str: Generated code.
    """
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  def get_device_check_syntax(self) -> str:
    """
    Syntax for checking GPU availability.

    Returns:
        str: Logic expression.
    """
    return "len(keras.config.list_logical_devices('GPU')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Keras handles RNG state internally.

    Returns:
        str: "pass".
    """
    return "pass"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Applies configuration wiring.

    Args:
        snapshot: Snapshot dict.
    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Provides a search URL for Keras documentation as direct API mapping is non-trivial.

    Args:
        api_name (str): API path.

    Returns:
        Optional[str]: URL.
    """
    return f"https://keras.io/search.html?q={api_name}"

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns tiered examples.

    Returns:
        Dict[str, str]: Map of tiers to examples.
    """
    return {
      "tier1_math": """import keras\nfrom keras import ops\n\ndef math_ops(x, y):\n  # Tier 1: Using keras.ops for backend-agnostic math\n  a = ops.abs(x)\n  b = ops.add(a, y)\n  return ops.mean(b)\n""",
      "tier2_neural": 'import keras\nfrom keras import layers\n\ndef build_model(input_shape):\n  inputs = keras.Input(shape=input_shape)\n  x = layers.Conv2D(32, 3, activation="relu")(inputs)\n  x = layers.Flatten()(x)\n  outputs = layers.Dense(10)(x)\n  return keras.Model(inputs, outputs)\n',
      "tier3_extras": "import keras\nfrom keras import random\n\ndef generate_noise(shape):\n  seed_gen = random.SeedGenerator(42)\n  return random.normal(shape, seed=seed_gen)\n",
    }
