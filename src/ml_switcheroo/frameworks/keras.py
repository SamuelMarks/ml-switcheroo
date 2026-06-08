"""Keras (v3) Framework Adapter.

This module implements the adapter for the Keras 3 framework, supporting
multi-backend translation (JAX/Torch/TensorFlow).

It handles:
1.  **Math**: Mapping `keras.ops.*` (backend-agnostic math).
2.  **Layers**: Mapping `keras.layers.*`.
3.  **Discovery**: Runtime introspection of the Keras API surface.
4.  **Ghost Mode**: Silent fallback when Keras is not installed.
5.  **Weight Migration**: Loading/saving .h5 or .keras files via h5py.
"""

import logging
import textwrap
from typing import Any, Dict, List, Optional, Tuple

try:
  import keras
  import keras.activations  # pragma: no cover
  import keras.layers  # pragma: no cover
  import keras.losses  # pragma: no cover
  import keras.ops  # pragma: no cover
  import keras.optimizers  # pragma: no cover
  import keras.random  # pragma: no cover
except Exception:
  keras = None
from ml_switcheroo_ir.schema.ghost import GhostRef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import (
  ImportConfig,
  InitMode,
  StandardMap,
  StructuralTraits,
  load_snapshot_for_adapter,
  register_framework,
)
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("keras")
class KerasAdapter:
  """Adapter for Keras v3 (Multi-backend).

  Provides definitions for Keras Core Ops, Layers, and Models.
  """

  display_name: str = "Keras"
  inherits_from: Optional[str] = None
  ui_priority: int = 25

  def __init__(self) -> None:
    """Initializes the adapter.

    Detects if Keras is installed. If not, attempts to load a static snapshot
    for Ghost Mode operation. Logs at DEBUG level if missing to avoid CLI noise.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if keras is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("keras")
      if not self._snapshot_data:
        logging.debug("Keras not installed and no snapshot found. Adapter disabled.")  # pragma: no cover

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Default import alias.

    Returns:
        Tuple[str, str]: ("keras", "keras").

    """
    return "keras", "keras"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Namespace mapping for import fixer.

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
  def test_config(self) -> Dict[str, str]:
    """Templates for test code generation.

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
    """Imports for verification harness.

    Returns:
        List[str]: Empty list.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Init code for verification harness.

    Returns:
        str: Empty string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns code to convert Keras tensors to NumPy.

    Returns:
        str: Conversion logic checking for `numpy` property.

    """
    return "if hasattr(obj, 'numpy'): return obj.numpy()"

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Supported semantic tiers.

    Returns:
        List[SemanticTier]: [ARRAY_API, NEURAL, EXTRAS].

    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def declared_magic_args(self) -> List[str]:
    """List of framework-specific magic arguments.

    Returns:
        List[str]: Empty list as Keras handles state implicitly.

    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Structural transformation rules.

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
    """Plugin behavior flags.

    Returns:
        PluginTraits: Object defining capabilities.

    """
    from ml_switcheroo.semantics.schema import PluginTraits  # pragma: no cover

    return PluginTraits(  # pragma: no cover
      has_numpy_compatible_arrays=True,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Static mappings for Keras.
    Loaded dynamically from `frameworks/definitions/keras.json`.

    Returns:
        Dict[str, StandardMap]: Definitions.

    """
    defs = load_definitions("keras")
    if "ReLU" not in defs:
      defs["ReLU"] = StandardMap(api="keras.layers.ReLU")  # pragma: no cover
    return defs

  @property
  def rng_seed_methods(self) -> List[str]:
    """Global seed methods.

    Returns:
        List[str]: Names like "set_random_seed".

    """
    return ["utils.set_random_seed"]

  def _collect_ghost(self, category: SemanticTier) -> List[GhostRef]:
    """Loads from snapshot data.

    Args:
        category (SemanticTier): Category to retrieve.

    Returns:
        List[GhostRef]: Hydrated references.

    """
    if not self._snapshot_data:  # pragma: no cover
      return []  # pragma: no cover
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])  # pragma: no cover
    return [GhostRef.model_validate(item) for item in raw_list]  # pragma: no cover

  def _collect_live(self, category: SemanticTier) -> List[GhostRef]:
    """Scans live modules.

    Args:
        category (SemanticTier): Category to scan.

    Returns:
        List[GhostRef]: Found items.

    """
    results = []  # pragma: no cover
    if category == SemanticTier.LOSS:  # pragma: no cover
      results.extend(
        self._scan_module(keras.losses, "keras.losses", kind="class", block_list={"Loss", "Container"})
      )  # pragma: no cover
    elif category == SemanticTier.OPTIMIZER:  # pragma: no cover
      results.extend(  # pragma: no cover
        self._scan_module(keras.optimizers, "keras.optimizers", kind="class", block_list={"Optimizer", "TFOptimizer"})
      )
    elif category == SemanticTier.ACTIVATION:  # pragma: no cover
      results.extend(self._scan_module(keras.activations, "keras.activations", kind="function"))  # pragma: no cover
    elif category == SemanticTier.LAYER:  # pragma: no cover
      results.extend(
        self._scan_module(keras.layers, "keras.layers", kind="class", block_list={"Layer"})
      )  # pragma: no cover
    return results  # pragma: no cover

  def convert(self, data: Any) -> Any:
    """Converts input data to Keras Tensor.

    Args:
        data (Any): Input data.

    Returns:
        Any: Keras Tensor or original data.

    """
    try:  # pragma: no cover
      import keras  # pragma: no cover

      return keras.ops.convert_to_tensor(data)  # pragma: no cover
    except (ImportError, AttributeError):  # pragma: no cover
      return data  # pragma: no cover

  def get_serialization_imports(self) -> List[str]:
    """Imports for saving/loading.

    Returns:
        List[str]: Imports.

    """
    return ["import keras"]  # pragma: no cover

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Syntax for saving/loading models.

    Args:
        op (str): 'save' or 'load'.
        file_arg (str): Path string.
        object_arg (Optional[str]): Object name.

    Returns:
        str: Generated python code.

    """
    if op == "save" and object_arg:  # pragma: no cover
      return f"{object_arg}.save({file_arg})"  # pragma: no cover
    elif op == "load":  # pragma: no cover
      return f"keras.saving.load_model({file_arg})"  # pragma: no cover
    return ""  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns imports required for the generated weight migration script.

    Returns:
        List[str]: List of import statements.

    """
    return ["import keras", "import numpy as np", "import h5py"]  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """Returns python code to load a checkpoint.
    Stub implemented as Keras models contain structure + weights, making raw dict handling tricky.
    """
    return textwrap.dedent(  # pragma: no cover
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
                except Exception: 
                    print("h5py not installed, cannot load raw weights.") 
                    raw_state = {{}} 
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Returns python expression string that converts `tensor_var` from Keras tensor to numpy array."""
    return f"{tensor_var}.numpy() if hasattr({tensor_var}, 'numpy') else np.array({tensor_var})"  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Returns Python code to save the dictionary `state_var` (mapping flat keys to numpy arrays)
    to `path_var` using h5py.

    Args:
        state_var (str): Variable name of the state dictionary.
        path_var (str): Variable name of the output path.

    Returns:
        str: Generated Python code block.

    """
    return textwrap.dedent(  # pragma: no cover
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
    """Syntax for device scoping.

    Args:
        device_type (str): 'cuda', 'cpu'.
        device_index (Optional[str]): Index.

    Returns:
        str: Generated code.

    """
    d_type = "gpu" if "cuda" in device_type.lower() else "cpu"
    return f"keras.name_scope('{d_type}')"

  def get_device_check_syntax(self) -> str:
    """Syntax for checking GPU availability.

    Returns:
        str: Logic expression.

    """
    return "len(keras.config.list_logical_devices('GPU')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Keras handles RNG state internally.

    Returns:
        str: "pass".

    """
    return "pass"  # pragma: no cover

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies configuration wiring.

    Args:
        snapshot: Snapshot dict.

    """
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Provides a search URL for Keras documentation as direct API mapping is non-trivial.

    Args:
        api_name (str): API path.

    Returns:
        Optional[str]: URL.

    """
    return f"https://keras.io/search.html?q={api_name}"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns tiered examples.

    Returns:
        Dict[str, str]: Map of tiers to examples.

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
      "tier4_qwen3-vl": """import keras
from keras import layers
import keras.ops as ops

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(keras.Layer):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = layers.Conv3D(
            filters=config.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            padding="valid",
            use_bias=True,
        )

    def call(self, hidden_states):
        cfg = self.config
        seq_len = ops.shape(hidden_states)[0]

        hidden_states = ops.reshape(
            hidden_states, 
            (seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        )
        hidden_states = ops.transpose(hidden_states, (0, 2, 3, 4, 1))
        
        out = self.proj(hidden_states)
        return ops.reshape(out, (seq_len, cfg.hidden_size))
""",
    }
