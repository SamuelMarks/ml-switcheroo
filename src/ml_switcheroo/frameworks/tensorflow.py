"""
TensorFlow Framework Adapter.

This module implements the adapter for TensorFlow (Core & Keras), providing
mappings for math operations, neural layers, and IO. It serves as the bridge
between abstract operations and ``tf.*`` or ``tf.keras.*`` APIs.

It supports:
1.  **Live Inspection**: Scanning installed TensorFlow modules for new operations.
2.  **Ghost Mode**: Loading API signatures from JSON snapshots when TF is not installed.
3.  **Semantic Definitions**: Static mappings for standard Math, NN, and Optimization ops.
4.  **Weight Migration**: Loading checkpoints via ``tf.train.load_checkpoint``.
"""

import logging
import textwrap
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
  InitMode,
  load_snapshot_for_adapter,
)
from ml_switcheroo.frameworks.loader import load_definitions

try:
  import tensorflow as tf
except ImportError:
  tf = None


@register_framework("tensorflow")
class TensorFlowAdapter:
  """
  Adapter for TensorFlow (Core & Keras).

  Handles the specific idioms of TensorFlow 2.x, including:
  -   mapping ``torch.nn.Module`` to ``keras.Layer``.
  -   mapping ``torch.*`` math to ``tf.math.*``.
  -   mapping IO operations to ``tf.io.*``.
  """

  display_name: str = "TensorFlow"
  inherits_from: Optional[str] = None
  ui_priority: int = 30

  def __init__(self) -> None:
    """
    Initializes the adapter.

    Detects if TensorFlow is installed to determine the initialization mode:
    -   **LIVE**: If ``tensorflow`` is importable, introspection runs on live objects.
    -   **GHOST**: If missing, it attempts to load a ``tensorflow_v*.json`` snapshot
        to allow basic translation without dependencies.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}

    if tf is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("tensorflow")
      if not self._snapshot_data:
        logging.debug("TensorFlow not installed and no snapshot found.")

  # --- Metadata ---

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Returns a set of submodule names to exclude from recursion.

    These modules often contain C-Extensions, deprecated logic, or
    heavy dependencies that crash the ``GhostInspector`` or cause
    infinite recursion during discovery.

    Returns:
        Set[str]: Blacklisted submodule names.
    """
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
    """
    Returns the list of top-level modules to scan during discovery.

    If in GHOST mode, returns an empty list to prevent import errors.

    Returns:
        List[str]: Module names (e.g. ``['tensorflow', 'keras.layers']``).
    """
    if self._mode == InitMode.GHOST:
      return []
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
    """
    Returns the primary import statement configuration.

    Returns:
        Tuple[str, str]: The module name and its standard alias (e.g. ``('tensorflow', 'tf')``).
    """
    return ("tensorflow", "tf")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Defines the semantic roles of TensorFlow namespaces.

    This config guides the ``ImportFixer`` in resolving source imports
    to target imports based on their Semantic Tier.

    Returns:
        Dict[str, ImportConfig]: Mapping of namespace strings to configuration objects.
    """
    return {
      "tensorflow": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="tf"),
      "tensorflow.data": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="tf.data"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Returns regex patterns used to categorize discovered APIs.

    Used by the ``Scaffolder`` to blindly sort discovered functions into
    Math, Neural, or Extras tiers based on their path.

    Returns:
        Dict[str, List[str]]: Mapping of Tier names to regex patterns.
    """
    return {
      "neural": [r"\\.keras\\.", r"Layer$"],
      "extras": [r"\\.io\\.", r"\\.data\\."],
    }

  @property
  def test_config(self) -> Dict[str, str]:
    """
    Returns templates for generating physical verification tests.

    These templates are used by ``gen-tests`` to create executable Python files
    that verify semantic correctness.

    Returns:
        Dict[str, str]: Templates for imports and data conversion.
    """
    return {
      "import": "import tensorflow as tf",
      "convert_input": "tf.convert_to_tensor({np_var})",
      "to_numpy": "{res_var}.numpy()",
    }

  # --- Harness Protocol ---

  @property
  def harness_imports(self) -> List[str]:
    """
    Returns extra imports required for the verification harness.
    TensorFlow requires no special initialization imports beyond the standard test config.

    Returns:
        List[str]: Empty list.
    """
    return []

  def get_harness_init_code(self) -> str:
    """
    Returns initialization logic for the verification harness.
    TensorFlow handles state implicitly, so no RNG setup code is needed.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns code to convert TF tensors to NumPy.

    Returns:
        str: Code string using safe attribute check.
    """
    return "if hasattr(obj, 'numpy'): return obj.numpy()"

  @property
  def declared_magic_args(self) -> List[str]:
    """
    Returns list of framework-specific 'magic' arguments to be stripped from other frameworks.
    TensorFlow generally uses explicit arguments or class attributes, not injected magic args.

    Returns:
        List[str]: Empty list.
    """
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Returns structural transformation rules for TensorFlow/Keras.

    Defines how classes should be rewritten (e.g. inheriting from ``keras.Layer``),
    what the forward method is named (``call``), and initialization requirements.

    Returns:
        StructuralTraits: The configuration object.
    """
    return StructuralTraits(
      module_base="keras.Layer",
      forward_method="call",
      requires_super_init=True,
      auto_strip_magic_args=True,  # Decoupled
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Returns capability flags for the TensorFlow ecosystem.

    Indicates support for NumPy-compatible methods (allowing ``.astype`` via ops)
    but lack of functional state requirements.

    Returns:
        PluginTraits: The capability flags.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=True,  # Supports .astype via TF/Keras ops
      requires_explicit_rng=False,  # TF handles RNG statefully (usually)
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """
    Returns the Semantic Tiers supported by this adapter.

    Returns:
        List[SemanticTier]: Array API, Neural, and Extras.
    """
    return [SemanticTier.ARRAY_API, SemanticTier.NEURAL, SemanticTier.EXTRAS]

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    Returns the static dictionary of Operation Mappings.
    Loaded dynamically from `frameworks/definitions/tensorflow.json`.

    Returns:
        Dict[str, StandardMap]: The mapping dictionary.
    """
    return load_definitions("tensorflow")

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns list of methods used to set global random seeds.
    Used by the PurityScanner to detect side-effects.

    Returns:
        List[str]: Method names (e.g. ``set_seed``).
    """
    return ["set_seed", "random.set_seed"]

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API definitions for the discovery process.

    Delegates to ``_collect_ghost`` if in Ghost Mode, otherwise runs
    ``_collect_live`` to inspect the installed library.

    Args:
        category (StandardCategory): The category of operations to scan (e.g. LOSS, LAYER).

    Returns:
        List[GhostRef]: A list of discovered API signatures.
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """
    Loads API definitions from the cached JSON snapshot.

    Args:
        category (StandardCategory): The category to retrieve.

    Returns:
        List[GhostRef]: Hydrated references from JSON data.
    """
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """
    Introspects the live ``tensorflow`` module to find operations matching the category.

    Scans ``tf.nn`` for activations and ``tf.keras.layers`` for layers.

    Args:
        category (StandardCategory): The category to scan for.

    Returns:
        List[GhostRef]: References extracted from live objects.
    """
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
        # Basic keras layer scanning if available
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
    """
    Applies manual wiring patches to the generated snapshot.
    TensorFlow currently requires no manual wiring corrections.

    Args:
        snapshot (Dict[str, Any]): The snapshot dictionary to modify in-place.
    """
    pass

  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns the primary example code used for documentation.

    Returns:
        str: The Neural Tier (Tier 2) example code.
    """
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns example snippets for each semantic tier.

    Used by the interactive web demo and documentation generator.

    Returns:
        Dict[str, str]: Mapping of tier IDs to Python code strings.
    """
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
    """
    Generates Python code for defining a device context in TensorFlow.

    Example: ``tf.device('GPU:0')``

    Args:
        device_type (str): The device type string (e.g. 'cuda', 'cpu').
        device_index (Optional[str]): The device index.

    Returns:
        str: Arguments for ``tf.device(...)``.
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
        return f"tf.device(f'{tf_type}:{{str({device_index})}}')"

    return f"tf.device('{tf_type}:{idx_str}')"

  def get_device_check_syntax(self) -> str:
    """
    Returns Python code to check for GPU availability.

    Returns:
        str: ``len(tf.config.list_physical_devices('GPU')) > 0``
    """
    return "len(tf.config.list_physical_devices('GPU')) > 0"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    Returns syntax for splitting RNG state.
    Since TF uses internal state, this returns 'pass' (no-op).

    Args:
        rng_var (str): Variable name for current RNG.
        key_var (str): Variable name for new key.

    Returns:
        str: "pass"
    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """
    Returns imports required for IO operations.

    Returns:
        List[str]: ``['import tensorflow as tf']``
    """
    return ["import tensorflow as tf"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Generates code for saving or loading artifacts.

    Args:
        op (str): Operation type ('save' or 'load').
        file_arg (str): Path to file.
        object_arg (Optional[str]): Object to save.

    Returns:
        str: Generated code (e.g. ``tf.io.write_file``).
    """
    if op == "save" and object_arg:
      return f"tf.io.write_file({file_arg}, {object_arg})"
    elif op == "load":
      return f"tf.io.read_file({file_arg})"
    return ""

  # --- Weight Handling Logic ---

  def get_weight_conversion_imports(self) -> List[str]:
    """
    Returns imports required for the generated weight migration script.

    Returns:
        List[str]: List of import statements.
    """
    return ["import tensorflow as tf", "import numpy as np"]

  def get_weight_load_code(self, path_var: str) -> str:
    """
    Returns Python code to load a TF checkpoint into a raw dict found in `raw_state`.
    Attempt to load variable map if available.

    Args:
        path_var: Variable name containing file path.

    Returns:
        Code block string.
    """
    return textwrap.dedent(
      f"""
            try:
                reader = tf.train.load_checkpoint({path_var})
                dtypes = reader.get_variable_to_dtype_map()
                raw_state = {{key: reader.get_tensor(key) for key in dtypes}}
            except Exception as e:
                print(f"Failed to load TF checkpoint: {{e}}")
                raw_state = {{}}
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """
    Returns expression to convert a TF tensor to a NumPy array.

    Args:
        tensor_var: Variable name of the tensor.

    Returns:
        Conversion expression string.
    """
    return f"{tensor_var}.numpy() if hasattr({tensor_var}, 'numpy') else np.array({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """
    Returns logic (stubbed with warning) for saving weights.
    TensorFlow checkpoint saving generally requires a model instance structure,
    so raw dictionary saving is not supported in the standalone migration script.

    Args:
        state_var: Unused state variable name.
        path_var: Unused path variable name.

    Returns:
        Warning print code.
    """
    return textwrap.dedent(
      """
            print("WARNING: Saving raw dictionary to TensorFlow checkpoint is not directly supported without model structure.")
            print("To save weights for TensorFlow, instantiate the Keras/TF model and use `model.set_weights()` or `root.save_weights()`.")
            print(f"Weights available in converted_state variable for manual assignment.")
            """
    )

  def convert(self, data: Any) -> Any:
    """
    Converts input data (NumPy/List) into TensorFlow Tensors.
    Used by the Fuzzer for validation.

    Args:
        data (Any): Input data.

    Returns:
        Any: ``tf.Tensor`` or original data if conversion fails.
    """
    try:
      import tensorflow as tf

      return tf.convert_to_tensor(data)
    except (ImportError, ValueError, TypeError, Exception):
      return data

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Generates TensorFlow API documentation URL.
    Corrects internal `tensorflow` path to `tf` for doc references.

    Args:
        api_name (str): API Path.

    Returns:
        Optional[str]: URL.
    """
    path = api_name.replace("tensorflow.", "tf.").replace(".", "/")
    return f"https://www.tensorflow.org/api_docs/python/{path}"
