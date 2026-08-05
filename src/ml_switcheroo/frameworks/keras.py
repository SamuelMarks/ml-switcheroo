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
from typing import Any, Dict, List, Optional, Tuple

try:
  import keras
  import keras.activations
  import keras.layers
  import keras.losses
  import keras.ops
  import keras.optimizers
  import keras.random
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


from ml_switcheroo.frameworks.keras_io import KerasIOMixin


@register_framework("keras")
class KerasAdapter(KerasIOMixin):
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

    Returns:
        None

    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if keras is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("keras")
      if not self._snapshot_data:
        logging.debug("Keras not installed and no snapshot found. Adapter disabled.")

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
    from ml_switcheroo.semantics.schema import PluginTraits

    return PluginTraits(
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
      defs["ReLU"] = StandardMap(api="keras.layers.ReLU")
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
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostRef.model_validate(item) for item in raw_list]

  def _collect_live(self, category: SemanticTier) -> List[GhostRef]:
    """Scans live modules.

    Args:
        category (SemanticTier): Category to scan.

    Returns:
        List[GhostRef]: Found items.

    """
    results: list["Any"] = []
    if category == SemanticTier.LOSS:
      results.extend(
        getattr(self, "_scan_module", lambda *args, **kwargs: [])(
          keras.losses, "keras.losses", kind="class", block_list={"Loss", "Container"}
        )
      )
    elif category == SemanticTier.OPTIMIZER:
      results.extend(
        getattr(self, "_scan_module", lambda *args, **kwargs: [])(
          keras.optimizers, "keras.optimizers", kind="class", block_list={"Optimizer", "TFOptimizer"}
        )
      )
    elif category == SemanticTier.ACTIVATION:
      results.extend(
        getattr(self, "_scan_module", lambda *args, **kwargs: [])(keras.activations, "keras.activations", kind="function")
      )
    elif category == SemanticTier.LAYER:
      results.extend(
        getattr(self, "_scan_module", lambda *args, **kwargs: [])(
          keras.layers, "keras.layers", kind="class", block_list={"Layer"}
        )
      )
    return results

  def convert(self, data: Any) -> Any:
    """Converts input data to Keras Tensor.

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

    Args:
        rng_var (str): The name of the input random generator/state variable.
        key_var (str): The variable name to bind the split keys to.

    Returns:
        str: Syntax to pass or execute, which is "pass" here since it is handled internally.

    """
    return "pass"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies configuration wiring.

    Args:
        snapshot (Dict[str, Any]): The snapshot configuration data to apply.

    Returns:
        None

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
    """Returns example snippets for each semantic tier.

    Returns:
        Dict[str, str]: Example snippets categorized by semantic tier.

    """
    from ml_switcheroo.frameworks.keras_examples import get_keras_tiered_examples

    return get_keras_tiered_examples()
