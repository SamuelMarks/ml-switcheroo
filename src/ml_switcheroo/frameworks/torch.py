"""PyTorch Framework Adapter.

This module implements the `FrameworkAdapter` protocol for PyTorch.
It provides:

1.  **Import Abstraction**: Self-declared namespace mappings (e.g., `torch.nn` is `NEURAL`).
2.  **Semantic Definitions**: Mappings loaded from `definitions/torch.json` via helper.
3.  **Discovery**: Heuristics and logic for scanning the installed `torch` library.
4.  **IO & Device Support**: Wires up serialization and device allocation.
5.  **Weight Migration**: Implements logic to generate scripts for converting .pth checkpoints
    to/from NumPy format for interoperability.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

try:
  import torch
  import torch.nn as nn
  import torch.optim as optim  # pragma: no cover
except Exception:
  torch: Any = None  # type: ignore
  nn = None  # type: ignore
  optim = None  # type: ignore
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardMap,
  ImportConfig,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.frameworks.loader import load_definitions


from ml_switcheroo.frameworks.torch_io import TorchIOMixin


@register_framework("torch")
class TorchAdapter(TorchIOMixin):
  """Adapter for PyTorch (Meta).

  Handles Source and Target translation rules for PyTorch, including
  `torch.nn`, `torch.optim`, and `torch.func` (vmap/grad).
  """

  display_name: str = "PyTorch"
  inherits_from: Optional[str] = None
  ui_priority: int = 0

  def __init__(self) -> None:
    """Initializes the adapter.

    Detects if PyTorch is installed. to switch between LIVE inspection
    and GHOST snapshot loading.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")  # pragma: no cover

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Returns the primary root import alias ('torch', 'torch').

    Returns:
        The module name and default alias.

    """
    return "torch", "torch"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Defines the semantic roles of PyTorch namespaces.

    Returns:
        Mapping of dot-path strings to configuration objects.

    """
    return {
      "torch": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="torch"),
      "torch.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "torch.nn.functional": ImportConfig(tier=SemanticTier.NEURAL_OPS, recommended_alias="F"),
      "torch.optim": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optim"),
      "torch.utils.data": ImportConfig(tier=SemanticTier.EXTRAS),
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Returns the semantic tiers fully supported by this adapter.

    Returns:
        List of supported tiers.

    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def test_config(self) -> Dict[str, str]:
    """Templates used by `gen-tests` to create physical verification files.

    Returns:
        Dictionary of code templates.

    """
    return {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.detach().cpu().numpy()",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Imports required for harness initialization.

    Returns:
        List of import statements.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Returns helper code for initializing the harness.

    Returns:
        Python source code string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns code to convert Torch tensors to NumPy (detach/cpu check).

    Returns:
        Python statement string.

    """
    return "if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()"

  @property
  def structural_traits(self) -> StructuralTraits:
    """Defines how classes and functions are rewritten when targeting PyTorch.

    Returns:
        Configuration object for structural rewriting.

    """
    return StructuralTraits(
      module_base="torch.nn.Module",
      forward_method="forward",
      requires_super_init=True,
      auto_strip_magic_args=True,
      lifecycle_strip_methods=["to", "cpu", "cuda", "detach", "clone", "requires_grad_", "share_memory_"],
      lifecycle_warn_methods=["eval", "train"],
      impurity_methods=["add_", "sub_", "mul_", "div_", "pow_", "zero_", "copy_", "fill_"],
      jit_static_args=[],
      implicit_method_roots=["torch.Tensor"],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Capabilities flags. PyTorch uses imperative state and eager execution.

    Returns:
        Configuration object for plugin logic.

    """
    return PluginTraits(  # pragma: no cover
      has_numpy_compatible_arrays=False,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
      sharding_wrapper_api="torch.distributed.fsdp.FSDP",
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    """Global seed setting methods detected as impure side-effects.

    Returns:
        List of method names.

    """
    return ["manual_seed", "seed"]

  @property
  def declared_magic_args(self) -> List[str]:
    """Returns list of framework-specific magic arguments.


    Torch emits no magic args; all state is implicit.

    Returns:
        Empty list.

    """
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """The definitive mapping of Abstract Operations to PyTorch APIs.

    Loaded dynamically from `frameworks/definitions/torch.json`.

    Returns:
        Dictionary mapping operation abstract IDs to implementation details.

    """
    defs = load_definitions("torch")
    if "ReLU" not in defs:
      defs["ReLU"] = StandardMap(api="torch.nn.ReLU")  # pragma: no cover
    if "relu" not in defs:
      defs["relu"] = StandardMap(api="torch.nn.functional.relu")  # pragma: no cover
    if "Linear" not in defs:
      defs["Linear"] = StandardMap(  # pragma: no cover
        api="torch.nn.Linear", args={"in_features": "in_features", "out_features": "out_features"}
      )
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(  # pragma: no cover
        api="torch.nn.Conv2d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    return defs

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generates code for device creation.

    Args:
        device_type: The device type string (e.g. 'cuda', 'cpu').
        device_index: The optional device index.

    Returns:
        Code string for device creation.

    """
    args = [str(device_type)]
    if device_index:
      args.append(str(device_index))
    arg_str = ", ".join(args)
    return f"torch.device({arg_str})"

  def get_device_check_syntax(self) -> str:
    """Returns PyTorch syntax for checking CUDA availability.

    Returns:
        Python expression string.

    """
    return "torch.cuda.is_available()"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Returns syntax for splitting RNG state.

    PyTorch uses global state-based randomness, so explicit splitting is a no-op.

    Args:
        rng_var: Variable name holding random state.
        key_var: Variable name for the new key.

    Returns:
        'pass' string (No-op).

    """
    return "pass"  # pragma: no cover

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Returns the official PyTorch documentation URL.

    Args:
        api_name: The fully qualified API name.

    Returns:
        URL string or None.

    """
    if "nn.init" in api_name:
      return f"https://pytorch.org/docs/stable/nn.init.html#{api_name}"
    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Returns example snippets for each semantic tier."""
    from ml_switcheroo.frameworks.torch_examples import get_torch_tiered_examples

    return get_torch_tiered_examples()

  def convert(self, data: Any) -> Any:
    """Converts input data (numpy, lists) into PyTorch Tensors for verification runners.

    Args:
        data: Input data structure.

    Returns:
        Converted PyTorch Tensor or original data if conversion fails.

    """
    try:
      import torch
      import numpy as np
    except Exception:
      return data
    if isinstance(data, (np.ndarray, np.generic)):
      try:
        return torch.from_numpy(data)
      except Exception:  # pragma: no cover
        return torch.tensor(data)  # pragma: no cover
    if isinstance(data, (list, tuple)):
      try:  # pragma: no cover
        return torch.tensor(data)  # pragma: no cover
      except Exception:  # pragma: no cover
        pass  # pragma: no cover
    return data

  def _collect_ghost(self, category: SemanticTier) -> List[GhostRef]:
    """Loads definitions from JSON snapshot.

    Args:
        category: The standard category to filter by.

    Returns:
        List of hydrated GhostRef objects.

    """
    if not self._snapshot_data:  # pragma: no cover
      return []  # pragma: no cover
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])  # pragma: no cover
    return [GhostRef.model_validate(item) for item in raw_list]  # pragma: no cover

  def _collect_live(self, category: SemanticTier) -> List[GhostRef]:
    """Introspects live torch modules.

    Args:
        category: The standard category to filter by.

    Returns:
        List of discovered GhostRef objects.

    """
    results: list = []  # pragma: no cover  # type: ignore
    if category == SemanticTier.LOSS:  # pragma: no cover
      results.extend(getattr(self, "_scan_losses", lambda: [])())  # pragma: no cover
    elif category == SemanticTier.OPTIMIZER:  # pragma: no cover
      results.extend(getattr(self, "_scan_optimizers", lambda: [])())  # pragma: no cover
    elif category == SemanticTier.ACTIVATION:  # pragma: no cover
      results.extend(getattr(self, "_scan_activations", lambda: [])())  # pragma: no cover
    elif category == SemanticTier.LAYER:  # pragma: no cover
      results.extend(getattr(self, "_scan_layers", lambda: [])())  # pragma: no cover
    return results  # pragma: no cover

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply manual patches to the standard mappings if necessary.

    Used to inject complex behaviors not captured by simple API scanning.

    Args:
        snapshot: The snapshot dictionary to update in-place.

    """
    snapshot.setdefault("mappings", {})
