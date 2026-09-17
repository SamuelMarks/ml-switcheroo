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

from typing import Any

import typing


import logging
from typing import List, Tuple, Dict, Optional

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

try:
  import torch as _torch_module
  import torch.nn as _nn_module
  import torch.optim as _optim_module
except Exception:
  _torch_module = None  # type: ignore[assignment]
  _nn_module = None  # type: ignore[assignment]
  _optim_module = None  # type: ignore[assignment]
torch: Optional[Any] = _torch_module
nn: Optional[Any] = _nn_module
optim: Optional[Any] = _optim_module


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
    """Initialize the adapter.

    Detects if PyTorch is installed. to switch between LIVE inspection
    and GHOST snapshot loading.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.debug("PyTorch not installed and no snapshot found. Scanning unavailable.")

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Return the primary root import alias ('torch', 'torch').

    Returns:
        The module name and default alias.

    """
    return "torch", "torch"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Define the semantic roles of PyTorch namespaces.

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
    """Return the semantic tiers fully supported by this adapter.

    Returns:
        List of supported tiers.

    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def test_config(self) -> Dict[str, str]:
    """Template used by `gen-tests` to create physical verification files.

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
    """Import required for harness initialization.

    Returns:
        List of import statements.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Return helper code for initializing the harness.

    Returns:
        Python source code string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Return code to convert Torch tensors to NumPy (detach/cpu check).

    Returns:
        Python statement string.

    """
    return "if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()"

  @property
  def structural_traits(self) -> StructuralTraits:
    """Define how classes and functions are rewritten when targeting PyTorch.

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
    """Capability flags. PyTorch uses imperative state and eager execution.

    Returns:
        Configuration object for plugin logic.

    """
    return PluginTraits(
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
    """Return list of framework-specific magic arguments.

    Torch emits no magic args; all state is implicit.

    Returns:
        Empty list.

    """
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Execute definitive mapping of Abstract Operations to PyTorch APIs.

    Loaded dynamically from `frameworks/definitions/torch.json`.

    Returns:
        Dictionary mapping operation abstract IDs to implementation details.

    """
    defs = load_definitions("torch")
    if "ReLU" not in defs:
      defs["ReLU"] = StandardMap(api="torch.nn.ReLU")
    # if "relu" not in defs:
    #   defs["relu"] = StandardMap(api="torch.relu")
    if "Linear" not in defs:
      defs["Linear"] = StandardMap(
        api="torch.nn.Linear", args={"in_features": "in_features", "out_features": "out_features"}
      )
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="torch.nn.Conv2d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    if "Conv1d" not in defs:
      defs["Conv1d"] = StandardMap(
        api="torch.nn.Conv1d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    if "Conv3d" not in defs:
      defs["Conv3d"] = StandardMap(
        api="torch.nn.Conv3d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    if "ConvTranspose2d" not in defs:
      defs["ConvTranspose2d"] = StandardMap(
        api="torch.nn.ConvTranspose2d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    return defs

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generate code for device creation.

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
    """Return PyTorch syntax for checking CUDA availability.

    Returns:
        Python expression string.

    """
    return "torch.cuda.is_available()"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Return syntax for splitting RNG state.

    PyTorch uses global state-based randomness, so explicit splitting is a no-op.

    Args:
        rng_var: Variable name holding random state.
        key_var: Variable name for the new key.

    Returns:
        'pass' string (No-op).

    """
    return "pass"

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Return the official PyTorch documentation URL.

    Args:
        api_name: The fully qualified API name.

    Returns:
        URL string or None.

    """
    if "nn.init" in api_name:
      return f"https://pytorch.org/docs/stable/nn.init.html#{api_name}"
    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Return example snippets for each semantic tier."""
    from ml_switcheroo.frameworks.torch_examples import get_torch_tiered_examples

    return get_torch_tiered_examples()

  def convert(
    self, data: typing.Union[int, float, str, list, dict]
  ) -> typing.Union[int, float, str, list, dict, typing.Any]:
    """Convert input data (NumPy, lists) into PyTorch Tensors for verification runners.

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
      except Exception:
        return torch.tensor(data)
    if isinstance(data, (list, tuple)):
      try:
        return torch.tensor(data)
      except Exception:
        pass
    return data

  def _collect_ghost(self, category: SemanticTier) -> List[GhostRef]:
    """Load definitions from JSON snapshot.

    Args:
        category: The standard category to filter by.

    Returns:
        List of hydrated GhostRef objects.

    """
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostRef.model_validate(item) for item in raw_list]

  def _collect_live(self, category: SemanticTier) -> List[GhostRef]:
    """Introspect live torch modules.

    Args:
        category: The standard category to filter by.

    Returns:
        List of discovered GhostRef objects.

    """
    results: list[Any] = []
    if category == SemanticTier.LOSS:
      results.extend(getattr(self, "_scan_losses", lambda: [])())
    elif category == SemanticTier.OPTIMIZER:
      results.extend(getattr(self, "_scan_optimizers", lambda: [])())
    elif category == SemanticTier.ACTIVATION:
      results.extend(getattr(self, "_scan_activations", lambda: [])())
    elif category == SemanticTier.LAYER:
      results.extend(getattr(self, "_scan_layers", lambda: [])())
    return results

  def apply_wiring(self, snapshot: typing.Dict[str, dict]) -> None:
    """Apply manual patches to the standard mappings if necessary.

    Used to inject complex behaviors not captured by simple API scanning.

    Args:
        snapshot: The snapshot dictionary to update in-place.

    """
    snapshot.setdefault("mappings", {})
