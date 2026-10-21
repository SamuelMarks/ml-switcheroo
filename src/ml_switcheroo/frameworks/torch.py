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
import typing
from types import ModuleType

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.frameworks.base import (
  GhostRef,
  ImportConfig,
  InitMode as InitMode,
  PluginTraits,
  StandardMap,
  StructuralTraits,
  load_snapshot_for_adapter,
  register_framework,
)
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.frameworks.torch_io import TorchIOMixin

torch: ModuleType | None = None
nn: ModuleType | None = None
optim: ModuleType | None = None
try:
  import torch as _torch_module
  import torch.nn as _nn_module
  import torch.optim as _optim_module

  torch = _torch_module
  nn = _nn_module
  optim = _optim_module
except (ImportError, AttributeError, TypeError, ValueError) as exc:
  logging.getLogger(__name__).debug("Failed to import torch modules: %s", exc)


@register_framework("torch")
class TorchAdapter(TorchIOMixin):
  """Adapter for PyTorch (Meta).

  Handles Source and Target translation rules for PyTorch, including
  `torch.nn`, `torch.optim`, and `torch.func` (vmap/grad).
  """

  display_name: str = "PyTorch"
  inherits_from: str | None = None
  ui_priority: int = 0

  def __init__(self) -> None:
    """Initialize the adapter.

    Detects if PyTorch is installed. to switch between LIVE inspection
    and GHOST snapshot loading.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: dict[str, JsonValue] = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.debug("PyTorch not installed and no snapshot found. Scanning unavailable.")

  @property
  def import_alias(self) -> tuple[str, str]:
    """Return the primary root import alias ('torch', 'torch').

    Returns:
        The module name and default alias.

    """
    return "torch", "torch"

  @property
  def import_namespaces(self) -> dict[str, ImportConfig]:
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
  def supported_tiers(self) -> list[SemanticTier]:
    """Return the semantic tiers fully supported by this adapter.

    Returns:
        List of supported tiers.

    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def test_config(self) -> dict[str, str]:
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
  def harness_imports(self) -> list[str]:
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
  def rng_seed_methods(self) -> list[str]:
    """Global seed setting methods detected as impure side-effects.

    Returns:
        List of method names.

    """
    return ["manual_seed", "seed"]

  @property
  def declared_magic_args(self) -> list[str]:
    """Return list of framework-specific magic arguments.

    Torch emits no magic args; all state is implicit.

    Returns:
        Empty list.

    """
    return []

  @property
  def definitions(self) -> dict[str, StandardMap]:
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

  def get_device_syntax(self, device_type: str, device_index: str | None = None) -> str:
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

  def get_doc_url(self, api_name: str) -> str | None:
    """Return the official PyTorch documentation URL.

    Args:
        api_name: The fully qualified API name.

    Returns:
        URL string or None.

    """
    if "nn.init" in api_name:
      return f"https://pytorch.org/docs/stable/nn.init.html#{api_name}"
    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"

  def get_tiered_examples(self) -> dict[str, str]:
    """Return example snippets for each semantic tier.

    Returns:
        Dict[str, str]: Mapping of tier identifiers to PyTorch example code snippets.
    """
    from ml_switcheroo.frameworks.torch_examples import get_torch_tiered_examples

    return get_torch_tiered_examples()

  def convert(self, data: typing.Hashable | typing.Sized) -> typing.Hashable | typing.Sized:
    """Convert input data (NumPy, lists) into PyTorch Tensors for verification runners.

    Args:
        data: Input data structure.

    Returns:
        Converted PyTorch Tensor or original data if conversion fails.

    """
    try:
      import numpy as np
      import torch
    except (ImportError, AttributeError) as exc:
      logging.getLogger(__name__).debug("Torch/NumPy import failed during convert: %s", exc)
      return data
    if isinstance(data, (np.ndarray, np.generic)):
      try:
        res_np: typing.Hashable | typing.Sized = torch.from_numpy(data)
        return res_np
      except (TypeError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).debug("torch.from_numpy fallback to torch.tensor: %s", exc)
        res_fb: typing.Hashable | typing.Sized = torch.tensor(data)
        return res_fb
    if isinstance(data, (list, tuple)):
      try:
        res_seq: typing.Hashable | typing.Sized = torch.tensor(data)
        return res_seq
      except (TypeError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).debug("torch.tensor conversion failed: %s", exc)
    return data

  def _collect_ghost(self, category: SemanticTier) -> list[GhostRef]:
    """Load definitions from JSON snapshot.

    Args:
        category: The standard category to filter by.

    Returns:
        List of hydrated GhostRef objects.

    """
    if not self._snapshot_data:
      return []
    raw_cats = getattr(self._snapshot_data, "get")("categories", {})
    raw_list = getattr(raw_cats, "get")(category.value, [])
    return [GhostRef.model_validate(item) for item in raw_list]

  def _collect_live(self, category: SemanticTier) -> list[GhostRef]:
    """Introspect live torch modules.

    Args:
        category: The standard category to filter by.

    Returns:
        List of discovered GhostRef objects.

    """
    results: list[GhostRef] = []
    if category == SemanticTier.LOSS:
      results.extend(getattr(self, "_scan_losses", lambda: [])())
    elif category == SemanticTier.OPTIMIZER:
      results.extend(getattr(self, "_scan_optimizers", lambda: [])())
    elif category == SemanticTier.ACTIVATION:
      results.extend(getattr(self, "_scan_activations", lambda: [])())
    elif category == SemanticTier.LAYER:
      results.extend(getattr(self, "_scan_layers", lambda: [])())
    return results

  def apply_wiring(self, snapshot: dict[str, dict[str, JsonValue]]) -> None:
    """Apply manual patches to the standard mappings if necessary.

    Used to inject complex behaviors not captured by simple API scanning.

    Args:
        snapshot: The snapshot dictionary to update in-place.

    """
    snapshot.setdefault("mappings", {})
