"""
Torch Adapter with Ghost Protocol Introspection.

This adapter implements `collect_api` to dynamically discover PyTorch components
(Losses, Optimizers, Activations) by introspecting the `torch` namespace at runtime.
It supports **Ghost Mode**: if PyTorch is not installed (e.g. within a lightweight
CI or WASM environment), it hydrates definitions from a pre-captured JSON snapshot.
"""

import inspect
import sys
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Conditional import to prevent hard crash if torch is missing
try:
  import torch
  import torch.nn as nn
  import torch.optim as optim
except ImportError:
  torch = None
  nn = None
  optim = None

from .base import (
  register_framework,
  StructuralTraits,
  StandardMap,
  StandardCategory,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector


@register_framework("torch")
class TorchAdapter:
  """Adapter for PyTorch."""

  display_name: str = "PyTorch"
  inherits_from: None = None
  ui_priority: int = 0  # Highest Priority

  def __init__(self):
    """
    Initializes the adapter.
    Detects if 'torch' library is present. If not, enters Ghost Mode and loads snapshots.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}

    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")

  @property
  def search_modules(self) -> List[str]:
    return ["torch", "torch.nn", "torch.linalg", "torch.special", "torch.fft"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("torch", "torch")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\\.nn\\.", r"\\.modules\\.", r"\\.layers\\.", r"Module$"],
      "extras": [r"\\.utils\\.", r"\\.hub\\.", r"\\.distributed\\.", r"\\.autograd\\.", r"save$", r"load$", r"seed$"],
    }

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="torch.nn.Module",
      forward_method="forward",
      requires_super_init=True,
      strip_magic_args=["rngs"],
      lifecycle_strip_methods=["to", "cpu", "cuda", "detach", "clone", "requires_grad_", "share_memory_"],
      lifecycle_warn_methods=["eval", "train", "half", "float", "double", "type"],
      impurity_methods=["add_", "sub_", "mul_", "div_", "pow_", "zero_", "copy_", "fill_"],
    )

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Static definitions for core ops."""
    return {
      "Abs": StandardMap(api="torch.abs"),
      "Conv2d": StandardMap(api="torch.nn.Conv2d"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["manual_seed", "seed"]

  # --- Ghost Protocol Implementation ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures for the requested category.

    In LIVE mode: Introspects installed `torch` modules.
    In GHOST mode: Hydrates from loaded JSON snapshot.
    """
    # 1. GHOST Mode Path
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)

    # 2. LIVE Mode Path
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """Reads from snapshot dict."""
    if not self._snapshot_data:
      return []

    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """Scans living objects."""
    results = []

    if category == StandardCategory.LOSS:
      results.extend(self._scan_losses())
    elif category == StandardCategory.OPTIMIZER:
      results.extend(self._scan_optimizers())
    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_activations())

    return results

  def _scan_losses(self) -> List[GhostRef]:
    """Scans torch.nn for Loss functions."""
    if not nn:
      return []

    found = []
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj):
        # Heuristic: Ends with 'Loss' (covers MSELoss, CrossEntropyLoss, etc.)
        # Also check inheritance from _Loss if possible, but naming is safer across versions
        if name.endswith("Loss") and name != "_Loss":
          ref = GhostInspector.inspect(obj, f"torch.nn.{name}")
          found.append(ref)
    return found

  def _scan_optimizers(self) -> List[GhostRef]:
    """Scans torch.optim for Optimizers."""
    if not optim:
      return []

    found = []
    for name, obj in inspect.getmembers(optim):
      if inspect.isclass(obj) and name != "Optimizer":
        # Check inheritance safety using getattr first (optimization)
        # issubclass(obj, optim.Optimizer)
        try:
          if issubclass(obj, optim.Optimizer):
            ref = GhostInspector.inspect(obj, f"torch.optim.{name}")
            found.append(ref)
        except TypeError:
          pass
    return found

  def _scan_activations(self) -> List[GhostRef]:
    """Scans torch.nn for Activations (Modules defined in torch.nn.modules.activation)."""
    if not nn:
      return []

    found = []
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj):
        # Filter for nn.Module subclasses
        try:
          if issubclass(obj, nn.Module):
            # Heuristic: Logic usually lives in 'activation' submodule in Source
            # obj.__module__ is 'torch.nn.modules.activation'
            mod_name = getattr(obj, "__module__", "")

            # Explicit check for known activation pattern or module path
            # Also list specific common activations to be safe against varying struct
            is_activation_mod = "activation" in mod_name

            known_names = {"ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "Elu", "SiLU"}

            if is_activation_mod or name in known_names:
              ref = GhostInspector.inspect(obj, f"torch.nn.{name}")
              found.append(ref)
        except TypeError:
          pass

    return found

  # --- Interface Implementation ---

  def get_device_syntax(self, device_type: str, device_index: None | str = None) -> str:
    args = [str(device_type)]
    if device_index:
      args.append(str(device_index))
    arg_str = ", ".join(args)
    return f"torch.device({arg_str})"

  def get_serialization_imports(self) -> List[str]:
    return ["import torch"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: None | str = None) -> str:
    if op == "save" and object_arg:
      return f"torch.save({object_arg}, {file_arg})"
    elif op == "load":
      return f"torch.load({file_arg})"
    return ""

    # --- Verification ---

  def convert(self, data):
    try:
      import torch
    except ImportError:
      return data

    if isinstance(data, (np.ndarray, np.generic)):
      try:
        return torch.from_numpy(data)
      except Exception:
        return torch.tensor(data)
    return data

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Injects PyTorch specific manual mapping hooks."""
    mappings = snapshot.setdefault("mappings", {})

    # Map Abstract Ops to internal Torch API paths to ensure detection
    # Logic from wire_state_container.py
    mappings["register_buffer"] = {"api": "torch.nn.Module.register_buffer"}
    mappings["register_parameter"] = {"api": "torch.nn.Module.register_parameter"}
    mappings["state_dict"] = {"api": "torch.nn.Module.state_dict"}
    mappings["load_state_dict"] = {"api": "torch.nn.Module.load_state_dict"}
    mappings["parameters"] = {"api": "torch.nn.Module.parameters"}
