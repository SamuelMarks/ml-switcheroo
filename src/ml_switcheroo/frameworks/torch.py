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
  ui_priority: int = 0

  def __init__(self):
    self._mode = InitMode.LIVE
    self._snapshot_data = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")

  @property
  def search_modules(self) -> List[str]:
    return [
      "torch",
      "torch.nn",
      "torch.linalg",
      "torch.special",
      "torch.fft",
      "torch.nn.functional",
      "torchvision.transforms",
    ]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("torch", "torch")

  @property
  def import_namespaces(self) -> Dict[str, Dict[str, str]]:
    return {
      "torch.nn": {"alias": "nn", "sub": "nn"},
      # Fix: map root correctly so it becomes 'import torch.nn.functional as F'
      # 'root' is the module to import, 'sub' is None means we import the root directly
      "torch.nn.functional": {"root": "torch.nn.functional", "alias": "F", "sub": None},
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\.nn\\.", r"\.modules\\.", r"\.layers\\.", r"Module$"],
      "extras": [r"\.utils\\.", r"\.hub\\.", r"\.distributed\\.", r"\.autograd\\.", r"save$", r"load$", r"seed$"],
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
    return {
      "Abs": StandardMap(api="torch.abs"),
      "Conv2d": StandardMap(api="torch.nn.Conv2d"),
      "relu": StandardMap(api="torch.nn.functional.relu"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["manual_seed", "seed"]

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural"]

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": """import torch\n\ndef math_ops(x, y):\n  return torch.mean(torch.add(torch.abs(x), y))""",
      "tier2_neural": """import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n  def forward(self, x): return x""",
      "tier3_extras": """import torch\nfrom torch.utils.data import DataLoader""",
    }

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    results = []
    if category == StandardCategory.LOSS:
      results.extend(self._scan_losses())
    elif category == StandardCategory.OPTIMIZER:
      results.extend(self._scan_optimizers())
    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_activations())
    return results

  def _scan_losses(self) -> List[GhostRef]:
    if not nn:
      return []
    found = []
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
        ref = GhostInspector.inspect(obj, f"torch.nn.{name}")
        found.append(ref)
    return found

  def _scan_optimizers(self) -> List[GhostRef]:
    if not optim:
      return []
    found = []
    for name, obj in inspect.getmembers(optim):
      if inspect.isclass(obj) and name != "Optimizer":
        try:
          if issubclass(obj, optim.Optimizer):
            ref = GhostInspector.inspect(obj, f"torch.optim.{name}")
            found.append(ref)
        except TypeError:
          pass
    return found

  def _scan_activations(self) -> List[GhostRef]:
    if not nn:
      return []
    found = []
    # 1. Class Activations
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj):
        try:
          if issubclass(obj, nn.Module):
            mod_name = getattr(obj, "__module__", "")
            known_names = {"ReLU", "GELU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "Elu", "SiLU"}
            if "activation" in mod_name or name in known_names:
              ref = GhostInspector.inspect(obj, f"torch.nn.{name}")
              found.append(ref)
        except TypeError:
          pass
    # 2. Functional Activations
    try:
      import torch.nn.functional as F

      targets = ["relu", "gelu", "sigmoid", "tanh", "softmax", "log_softmax", "silu", "elu", "leaky_relu"]
      for name in targets:
        if hasattr(F, name):
          obj = getattr(F, name)
          ref = GhostInspector.inspect(obj, f"torch.nn.functional.{name}")
          found.append(ref)
    except ImportError:
      pass
    return found

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
    mappings = snapshot.setdefault("mappings", {})
    imports = snapshot.setdefault("imports", {})

    mappings["register_buffer"] = {"api": "torch.nn.Module.register_buffer"}
    mappings["register_parameter"] = {"api": "torch.nn.Module.register_parameter"}
    mappings["state_dict"] = {"api": "torch.nn.Module.state_dict"}
    mappings["load_state_dict"] = {"api": "torch.nn.Module.load_state_dict"}
    mappings["parameters"] = {"api": "torch.nn.Module.parameters"}

    for vision_op in [
      "Resize",
      "Normalize",
      "ToTensor",
      "CenterCrop",
      "RandomCrop",
      "RandomHorizontalFlip",
      "RandomVerticalFlip",
      "Pad",
      "Grayscale",
    ]:
      if vision_op not in mappings:
        mappings[vision_op] = {"api": f"torchvision.transforms.{vision_op}"}

    imports["torch.nn"] = {"root": "torch", "sub": "nn", "alias": "nn"}
    imports["torch.nn.functional"] = {"root": "torch.nn.functional", "alias": "F", "sub": None}

    if "sort" in mappings:
      mappings["sort"]["output_adapter"] = "lambda x: x.values"

    mappings["relu"] = {"api": "torch.nn.functional.relu"}
    activations = ["gelu", "sigmoid", "tanh", "softmax", "log_softmax", "silu", "elu", "leaky_relu"]
    for act in activations:
      if act not in mappings:
        mappings[act] = {"api": f"torch.nn.functional.{act}"}
