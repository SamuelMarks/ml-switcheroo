"""
PyTorch Adapter with Dynamic Introspection.

This adapter serves as the primary "Source of Truth" for many Deep Learning
standards (Layers, Optimizers).

Updates:
- Removed hardcoded lists for Activations. Now scans ``torch.nn.modules.activation``.
- Dynamic scanning of Optimizers via ``torch.optim.Optimizer`` inheritance check.
- Dynamic scanning of Losses via naming convention.
- Strict filtering applied to activation scanning to prevent layer leakage.
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

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import (
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
    """
    Initializes the adapter.
    Checks for Torch installation to determine Live vs Ghost mode.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")

  # --- Metadata & Discovery Config ---

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
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def import_namespaces(self) -> Dict[str, Dict[str, str]]:
    return {
      "torch.nn": {"alias": "nn", "sub": "nn"},
      "torch.nn.functional": {"root": "torch.nn.functional", "alias": "F", "sub": None},
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\\.nn\\.", r"\\.modules\\.", r"\\.layers\\.", r"Module$"],
      "extras": [
        r"\\.utils\\.",
        r"\\.hub\\.",
        r"\\.distributed\\.",
        r"\\.autograd\\.",
        r"save$",
        r"load$",
        r"seed$",
      ],
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
    """
    Static definitions for base operations to ensure robustness.
    """
    return {
      "Abs": StandardMap(api="torch.abs"),
      "Conv2d": StandardMap(api="torch.nn.Conv2d"),
      "relu": StandardMap(api="torch.nn.functional.relu"),
      "permute_dims": StandardMap(api="torch.permute"),
    }

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["manual_seed", "seed"]

  # --- Discovery Logic (Dynamic) ---

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Collects API signatures dynamically.
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
    results = []
    if category == StandardCategory.LOSS:
      results.extend(self._scan_losses())
    elif category == StandardCategory.OPTIMIZER:
      results.extend(self._scan_optimizers())
    elif category == StandardCategory.ACTIVATION:
      results.extend(self._scan_activations())
    elif category == StandardCategory.LAYER:
      results.extend(self._scan_layers())
    return results

  def _scan_losses(self) -> List[GhostRef]:
    if not nn:
      return []
    found = []
    # Dynamic scan of torch.nn for Loss classes
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
        if issubclass(obj, nn.Module):
          found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

  def _scan_optimizers(self) -> List[GhostRef]:
    if not optim:
      return []
    found = []
    # Dynamic scan of torch.optim for Optimizer subclasses
    for name, obj in inspect.getmembers(optim):
      if inspect.isclass(obj) and name != "Optimizer":
        try:
          if issubclass(obj, optim.Optimizer):
            found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
        except TypeError:
          pass
    return found

  def _scan_activations(self) -> List[GhostRef]:
    """Scans for Activations (both Class and Functional)."""
    found = []
    # Allowlist of known activations to prevent layers like Conv2d or losses leaking in
    known_names = {
      "ReLU",
      "GELU",
      "Sigmoid",
      "Tanh",
      "Softmax",
      "LeakyReLU",
      "Elu",
      "SiLU",
      "Hardswish",
      "Mish",
      "LogSoftmax",
      "ReLU6",
      "PReLU",
      "SELU",
      "CELU",
      "Softplus",
      "Softshrink",
      "Softsign",
      "Tanhshrink",
      "Threshold",
      "GLU",
      "Hardsigmoid",
      "Hardtanh",
    }

    # 1. Class-based Activations (torch.nn.modules.activation)
    # We try to import the dedicated submodule to be precise, or scan nn if that fails
    try:
      import torch.nn.modules.activation as activ

      for name, obj in inspect.getmembers(activ):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
          # Ensure strict filtering for unexpected leaks (even in dedicated module)
          # Only accept known activation names
          if name in known_names:
            found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    except ImportError:
      # Fallback scan of nn
      if nn:
        for name, obj in inspect.getmembers(nn):
          if name in known_names and inspect.isclass(obj):
            found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))

    # 2. Functional Activations
    # We scan functional, filtering for things with matching class names or lowercase equivalents
    try:
      import torch.nn.functional as F

      for name, obj in inspect.getmembers(F):
        if name.startswith("_"):
          continue
        if not inspect.isfunction(obj):
          continue

        # Heuristic: Activations usually lowercase equivalents of classes found above
        # Or we liberally accept mathematical functions
        if name in [
          "relu",
          "gelu",
          "sigmoid",
          "tanh",
          "softmax",
          "log_softmax",
          "silu",
          "elu",
          "leaky_relu",
          "hardswish",
          "mish",
        ]:
          found.append(GhostInspector.inspect(obj, f"torch.nn.functional.{name}"))
    except ImportError:
      pass

    return found

  def _scan_layers(self) -> List[GhostRef]:
    """Scans for general Neural Layers (Linear, Conv, RNN, etc)."""
    if not nn:
      return []
    found = []

    # Broad scan of nn
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj) and issubclass(obj, nn.Module):
        # Filter out Losses (handled separately) and Activations (handled separately)
        if name.endswith("Loss") or name.startswith("_"):
          continue

        # We assume anything else is a Layer (or Container)
        found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

  # --- Syntax & Conversion ---

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
    """Applies manual fixes."""
    mappings = snapshot.setdefault("mappings", {})
    imports = snapshot.setdefault("imports", {})

    # Torch Module wiring
    mappings["register_buffer"] = {"api": "torch.nn.Module.register_buffer"}
    mappings["register_parameter"] = {"api": "torch.nn.Module.register_parameter"}
    mappings["state_dict"] = {"api": "torch.nn.Module.state_dict"}
    mappings["load_state_dict"] = {"api": "torch.nn.Module.load_state_dict"}
    mappings["parameters"] = {"api": "torch.nn.Module.parameters"}

    # TorchVision wiring
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

    # Ensure F.relu fallback exists
    if "relu" not in mappings:
      mappings["relu"] = {"api": "torch.nn.functional.relu"}

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier2_neural_simple"]

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier1_math": """import torch\n\ndef math_ops(x, y):\n    a = torch.abs(x)\n    b = torch.add(a, y)\n    return torch.mean(b)""",
      "tier2_neural_simple": """from torch import nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 10)\n\n    def forward(self, x):\n        x = self.linear(x)\n        return nn.functional.relu(x)""",
      "tier2_neural_cnn": """import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(1, 32, 3, 1)\n        self.fc = nn.Linear(9216, 10)\n\n    def forward(self, x):\n        x = self.conv1(x)\n        x = F.relu(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)""",
      "tier3_extras_dataloader": """import torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\ndef get_loader():\n    x = torch.randn(100, 10)\n    y = torch.randint(0, 2, (100,))\n    ds = TensorDataset(x, y)\n    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)\n    return loader""",
    }
