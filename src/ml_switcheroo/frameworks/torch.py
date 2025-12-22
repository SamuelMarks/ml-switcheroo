"""
PyTorch Adapter with Dynamic Introspection.

This adapter serves as the primary "Source of Truth" for many Deep Learning
standards (Layers, Optimizers).

Refactor: Now includes full distributed definitions for PyTorch mappings,
migrated from the core standards file.
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
      "torchvision": {"root": "torchvision", "alias": None},
      "torchvision.transforms": {"root": "torchvision", "sub": "transforms", "alias": "T"},
      "flax.nnx": {"root": "torch", "sub": "nn", "alias": "nn"},
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
  def test_config(self) -> Dict[str, str]:
    """Test templates for Torch."""
    return {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.detach().cpu().numpy()",
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
    Comprehensive distributed definitions for PyTorch mappings.
    Moved from standards_internal.py.
    """
    return {
      "TorchFunctional": StandardMap(api="torch.nn.functional"),
      # Optimization
      "Adam": StandardMap(api="torch.optim.Adam"),
      "SGD": StandardMap(api="torch.optim.SGD"),
      "RMSprop": StandardMap(api="torch.optim.RMSprop"),
      "StepLR": StandardMap(api="torch.optim.lr_scheduler.StepLR"),
      "CosineAnnealingLR": StandardMap(api="torch.optim.lr_scheduler.CosineAnnealingLR"),
      "ClipGradNorm": StandardMap(api="torch.nn.utils.clip_grad_norm_"),
      "step": StandardMap(api="optimizer.step"),
      "zero_grad": StandardMap(api="optimizer.zero_grad"),
      # Array / Math
      "randn": StandardMap(api="torch.randn"),
      "Clamp": StandardMap(api="torch.clamp"),
      "Gather": StandardMap(api="torch.gather"),
      "Scatter": StandardMap(api="torch.Tensor.scatter_"),
      "Flatten": StandardMap(api="torch.flatten"),
      "Reshape": StandardMap(api="torch.reshape"),
      "View": StandardMap(api="torch.Tensor.view"),
      "Squeeze": StandardMap(api="torch.squeeze"),
      "Unsqueeze": StandardMap(api="torch.unsqueeze"),
      "TopK": StandardMap(api="torch.topk"),
      "ArgMax": StandardMap(api="torch.argmax"),
      "ArgMin": StandardMap(api="torch.argmin"),
      "Pad": StandardMap(api="torch.nn.functional.pad"),
      "Einsum": StandardMap(api="torch.einsum"),
      "permute_dims": StandardMap(api="torch.permute"),
      "Abs": StandardMap(api="torch.abs"),
      "Mean": StandardMap(api="torch.mean"),
      "Sum": StandardMap(api="torch.sum"),
      # Casting
      "CastFloat": StandardMap(api="torch.Tensor.float"),
      "CastDouble": StandardMap(api="torch.Tensor.double"),
      "CastHalf": StandardMap(api="torch.Tensor.half"),
      "CastLong": StandardMap(api="torch.Tensor.long"),
      "CastInt": StandardMap(api="torch.Tensor.int"),
      "CastBool": StandardMap(api="torch.Tensor.bool"),
      "size": StandardMap(api="torch.Tensor.size"),
      # Neural Layers
      "MultiheadAttention": StandardMap(api="torch.nn.MultiheadAttention"),
      "Embedding": StandardMap(api="torch.nn.Embedding"),
      "Linear": StandardMap(api="torch.nn.Linear"),
      "Sequential": StandardMap(api="torch.nn.Sequential"),
      "BatchNorm": StandardMap(api="torch.nn.BatchNorm2d"),
      "LayerNorm": StandardMap(api="torch.nn.LayerNorm"),
      "GELU": StandardMap(api="torch.nn.GELU"),
      "OneHot": StandardMap(api="torch.nn.functional.one_hot"),
      "CrossEntropyLoss": StandardMap(api="torch.nn.functional.cross_entropy"),
      "MSELoss": StandardMap(api="torch.nn.functional.mse_loss"),
      "Conv2d": StandardMap(api="torch.nn.Conv2d"),
      "relu": StandardMap(api="torch.nn.functional.relu"),
      "MaxPool2d": StandardMap(api="torch.nn.MaxPool2d"),
      # Vision
      "Resize": StandardMap(api="torchvision.transforms.Resize"),
      "Normalize": StandardMap(api="torchvision.transforms.Normalize"),
      "ToTensor": StandardMap(api="torchvision.transforms.ToTensor"),
      "CenterCrop": StandardMap(api="torchvision.transforms.CenterCrop"),
      "RandomCrop": StandardMap(api="torchvision.transforms.RandomCrop"),
      "RandomHorizontalFlip": StandardMap(api="torchvision.transforms.RandomHorizontalFlip"),
      "RandomVerticalFlip": StandardMap(api="torchvision.transforms.RandomVerticalFlip"),
      "Grayscale": StandardMap(api="torchvision.transforms.Grayscale"),
      # State & Extras
      "no_grad": StandardMap(api="torch.no_grad"),
      "enable_grad": StandardMap(api="torch.enable_grad"),
      "register_buffer": StandardMap(api="torch.nn.Module.register_buffer"),
      "register_parameter": StandardMap(api="torch.nn.Module.register_parameter"),
      "state_dict": StandardMap(api="torch.nn.Module.state_dict"),
      "load_state_dict": StandardMap(api="torch.nn.Module.load_state_dict"),
      "parameters": StandardMap(api="torch.nn.Module.parameters"),
      "DataLoader": StandardMap(api="torch.utils.data.DataLoader"),
      "Param": StandardMap(api="torch.nn.Parameter"),
      "Variable": StandardMap(api="torch.nn.Parameter", requires_plugin="nnx_param_to_torch"),
      "Cache": StandardMap(api="torch.nn.Parameter", args={}, requires_plugin="nnx_param_to_torch"),
      # Functional Transforms
      "vmap": StandardMap(api="torch.vmap", args={"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}),
      "grad": StandardMap(api="torch.func.grad"),
      "jit": StandardMap(api="torch.compile"),
      "Compile": StandardMap(api="torch.compile"),
      "Synchronize": StandardMap(api="torch.cuda.synchronize"),
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
    try:
      import torch.nn.functional as F

      for name, obj in inspect.getmembers(F):
        if name.startswith("_"):
          continue
        if not inspect.isfunction(obj):
          continue

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

  @classmethod
  def get_example_code(cls) -> str:
    return cls().get_tiered_examples()["tier1_math"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Returns PyTorch idiomatic examples used for validity testing.
    """
    return {
      "tier1_math": """import torch

def math_ops(x, y): 
    # Tier 1: Core Tensor Operations
    a = torch.abs(x) 
    b = torch.add(a, y) 
    
    # Reduction
    return torch.mean(b) 
""",
      "tier2_neural_simple": """import torch
import torch.nn as nn

class Net(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.fc = nn.Linear(10, 10) 
        
    def forward(self, x): 
        x = self.fc(x) 
        return nn.functional.relu(x) 
""",
      "tier2_neural_cnn": """import torch
import torch.nn as nn

class ConvNet(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv = nn.Conv2d(1, 32, 3) 
        self.fc = nn.Linear(32 * 26 * 26, 10) 

    def forward(self, x): 
        x = self.conv(x) 
        x = torch.flatten(x, 1) 
        return self.fc(x) 
""",
      "tier3_extras_dataloader": """import torch
from torch.utils.data import DataLoader, TensorDataset

def create_loader(data, targets): 
    # Tier 3: Data Loader
    ds = TensorDataset(data, targets) 
    return DataLoader(ds, batch_size=32, num_workers=4) 
""",
    }

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

    if "sort" in mappings:
      mappings["sort"]["output_adapter"] = "lambda x: x.values"
