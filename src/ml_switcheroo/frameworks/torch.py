"""
PyTorch Framework Adapter.

This module implements the `FrameworkAdapter` protocol for PyTorch.
It provides:

1.  **Import Abstraction**: Self-declared namespace mappings (e.g. `torch.nn` is `NEURAL`).
2.  **Semantic Definitions**: A comprehensive mapping of Abstract Operations to `torch.*` APIs.
    This moves the "Golden Set" logic out of the central hub and into this file.
3.  **Discovery**: Heuristics and logic for scanning the installed `torch` library
    to discover new operations dynamically.
4.  **IO & Device Support**: Wires up serialization (`save`/`load`) and device allocation logic.
"""

import inspect
import logging
import sys
from typing import List, Tuple, Dict, Any, Optional, Set

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
  PluginTraits,
  StandardMap,
  ImportConfig,
  StandardCategory,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.core.ghost import GhostInspector


@register_framework("torch")
class TorchAdapter:
  """
  Adapter for PyTorch (Meta).

  Handles Source and Target translation rules for PyTorch, including
  `torch.nn`, `torch.optim`, and `torch.func` (vmap/grad).
  """

  display_name: str = "PyTorch"
  inherits_from: Optional[str] = None
  # Explicitly set Priority 0 to ensure it is the Default Source
  ui_priority: int = 0

  def __init__(self) -> None:
    """
    Initializes the adapter.
    Detects if PyTorch is installed to switch between LIVE inspection
    and GHOST snapshot loading.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")

  # --- Import Abstraction (Self-Declaration) ---

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Returns the primary root import alias ('torch', 'torch')."""
    return ("torch", "torch")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """
    Defines the semantic roles of PyTorch namespaces.
    Used by the SemanticsManager to link Source imports to Target imports.
    """
    return {
      "torch": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="torch"),
      "torch.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "torch.nn.functional": ImportConfig(tier=SemanticTier.NEURAL_OPS, recommended_alias="F"),
      "torch.optim": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optim"),
      "torch.utils.data": ImportConfig(tier=SemanticTier.EXTRAS),
      "torchvision.transforms": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="T"),
    }

  # --- Discovery Configuration ---

  @property
  def search_modules(self) -> List[str]:
    """Modules to scan during `scaffold` or `sync` operations."""
    if self._mode == InitMode.GHOST:
      return []
    return [
      "torch.nn",
      "torch.linalg",
      "torch.special",
      "torch.fft",
      "torch.nn.functional",
      "torchvision.transforms",
    ]

  @property
  def unsafe_submodules(self) -> Set[str]:
    """
    Submodules that cause recursion depth errors or C-Extension crashes
    during dynamic introspection.
    """
    return {
      "_C",
      "distributed",
      "cuda",
      "backends",
      "fx",
      "masked",
      "ao",
      "quantization",
      "testing",
      "compiler",
      "contrib",
      "examples",
      "tools",
      "utils",
      "autograd",
      "onnx",
      "jit",
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Regex patterns (strings) to categorize discovered APIs."""
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

  # --- Code Generation Traits ---

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Returns the semantic tiers fully supported by this adapter."""
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def test_config(self) -> Dict[str, str]:
    """Templates used by `gen-tests` to create physical verification files."""
    return {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.detach().cpu().numpy()",
    }

  @property
  def harness_imports(self) -> List[str]:
    """No special imports needed for harness initialization."""
    return []

  def get_harness_init_code(self) -> str:
    """No special initialization helpers needed."""
    return ""

  def get_to_numpy_code(self) -> str:
    """
    Returns code to convert Torch tensors to NumPy (detach/cpu check).
    """
    return "if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()"

  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines how classes and functions are rewritten when targeting PyTorch.
    """
    return StructuralTraits(
      module_base="torch.nn.Module",
      forward_method="forward",
      requires_super_init=True,
      auto_strip_magic_args=True,  # Decoupled
      lifecycle_strip_methods=[
        "to",
        "cpu",
        "cuda",
        "detach",
        "clone",
        "requires_grad_",
        "share_memory_",
      ],
      lifecycle_warn_methods=["eval", "train"],
      impurity_methods=[
        "add_",
        "sub_",
        "mul_",
        "div_",
        "pow_",
        "zero_",
        "copy_",
        "fill_",
      ],
      jit_static_args=[],  # Torch imperative doesn't require static args annotations
      implicit_method_roots=["torch.Tensor"],  # Explicitly declare implicit method roots
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """
    Capabilities flags. PyTorch uses imperative state and eager execution.
    """
    return PluginTraits(
      has_numpy_compatible_arrays=False,  # Uses .to() not .astype()
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    """Global seed setting methods detected as impure side-effects."""
    return ["manual_seed", "seed"]

  @property
  def declared_magic_args(self) -> List[str]:
    """Torch emits no magic args, all state is implicit."""
    return []

  # --- Semantic Definitions (The Spoke) ---

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """
    The definitive mapping of Abstract Operations to PyTorch APIs.
    Moved from `standards_internal.py` to decouple the Core from Frameworks.
    """
    return {
      # --- 1. Math / Array Operations ---
      "TorchFunctional": StandardMap(api="torch.nn.functional"),
      "Abs": StandardMap(api="torch.abs"),
      "Mean": StandardMap(api="torch.mean"),
      "Sum": StandardMap(api="torch.sum"),
      "Add": StandardMap(api="torch.add"),
      "Sub": StandardMap(api="torch.sub"),
      "Mul": StandardMap(api="torch.mul"),
      "Div": StandardMap(api="torch.div"),
      "Pow": StandardMap(api="torch.pow"),
      "exp": StandardMap(api="torch.exp"),
      "log": StandardMap(api="torch.log"),
      "sqrt": StandardMap(api="torch.sqrt"),
      "square": StandardMap(api="torch.square"),
      # --- 2. Tensor Manipulation ---
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
      "size": StandardMap(api="torch.Tensor.size"),
      "OneHot": StandardMap(api="torch.nn.functional.one_hot"),
      "max": StandardMap(api="torch.max"),
      "min": StandardMap(api="torch.min"),
      # --- 3. Types (Target Mapping) ---
      "Float32": StandardMap(api="torch.float32"),
      "Float64": StandardMap(api="torch.float64"),
      "Float16": StandardMap(api="torch.float16"),
      "Int64": StandardMap(api="torch.int64"),
      "Int32": StandardMap(api="torch.int32"),
      "Int16": StandardMap(api="torch.int16"),
      "UInt8": StandardMap(api="torch.uint8"),
      "Bool": StandardMap(api="torch.bool"),
      # --- 4. Casting Logic ---
      # Note: Torch casting is usually a method call `.float()`
      # These mappings allow `CastFloat(x)` -> `x.float()` via transformation types if plugins support it,
      # otherwise they map to `torch.Tensor.float` which rewriter handles as methods.
      "CastFloat": StandardMap(api="torch.Tensor.float"),
      "CastDouble": StandardMap(api="torch.Tensor.double"),
      "CastHalf": StandardMap(api="torch.Tensor.half"),
      "CastLong": StandardMap(api="torch.Tensor.long"),
      "CastInt": StandardMap(api="torch.Tensor.int"),
      "CastShort": StandardMap(api="torch.Tensor.short"),
      "CastByte": StandardMap(api="torch.Tensor.byte"),
      "CastBool": StandardMap(api="torch.Tensor.bool"),
      "CastChar": StandardMap(api="torch.Tensor.char"),
      # --- 5. Optimization ---
      "Adam": StandardMap(api="torch.optim.Adam"),
      "SGD": StandardMap(api="torch.optim.SGD"),
      "RMSprop": StandardMap(api="torch.optim.RMSprop"),
      "StepLR": StandardMap(api="torch.optim.lr_scheduler.StepLR"),
      "CosineAnnealingLR": StandardMap(api="torch.optim.lr_scheduler.CosineAnnealingLR"),
      "ClipGradNorm": StandardMap(api="torch.nn.utils.clip_grad_norm_"),
      "step": StandardMap(api="optimizer.step"),
      "zero_grad": StandardMap(api="optimizer.zero_grad"),
      # --- 6. Neural Layers ---
      "Linear": StandardMap(api="torch.nn.Linear"),
      "Conv2d": StandardMap(api="torch.nn.Conv2d"),
      "MaxPool2d": StandardMap(api="torch.nn.MaxPool2d"),
      "MultiheadAttention": StandardMap(api="torch.nn.MultiheadAttention"),
      "Embedding": StandardMap(api="torch.nn.Embedding"),
      "Sequential": StandardMap(api="torch.nn.Sequential"),
      "BatchNorm": StandardMap(api="torch.nn.BatchNorm2d"),
      "LayerNorm": StandardMap(api="torch.nn.LayerNorm"),
      "Dropout": StandardMap(api="torch.nn.Dropout"),
      # --- 7. Activations & Loss ---
      "GELU": StandardMap(api="torch.nn.GELU"),
      "relu": StandardMap(api="torch.nn.functional.relu"),
      "softmax": StandardMap(api="torch.nn.functional.softmax"),
      "log_softmax": StandardMap(api="torch.nn.functional.log_softmax"),
      "CrossEntropyLoss": StandardMap(api="torch.nn.functional.cross_entropy"),
      "MSELoss": StandardMap(api="torch.nn.functional.mse_loss"),
      # --- 8. Vision Transforms ---
      "Resize": StandardMap(api="torchvision.transforms.Resize"),
      "Normalize": StandardMap(api="torchvision.transforms.Normalize"),
      "ToTensor": StandardMap(api="torchvision.transforms.ToTensor"),
      "CenterCrop": StandardMap(api="torchvision.transforms.CenterCrop"),
      "RandomCrop": StandardMap(api="torchvision.transforms.RandomCrop"),
      "RandomHorizontalFlip": StandardMap(api="torchvision.transforms.RandomHorizontalFlip"),
      "RandomVerticalFlip": StandardMap(api="torchvision.transforms.RandomVerticalFlip"),
      "Grayscale": StandardMap(api="torchvision.transforms.Grayscale"),
      # --- 9. State Management & Extras ---
      "no_grad": StandardMap(api="torch.no_grad"),
      "enable_grad": StandardMap(api="torch.enable_grad"),
      "register_buffer": StandardMap(api="torch.nn.Module.register_buffer"),
      "register_parameter": StandardMap(api="torch.nn.Module.register_parameter"),
      "state_dict": StandardMap(api="torch.nn.Module.state_dict"),
      "load_state_dict": StandardMap(api="torch.nn.Module.load_state_dict"),
      "parameters": StandardMap(api="torch.nn.Module.parameters"),
      "DataLoader": StandardMap(api="torch.utils.data.DataLoader"),
      # --- Wired Orphans: IO & Devices ---
      "Save": StandardMap(api="torch.save", args={"obj": "obj", "f": "f"}),
      "Load": StandardMap(api="torch.load", args={"f": "f"}),
      "Device": StandardMap(api="torch.device", requires_plugin="device_allocator"),
      "CudaAvailable": StandardMap(api="torch.cuda.is_available"),  # <--- NEW: Wired Orphan
      # --- 10. Container Logic (Reverse Mapping from Flax NNX) ---
      "Param": StandardMap(api="torch.nn.Parameter"),
      "Variable": StandardMap(api="torch.nn.Parameter", requires_plugin="nnx_param_to_torch"),
      "Cache": StandardMap(api="torch.nn.Parameter", requires_plugin="nnx_param_to_torch"),
      # --- 11. Functional Transforms ---
      "vmap": StandardMap(
        api="torch.vmap",
        args={"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"},
      ),
      "grad": StandardMap(api="torch.func.grad"),
      "jit": StandardMap(api="torch.compile"),
      "Compile": StandardMap(api="torch.compile"),
      "Synchronize": StandardMap(api="torch.cuda.synchronize"),
      "SiLU": StandardMap(api="torch.nn.functional.silu"),
      "ModuleList": StandardMap(api="torch.nn.ModuleList"),
      "TensorType": StandardMap(api="torch.Tensor"),
      "Arange": StandardMap(api="torch.arange", args={"stop": "end"}),
      "Ones": StandardMap(api="torch.ones", args={"shape": "size"}),
      "Concatenate": StandardMap(api="torch.cat", args={"axis": "dim"}),
      "Zeros": StandardMap(api="torch.zeros", args={"shape": "size"}),
      "Concatenate": StandardMap(api="torch.cat", args={"axis": "dim"}),
      "Zeros": StandardMap(api="torch.zeros", args={"shape": "size"}),
      "RandInt": StandardMap(api="torch.randint", args={"shape": "size"}),
      "Array": StandardMap(api="torch.tensor"),
      "Pad": StandardMap(api="torch.nn.functional.pad"),
      "AssertClose": StandardMap(api="torch.testing.assert_close"),
    }

  # --- Syntax Generators ---

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """
    Generates code for device creation.
    Example: "torch.device('cuda', 0)"
    """
    args = [str(device_type)]
    if device_index:
      args.append(str(device_index))
    arg_str = ", ".join(args)
    return f"torch.device({arg_str})"

  def get_device_check_syntax(self) -> str:
    """
    Returns PyTorch syntax for checking CUDA availability.
    """
    return "torch.cuda.is_available()"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """
    PyTorch uses global state-based randomness, so explicit splitting is a no-op.
    Returns 'pass' to maintain valid block syntax if injected blindly.
    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """Returns imports required for IO operations."""
    return ["import torch"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """
    Generates save/load syntax.
    """
    if op == "save" and object_arg:
      return f"torch.save({object_arg}, {file_arg})"
    elif op == "load":
      return f"torch.load({file_arg})"
    return ""

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """
    Returns the official PyTorch documentation URL.

    Handles namespaces:
    - `torch.nn.init.*` -> `https://pytorch.org/docs/stable/nn.init.html#{name}`
    - `torch.optim.*` -> `https://pytorch.org/docs/stable/optim.html#{name}` (Often grouped)
    - Default -> `https://pytorch.org/docs/stable/generated/{name}.html`
    """
    if "nn.init" in api_name:
      return f"https://pytorch.org/docs/stable/nn.init.html#{api_name}"

    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"

  @classmethod
  def get_example_code(cls) -> str:
    """Default example for documentation."""
    return cls().get_tiered_examples()["tier1_math"]

  def get_tiered_examples(self) -> Dict[str, str]:
    """
    Provides code snippets for "Wizard" or "Demo" usage.
    """
    return {
      "tier1_math": """import torch\n\ndef math_ops(x, y):\n    # Tier 1: Core Tensor Operations\n    a = torch.abs(x)\n    b = torch.add(a, y)\n    \n    # Reduction\n    return torch.mean(b)\n""",
      "tier2_neural_simple": """import torch\nimport torch.nn as nn\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 10)\n        \n    def forward(self, x):\n        x = self.fc(x)\n        return nn.functional.relu(x)\n""",
      "tier2_neural_cnn": """import torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n""",
      "tier3_extras_dataloader": """import torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\ndef create_loader(data, targets):\n    # Tier 3: Data Loader\n    ds = TensorDataset(data, targets)\n    return DataLoader(ds, batch_size=32, num_workers=4)\n""",
    }

  def convert(self, data: Any) -> Any:
    """
    Converts input data (numpy, lists) into PyTorch Tensors for verification runners.
    """
    try:
      import torch
      import numpy as np
    except ImportError:
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

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """
    Implementation of the Ghost Protocol.
    Scans the locally installed PyTorch library for API definitions corresponding
    to the requested category (Loss, Layer, etc.).
    """
    if self._mode == InitMode.GHOST:
      return self._collect_ghost(category)
    return self._collect_live(category)

  def _collect_ghost(self, category: StandardCategory) -> List[GhostRef]:
    """Loads definitions from JSON snapshot."""
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostInspector.hydrate(item) for item in raw_list]

  def _collect_live(self, category: StandardCategory) -> List[GhostRef]:
    """Introspects live torch modules."""
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
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj) and name.endswith("Loss") and name != "_Loss":
        if issubclass(obj, nn.Module):
          found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

  def _scan_optimizers(self) -> List[GhostRef]:
    if not optim:
      return []
    found = []
    for name, obj in inspect.getmembers(optim):
      if inspect.isclass(obj) and name != "Optimizer":
        try:
          if issubclass(obj, optim.Optimizer):
            found.append(GhostInspector.inspect(obj, f"torch.optim.{name}"))
        except TypeError:
          pass
    return found

  def _scan_activations(self) -> List[GhostRef]:
    found = []
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
    try:
      import torch.nn.modules.activation as activ

      for name, obj in inspect.getmembers(activ):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
          if name in known_names:
            found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    except ImportError:
      if nn:
        for name, obj in inspect.getmembers(nn):
          if name in known_names and inspect.isclass(obj):
            found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    try:
      import torch.nn.functional as F

      for name, obj in inspect.getmembers(F):
        if name.startswith("_") or not inspect.isfunction(obj):
          continue
        if name.lower() in [k.lower() for k in known_names]:
          found.append(GhostInspector.inspect(obj, f"torch.nn.functional.{name}"))
    except ImportError:
      pass
    return found

  def _scan_layers(self) -> List[GhostRef]:
    if not nn:
      return []
    found = []
    for name, obj in inspect.getmembers(nn):
      if inspect.isclass(obj) and issubclass(obj, nn.Module):
        if name.endswith("Loss") or name.startswith("_"):
          continue
        found.append(GhostInspector.inspect(obj, f"torch.nn.{name}"))
    return found

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """
    Apply manual patches to the standard mappings if necessary.
    """
    mappings = snapshot.setdefault("mappings", {})
    if "sort" in mappings:
      mappings["sort"]["output_adapter"] = "lambda x: x.values"
