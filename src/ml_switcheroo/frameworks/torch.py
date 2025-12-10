import numpy as np
from typing import List, Tuple, Optional, Dict
from .base import register_framework, StructuralTraits


@register_framework("torch")
class TorchAdapter:
  """Adapter for PyTorch."""

  display_name: str = "PyTorch"
  inherits_from: Optional[str] = None
  ui_priority: int = 0  # Highest Priority

  @property
  def search_modules(self) -> List[str]:
    return ["torch", "torch.nn", "torch.linalg", "torch.special", "torch.fft"]

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("torch", "torch")

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {
      "neural": [r"\.nn\.", r"\.modules\.", r"\.layers\.", r"Module$"],
      "extras": [r"\.utils\.", r"\.hub\.", r"\.distributed\.", r"\.autograd\.", r"save$", r"load$", r"seed$"],
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
      # Define imperative in-place methods that violate functional purity
      impurity_methods=["add_", "sub_", "mul_", "div_", "pow_", "zero_", "copy_", "fill_"],
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    return ["manual_seed", "seed"]

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    args = [str(device_type)]
    if device_index:
      args.append(str(device_index))
    arg_str = ", ".join(args)
    return f"torch.device({arg_str})"

  def get_serialization_imports(self) -> List[str]:
    return ["import torch"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
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

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import torch"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"torch.from_numpy({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"{var_name}.detach().cpu().numpy()"

  @classmethod
  def get_example_code(cls) -> str:
    return (
      "import torch.nn as nn\n\n"
      "class Simple(nn.Module):\n"
      "    def __init__(self):\n"
      "        super().__init__()\n"
      "        self.l = nn.Linear(10, 10)\n\n"
      "    def forward(self, x):\n"
      "        return self.l(x)"
    )
