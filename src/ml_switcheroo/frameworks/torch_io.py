"""PyTorch IO Mixin."""

import textwrap
from typing import List, Optional


class TorchIOMixin:
  """Docstring."""

  def get_serialization_imports(self) -> List[str]:
    """Returns imports required for IO operations."""
    return ["import torch"]  # pragma: no cover

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates save/load syntax."""
    if op == "save" and object_arg:  # pragma: no cover
      return f"torch.save({object_arg}, {file_arg})"  # pragma: no cover
    elif op == "load":  # pragma: no cover
      return f"torch.load({file_arg})"  # pragma: no cover
    return ""  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns imports required for the generated weight migration script logic."""
    return ["import torch"]  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """Returns Python code to load a .pth file into a raw state dictionary."""
    return textwrap.dedent(  # pragma: no cover
      f"""
            # Load PyTorch checkpoint to CPU to avoid CUDA dependency
            loaded = torch.load({path_var}, map_location='cpu')

            # Unwrap common checkpoint formats
            if isinstance(loaded, dict) and 'state_dict' in loaded:
                raw_state = loaded['state_dict']
            else:
                raw_state = loaded

            if not isinstance(raw_state, dict):
                raise ValueError(f"Expected dict-like checkpoint, got {{type(loaded)}}")
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Returns expression to convert a Torch tensor variable to a NumPy array."""
    return f"{tensor_var}.detach().cpu().numpy()"  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Returns logic for converting a numpy dict and saving it as a PyTorch checkpoint."""
    return textwrap.dedent(  # pragma: no cover
      f"""
            converted_state = {{k: torch.from_numpy(v) for k, v in {state_var}.items()}}
            torch.save(converted_state, {path_var})
            """
    )
