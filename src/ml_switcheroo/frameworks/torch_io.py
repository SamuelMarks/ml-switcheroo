"""PyTorch IO Mixin and serialization helpers."""

import textwrap
from typing import List, Optional


class TorchIOMixin:
  """Mixin class providing PyTorch-specific IO and serialization capabilities.

  This class implements helper methods for generating import statements,
  saving and loading checkpoints, and handling weight conversion operations
  for PyTorch-compatible files (such as .pth, .pt checkpoints).
  """

  def get_serialization_imports(self) -> List[str]:
    """Returns Python import statements required for PyTorch IO operations.

    Returns:
        List[str]: A list of Python import statements needed to serialize or
            deserialize PyTorch objects.
    """
    return ["import torch"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates PyTorch-specific syntax for save and load operations.

    Args:
        op (str): The operation to perform, either "save" or "load".
        file_arg (str): The name or path of the target file as a string expression.
        object_arg (Optional[str]): The name of the PyTorch object to save.
            Required if op is "save". Defaults to None.

    Returns:
        str: A Python code snippet representing the torch.save or torch.load
            operation, or an empty string if the operation is unsupported.
    """
    if op == "save" and object_arg:
      return f"torch.save({object_arg}, {file_arg})"
    elif op == "load":
      return f"torch.load({file_arg})"
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns Python imports required for weight migration script logic.

    Returns:
        List[str]: A list of PyTorch-related import statements necessary
            for running generated weight migration or conversion scripts.
    """
    return ["import torch"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Generates PyTorch-specific python code to load checkpoint weights into a dictionary.

    Args:
        path_var (str): The variable name containing the file path of the
            checkpoint (.pth/.pt) file to be loaded.

    Returns:
        str: PyTorch code representing checkpoint loading logic, CPU mapping to avoid
            CUDA dependencies, and common state-dict structure unwrapping.
    """
    return textwrap.dedent(
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
    """Generates the PyTorch expression for converting a tensor to a NumPy array.

    Args:
        tensor_var (str): The name of the PyTorch tensor variable.

    Returns:
        str: A PyTorch code expression that detaches the tensor, moves it to CPU,
            and converts it to a NumPy ndarray.
    """
    return f"{tensor_var}.detach().cpu().numpy()"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generates PyTorch-specific python code to save dictionary weights.

    Args:
        state_var (str): The name of the dictionary variable containing NumPy arrays.
        path_var (str): The file path variable where the PyTorch checkpoint should
            be saved.

    Returns:
        str: Python code that converts each NumPy array in the state dictionary
            to a PyTorch tensor, and saves the resulting dictionary using torch.save.
    """
    return textwrap.dedent(
      f"""
            converted_state = {{k: torch.from_numpy(v) for k, v in {state_var}.items()}}
            torch.save(converted_state, {path_var})
            """
    )
