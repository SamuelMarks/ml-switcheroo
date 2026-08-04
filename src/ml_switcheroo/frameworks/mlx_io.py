"""MLX IO Mixin and serialization helpers.

This module provides the `MlxIOMixin` class, which implements helper methods
for MLX-specific serialization, checkpoint handling, and weight migrations.
These methods facilitate generating MLX-compatible Python scripts for saving/loading
arrays and converting weights between MLX and intermediate representations.
"""

import textwrap
from typing import List, Optional


class MlxIOMixin:
  """Mixin class providing MLX-specific IO and serialization capabilities.

  This class implements helper methods for generating import statements,
  saving and loading checkpoints, and handling weight conversion operations
  for Apple MLX framework compatible files (such as .npz, .safetensors checkpoints).
  """

  def get_serialization_imports(self) -> List[str]:
    """Returns Python import statements required for MLX IO operations.

    Returns:
        List[str]: A list of Python import statements needed to serialize or
            deserialize MLX objects.
    """
    return ["import mlx.core as mx"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates MLX-specific syntax for save and load operations.

    Args:
        op (str): The operation to perform, either "save" or "load".
        file_arg (str): The name or path of the target file as a string expression.
        object_arg (Optional[str]): The name of the MLX object to save.
            Required if op is "save". Defaults to None.

    Returns:
        str: A Python code snippet representing the mx.save or mx.load
            operation, or an empty string if the operation is unsupported.
    """
    if op == "save" and object_arg:
      return f"mx.save({file_arg}, {object_arg})"
    elif op == "load":
      return f"mx.load({file_arg})"
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns Python imports required for MLX weight migration and conversion scripts.

    Returns:
        List[str]: A list of MLX-related import statements necessary
            for running generated weight migration or conversion scripts.
    """
    return ["import mlx.core as mx"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Generates MLX-specific Python code to load checkpoint weights into a dictionary.

    The generated code handles standard .npz loading as well as default MLX load mechanisms,
    and returns a wrapped state dictionary.

    Args:
        path_var (str): The variable name containing the file path of the
            checkpoint file to be loaded.

    Returns:
        str: MLX code representing checkpoint loading logic, file format handling,
            and common state-dict structure wrapping.
    """
    return textwrap.dedent(
      f"""
            if str({path_var}).endswith(".npz"):
                loaded = mx.load({path_var})
            else:
                loaded = mx.load({path_var}) # supports safetensors implicitly usually

            # If load returns array, wrap in dict
            if isinstance(loaded, dict):
                 raw_state = loaded
            else:
                 raw_state = loaded
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Generates the MLX expression for converting a tensor/array to a NumPy array.

    Args:
        tensor_var (str): The name of the MLX array variable.

    Returns:
        str: An MLX code expression that converts the MLX array to a NumPy ndarray.
    """
    return f"np.array({tensor_var})"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Generates MLX-specific Python code to save dictionary weights.

    The generated code converts numpy arrays in the state dictionary to MLX arrays,
    and saves them in the Safetensors format.

    Args:
        state_var (str): The name of the dictionary variable containing weights.
        path_var (str): The file path variable where the MLX checkpoint should
            be saved.

    Returns:
        str: Python code that converts each NumPy array in the state dictionary
            and saves it as a Safetensors file.
    """
    return textwrap.dedent(
      f"""
            # Convert to MLX arrays if numpy
            mlx_state = {{k: mx.array(v) for k, v in {state_var}.items()}}
            mx.save_safetensors({path_var}, mlx_state)
            """
    )
