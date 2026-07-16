"""MLX IO Mixin."""

import textwrap
from typing import List, Optional


class MlxIOMixin:
  """Docstring."""

  def get_serialization_imports(self) -> List[str]:
    """Docstring."""
    return ["import mlx.core as mx"]  # pragma: no cover

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Docstring."""
    if op == "save" and object_arg:  # pragma: no cover
      return f"mx.save({file_arg}, {object_arg})"  # pragma: no cover
    elif op == "load":  # pragma: no cover
      return f"mx.load({file_arg})"  # pragma: no cover
    return ""  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """Docstring."""
    return ["import mlx.core as mx"]  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """Docstring."""
    return textwrap.dedent(  # pragma: no cover
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
    """Docstring."""
    return f"np.array({tensor_var})"  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Docstring."""
    return textwrap.dedent(  # pragma: no cover
      f"""
            # Convert to MLX arrays if numpy
            mlx_state = {{k: mx.array(v) for k, v in {state_var}.items()}}
            mx.save_safetensors({path_var}, mlx_state)
            """
    )
