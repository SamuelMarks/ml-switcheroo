"""
Framework Adapter Registry for Input Fuzzing and Output Normalization.

This module provides the interface and registration mechanism for converting
NumPy arrays into framework-specific tensors (e.g., Torch, JAX, TensorFlow, MLX)
AND converting those tensors back into NumPy arrays for verification.

Refactor:
    Adapters can now implement `get_import_stmts`, `get_creation_syntax`, and
    `get_numpy_conversion_syntax` to act as TemplateProviders for the Registry Sync logic.
"""

from typing import Any, Dict, Protocol, Type, Optional
import numpy as np


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.
  Implementations must provide a mechanism to convert a NumPy array
  into the target framework's native tensor format.
  """

  def convert(self, data: Any) -> Any:
    """
    Converts data between formats.
    """
    ...


class TorchAdapter:
  """Default adapter for PyTorch."""

  def convert(self, data: Any) -> Any:
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

  # --- Template Provider Protocol ---
  @classmethod
  def get_import_stmts(cls) -> str:
    return "import torch"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"torch.from_numpy({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"{var_name}.detach().cpu().numpy()"


class JaxAdapter:
  """Default adapter for JAX."""

  def convert(self, data: Any) -> Any:
    try:
      import jax.numpy as jnp
    except ImportError:
      return data

    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return jnp.array(data)
    return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import jax\nimport jax.numpy as jnp"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"jnp.array({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"np.array({var_name})"


class TensorFlowAdapter:
  """Adapter for TensorFlow."""

  def convert(self, data: Any) -> Any:
    try:
      import tensorflow as tf
    except ImportError:
      return data

    try:
      return tf.convert_to_tensor(data)
    except (ValueError, TypeError, Exception):
      return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import tensorflow as tf"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"tf.convert_to_tensor({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"{var_name}.numpy()"


class MLXAdapter:
  """Adapter for Apple MLX."""

  def convert(self, data: Any) -> Any:
    try:
      import mlx.core as mx
    except ImportError:
      return data

    if isinstance(data, (np.ndarray, list, tuple, np.generic)):
      return mx.array(data)
    return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import mlx.core as mx"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return f"mx.array({var_name})"

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return f"np.array({var_name})"


class NumpyAdapter:
  """
  Universal Adapter to normalize any framework tensor back to a NumPy array.
  """

  def convert(self, data: Any) -> Any:
    # 1. Recursive Unwrapping
    if isinstance(data, (list, tuple)):
      return type(data)(self.convert(x) for x in data)

    if isinstance(data, dict):
      return {k: self.convert(v) for k, v in data.items()}

    # 2. Framework Specific Attributes
    if hasattr(data, "detach"):
      try:
        return data.detach().cpu().numpy()
      except Exception:
        pass

    if hasattr(data, "numpy"):
      try:
        val = data.numpy()
        return val
      except Exception:
        pass

    if hasattr(data, "__array__"):
      try:
        return np.array(data)
      except Exception:
        pass

    return data

  @classmethod
  def get_import_stmts(cls) -> str:
    return "import numpy as np"

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    return var_name

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    return var_name


# --- Registry ---

_ADAPTER_REGISTRY: Dict[str, Type[FrameworkAdapter]] = {
  "torch": TorchAdapter,
  "jax": JaxAdapter,
  "numpy": NumpyAdapter,
  "tensorflow": TensorFlowAdapter,
  "mlx": MLXAdapter,
}


def register_adapter(framework: str, adapter_cls: Type[FrameworkAdapter]) -> None:
  """
  Registers a new framework adapter.

  Args:
      framework: The canonical name of the framework (e.g., 'tensorflow').
      adapter_cls: A class implementing the convert method.
  """
  _ADAPTER_REGISTRY[framework] = adapter_cls


def get_adapter(framework: str) -> Optional[FrameworkAdapter]:
  """
  Retrieves an instantiated adapter for the requested framework.

  Args:
      framework: The framework name.

  Returns:
      An instance of the adapter, or None if not registered.
  """
  adapter_cls = _ADAPTER_REGISTRY.get(framework)
  if adapter_cls:
    return adapter_cls()
  return None
