"""Type stubs for ``flax.nnx``."""

from typing import Callable, Generic, Sequence, TypeVar
import jax
import jax.numpy as jnp

_ArrayLike = jax.Array | jnp.ndarray
_T = TypeVar("_T")

class Rngs:
  """Typed stub for ``flax.nnx.Rngs``."""

  def __init__(self, default: int = ..., *, params: int = ..., dropout: int = ...) -> None: ...

class Param(Generic[_T]):
  """Typed stub for ``flax.nnx.Param``."""

  value: _T
  def __init__(self, value: _T) -> None: ...

class Module:
  """Typed stub for ``flax.nnx.Module``."""

class Linear(Module):
  """Typed stub for ``flax.nnx.Linear``."""

  kernel: Param[jnp.ndarray]
  bias: Param[jnp.ndarray]
  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = ...,
    rngs: Rngs | None = ...,
  ) -> None: ...
  def __call__(self, x: _ArrayLike) -> jnp.ndarray: ...

class Conv(Module):
  """Typed stub for ``flax.nnx.Conv``."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | Sequence[int],
    *,
    strides: int | Sequence[int] = ...,
    padding: str | Sequence[tuple[int, int]] = ...,
    rngs: Rngs | None = ...,
  ) -> None: ...
  def __call__(self, x: _ArrayLike) -> jnp.ndarray: ...

class BatchNorm(Module):
  """Typed stub for ``flax.nnx.BatchNorm``."""

  def __init__(self, num_features: int, *, rngs: Rngs | None = ...) -> None: ...
  def __call__(self, x: _ArrayLike, *, use_running_average: bool = ...) -> jnp.ndarray: ...

class LayerNorm(Module):
  """Typed stub for ``flax.nnx.LayerNorm``."""

  def __init__(self, num_features: int, *, rngs: Rngs | None = ...) -> None: ...
  def __call__(self, x: _ArrayLike) -> jnp.ndarray: ...

class GroupNorm(Module):
  """Typed stub for ``flax.nnx.GroupNorm``."""

  def __init__(
    self,
    num_features: int = ...,
    *,
    num_groups: int = ...,
    rngs: Rngs | _ArrayLike | None = ...,
  ) -> None: ...
  def __call__(self, x: _ArrayLike) -> jnp.ndarray: ...

class Dropout(Module):
  """Typed stub for ``flax.nnx.Dropout``."""

  def __init__(self, rate: float, *, rngs: Rngs | None = ...) -> None: ...
  def __call__(self, x: _ArrayLike, *, deterministic: bool = ...) -> jnp.ndarray: ...

class Embed(Module):
  """Typed stub for ``flax.nnx.Embed``."""

  def __init__(self, num_embeddings: int, features: int, *, rngs: Rngs | None = ...) -> None: ...
  def __call__(self, inputs: _ArrayLike) -> jnp.ndarray: ...

class MultiHeadAttention(Module):
  """Typed stub for ``flax.nnx.MultiHeadAttention``."""

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    *,
    qkv_features: int | None = ...,
    out_features: int | None = ...,
    dropout_rate: float = ...,
    decode: bool = ...,
    rngs: Rngs | None = ...,
  ) -> None: ...
  def __call__(
    self,
    inputs_q: _ArrayLike,
    inputs_k: _ArrayLike | None = ...,
    inputs_v: _ArrayLike | None = ...,
    *,
    mask: _ArrayLike | None = ...,
    deterministic: bool | None = ...,
  ) -> jnp.ndarray: ...

class Sequential(Module):
  """Typed stub for ``flax.nnx.Sequential``."""

  def __init__(self, *fns: Module | Callable[[_ArrayLike], _ArrayLike]) -> None: ...
  def __call__(self, x: _ArrayLike) -> jnp.ndarray: ...

class Optimizer:
  """Typed stub for ``flax.nnx.Optimizer``."""

  def __init__(self, model: Module, tx: _ArrayLike) -> None: ...
  def update(self, grads: _ArrayLike) -> None: ...

def value_and_grad(
  f: Callable[[_ArrayLike], _ArrayLike],
) -> Callable[[Module | _ArrayLike], tuple[jnp.ndarray, jnp.ndarray]]: ...
