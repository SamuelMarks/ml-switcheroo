"""Type stubs for ``praxis.base_layer``."""

from typing import Protocol
import jax.numpy as jnp

class HParamsProtocol(Protocol):
  """Protocol representing Praxis layer HParams configuration objects."""

class ChildLayerCallable(Protocol):
  """Callable child layer registered via ``BaseLayer.create_child``."""

  def __call__(
    self,
    *args: jnp.ndarray,
    atten_mask: jnp.ndarray | None = ...,
  ) -> jnp.ndarray: ...

class AttentionChildCallable(Protocol):
  """Callable attention child layer supporting both array and tuple unpacking."""

  def __call__(
    self,
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    atten_mask: jnp.ndarray | None = ...,
  ) -> jnp.ndarray: ...

class BaseLayer:
  """Typed stub for ``praxis.base_layer.BaseLayer``."""

  linear: ChildLayerCallable
  conv: ChildLayerCallable
  conv1: ChildLayerCallable
  conv2: ChildLayerCallable
  bn: ChildLayerCallable
  bn1: ChildLayerCallable
  bn2: ChildLayerCallable
  gn: ChildLayerCallable
  ln: ChildLayerCallable
  norm1: ChildLayerCallable
  norm2: ChildLayerCallable
  dropout: ChildLayerCallable
  emb: ChildLayerCallable
  pool: ChildLayerCallable
  lstm_cell: ChildLayerCallable
  fc1: ChildLayerCallable
  fc2: ChildLayerCallable
  ff1: ChildLayerCallable
  ff2: ChildLayerCallable
  relu: ChildLayerCallable
  relu1: ChildLayerCallable
  relu2: ChildLayerCallable
  mha: ChildLayerCallable
  attn: AttentionChildCallable

  def setup(self) -> None: ...
  def create_child(self, name: str, params: HParamsProtocol) -> None: ...
