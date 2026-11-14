"""Type stubs for ``torch.nn``."""

from typing import Iterator, Sequence
from torch import Tensor
from torch.nn import functional as functional
from torch.nn import utils as utils

class Parameter(Tensor):
  """Typed stub for ``torch.nn.Parameter``."""

class Module:
  """Typed stub for ``torch.nn.Module``."""

  def __init__(self) -> None: ...
  def parameters(self, recurse: bool = ...) -> Iterator[Parameter]: ...
  def train(self, mode: bool = ...) -> Module: ...
  def eval(self) -> Module: ...
  def __call__(self, *args: Tensor | None, **kwargs: Tensor | bool | None) -> Tensor: ...

class Linear(Module):
  """Typed stub for ``torch.nn.Linear``."""

  def __init__(self, in_features: int, out_features: int, bias: bool = ...) -> None: ...

class Conv2d(Module):
  """Typed stub for ``torch.nn.Conv2d``."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = ...,
    padding: int | str | tuple[int, int] = ...,
    dilation: int | tuple[int, int] = ...,
    groups: int = ...,
    bias: bool = ...,
  ) -> None: ...

class BatchNorm2d(Module):
  """Typed stub for ``torch.nn.BatchNorm2d``."""

  def __init__(self, num_features: int, eps: float = ..., momentum: float = ...) -> None: ...

class GroupNorm(Module):
  """Typed stub for ``torch.nn.GroupNorm``."""

  def __init__(self, num_groups: int, num_channels: int, eps: float = ..., affine: bool = ...) -> None: ...

class LayerNorm(Module):
  """Typed stub for ``torch.nn.LayerNorm``."""

  def __init__(self, normalized_shape: int | Sequence[int], eps: float = ...) -> None: ...

class Dropout(Module):
  """Typed stub for ``torch.nn.Dropout``."""

  def __init__(self, p: float = ..., inplace: bool = ...) -> None: ...

class Embedding(Module):
  """Typed stub for ``torch.nn.Embedding``."""

  def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...

class Flatten(Module):
  """Typed stub for ``torch.nn.Flatten``."""

  def __init__(self, start_dim: int = ..., end_dim: int = ...) -> None: ...

class MaxPool2d(Module):
  """Typed stub for ``torch.nn.MaxPool2d``."""

  def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = ...) -> None: ...

class ReLU(Module):
  """Typed stub for ``torch.nn.ReLU``."""

  def __init__(self, inplace: bool = ...) -> None: ...

class GELU(Module):
  """Typed stub for ``torch.nn.GELU``."""

  def __init__(self, approximate: str = ...) -> None: ...

class CrossEntropyLoss(Module):
  """Typed stub for ``torch.nn.CrossEntropyLoss``."""

  def __init__(self) -> None: ...

class Sequential(Module):
  """Typed stub for ``torch.nn.Sequential``."""

  def __init__(self, *args: Module) -> None: ...

class MultiheadAttention(Module):
  """Typed stub for ``torch.nn.MultiheadAttention``."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    dropout: float = ...,
    bias: bool = ...,
    batch_first: bool = ...,
  ) -> None: ...

class LSTM(Module):
  """Typed stub for ``torch.nn.LSTM``."""

  def __init__(
    self,
    input_size: int,
    hidden_size: int,
    num_layers: int = ...,
    batch_first: bool = ...,
  ) -> None: ...
