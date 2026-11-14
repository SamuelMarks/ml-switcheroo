"""Type stubs for ``torch.nn.utils``."""

from typing import Iterable, TypeVar
from torch import Tensor
from torch.nn import Module

_M = TypeVar("_M", bound=Module)

def clip_grad_norm_(parameters: Iterable[Tensor] | Tensor, max_norm: float) -> Tensor: ...
def weight_norm(module: _M, name: str = ..., dim: int = ...) -> _M: ...
