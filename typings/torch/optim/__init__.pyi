"""Type stubs for ``torch.optim``."""

from typing import Iterable
from torch.nn import Parameter
from torch import Tensor

class Optimizer:
  """Typed stub for ``torch.optim.Optimizer``."""

  def zero_grad(self, set_to_none: bool = ...) -> None: ...
  def step(self) -> float | None: ...

class Adam(Optimizer, Tensor):
  """Typed stub for ``torch.optim.Adam``."""

  def __init__(self, params: Iterable[Parameter], lr: float = ...) -> None: ...
