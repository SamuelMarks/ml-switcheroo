"""Module docstring."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
  RegisterAllocatorProtocol,
  NvidiaSassRegister,
  expand_adam,
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
  expand_flatten,
  expand_l,
  expand_linear,
  expand_mean,
  expand_reshape,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode


class MockAllocator(RegisterAllocatorProtocol):
  """Mock allocator."""

  def __init__(self) -> None:
    """Init."""
    self.counter = 0

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Get register."""
    return NvidiaSassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> NvidiaSassRegister:
    """Allocate temp."""
    self.counter += 1
    return NvidiaSassRegister(name=f"R_TMP_{self.counter}")

  def free_register(self, name: str) -> None:
    """Free."""
    pass


def test_expand_conv2d() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_conv2d(alloc, "conv1", {"k": 3})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Conv2d" in str(n) for n in nodes)


def test_expand_linear() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_linear(alloc, "lin1", {"in_features": 64})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Linear" in str(n) for n in nodes)


def test_expand_mean() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_mean(alloc, "mean1", {"elements": 32})
  assert len(nodes) > 0
  assert any("BEGIN Mean" in str(n) for n in nodes)


def test_expand_flatten() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_flatten(alloc, "flat1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Flatten" in str(n) for n in nodes)


def test_expand_reshape() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_reshape(alloc, "res1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Reshape" in str(n) for n in nodes)


def test_expand_conv3d() -> None:
  """Docstring."""
  alloc = MockAllocator()
  nodes: list[NvidiaSassNode] = expand_conv3d(alloc, "conv3d1", {"k": 5})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Conv3d" in str(n) for n in nodes)


def test_expand_generic_macros() -> None:
  """Docstring."""
  alloc = MockAllocator()
  for func in [expand_variable, expand_transpose, expand_conv_general_dilated, expand_adam, expand_l]:
    nodes: list[NvidiaSassNode] = func(alloc, "node1", {})  # type: ignore
    assert len(nodes) > 0


def test_expand_relu() -> None:
  """Docstring."""
  alloc = MockAllocator()
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_relu

  nodes: list[NvidiaSassNode] = expand_relu(alloc, "relu1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN ReLU" in str(n) for n in nodes)
