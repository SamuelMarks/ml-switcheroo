"""Module docstring."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  RegisterAllocatorProtocol,
  SassRegister,
  expand_conv2d,
  expand_linear,
  expand_mean,
  expand_flatten,
  expand_reshape,
  expand_conv3d,
  expand_variable,
  expand_transpose,
  expand_conv_general_dilated,
  expand_adam,
  expand_l,
)
from ml_switcheroo.core.compiler.frontends.sass.cst import SassNode


class MockAllocator(RegisterAllocatorProtocol):
  """Mock allocator."""

  def __init__(self) -> None:
    """Init."""
    self.counter = 0

  def get_register(self, var_name: str) -> SassRegister:
    """Get register."""
    return SassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> SassRegister:
    """Allocate temp."""
    self.counter += 1
    return SassRegister(name=f"R_TMP_{self.counter}")

  def free_register(self, name: str) -> None:
    """Free."""
    pass


def test_expand_conv2d() -> None:
  """Test expand conv2d."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_conv2d(alloc, "conv1", {"k": 3})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Conv2d" in str(n) for n in nodes)


def test_expand_linear() -> None:
  """Test expand linear."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_linear(alloc, "lin1", {"in_features": 64})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Linear" in str(n) for n in nodes)


def test_expand_mean() -> None:
  """Test expand mean."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_mean(alloc, "mean1", {"elements": 32})
  assert len(nodes) > 0
  assert any("BEGIN Mean" in str(n) for n in nodes)


def test_expand_flatten() -> None:
  """Test expand flatten."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_flatten(alloc, "flat1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Flatten" in str(n) for n in nodes)


def test_expand_reshape() -> None:
  """Test expand reshape."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_reshape(alloc, "res1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Reshape" in str(n) for n in nodes)


def test_expand_conv3d() -> None:
  """Test expand conv3d."""
  alloc = MockAllocator()
  nodes: list[SassNode] = expand_conv3d(alloc, "conv3d1", {"k": 5})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN Conv3d" in str(n) for n in nodes)


def test_expand_generic_macros() -> None:
  """Test expand generic macros."""
  alloc = MockAllocator()
  for func in [expand_variable, expand_transpose, expand_conv_general_dilated, expand_adam, expand_l]:
    nodes: list[SassNode] = func(alloc, "node1", {})  # type: ignore
    assert len(nodes) > 0


def test_expand_relu() -> None:
  """Test expand relu."""
  alloc = MockAllocator()
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_relu

  nodes: list[SassNode] = expand_relu(alloc, "relu1", {})  # type: ignore
  assert len(nodes) > 0
  assert any("BEGIN ReLU" in str(n) for n in nodes)
