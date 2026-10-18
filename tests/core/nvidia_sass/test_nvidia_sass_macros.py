"""Test suite for nvidia sass macros."""

import typing

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
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


class MockAllocator:
  """Mock allocator."""

  def __init__(self) -> None:
    """Init."""
    self.counter = 0

  def get_register(self, var_name: typing.Any) -> typing.Any:  # noqa: F811
    """Get register.

    Args:
        var_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> typing.Any:  # noqa: F811
    """Allocate temp.

    Returns:
        Computed test result or fixture value.
    """
    self.counter += 1
    return NvidiaSassRegister(name=f"R_TMP_{self.counter}")

  def free_register(self, name: typing.Any) -> None:
    """Free.

    Args:
        name: Test argument or fixture.
    """
    pass


def test_expand_conv2d() -> None:
  """Verify expand conv2d behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_conv2d(alloc, "conv1", {"k": 3})
  assert len(nodes) > 0
  assert any("BEGIN Conv2d" in str(n) for n in nodes)


def test_expand_linear() -> None:
  """Verify expand linear behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_linear(alloc, "lin1", {"in_features": 64})
  assert len(nodes) > 0
  assert any("BEGIN Linear" in str(n) for n in nodes)


def test_expand_mean() -> None:
  """Verify expand mean behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_mean(alloc, "mean1", {"elements": 32})
  assert len(nodes) > 0
  assert any("BEGIN Mean" in str(n) for n in nodes)


def test_expand_flatten() -> None:
  """Verify expand flatten behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_flatten(alloc, "flat1", {})
  assert len(nodes) > 0
  assert any("BEGIN Flatten" in str(n) for n in nodes)


def test_expand_reshape() -> None:
  """Verify expand reshape behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_reshape(alloc, "res1", {})
  assert len(nodes) > 0
  assert any("BEGIN Reshape" in str(n) for n in nodes)


def test_expand_conv3d() -> None:
  """Verify expand conv3d behavior."""
  alloc: typing.Any = MockAllocator()
  nodes: typing.Any = expand_conv3d(alloc, "conv3d1", {"k": 5})
  assert len(nodes) > 0
  assert any("BEGIN Conv3d" in str(n) for n in nodes)


def test_expand_generic_macros() -> None:
  """Verify expand generic macros behavior."""
  alloc: typing.Any = MockAllocator()
  for func in [expand_variable, expand_transpose, expand_conv_general_dilated, expand_adam, expand_l]:
    nodes: typing.Any = func(alloc, "node1", {})
    assert len(nodes) > 0


def test_expand_relu() -> None:
  """Verify expand relu behavior."""
  alloc: typing.Any = MockAllocator()
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_relu

  nodes: typing.Any = expand_relu(alloc, "relu1", {})
  assert len(nodes) > 0
  assert any("BEGIN ReLU" in str(n) for n in nodes)
