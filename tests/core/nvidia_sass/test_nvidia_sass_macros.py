"""Test suite for nvidia sass macros."""

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

  def get_register(self, var_name: str) -> NvidiaSassRegister:  # noqa: F811
    """Get register.

    Args:
        var_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name=f"R_VAR_{var_name}")

  def allocate_temp(self) -> NvidiaSassRegister:  # noqa: F811
    """Allocate temp.

    Returns:
        Computed test result or fixture value.
    """
    self.counter += 1
    return NvidiaSassRegister(name=f"R_TMP_{self.counter}")

  def free_register(self, name: str) -> None:
    """Free.

    Args:
        name: Test argument or fixture.
    """
    pass


def test_expand_conv2d() -> None:
  """Verify expand conv2d behavior."""
  alloc = MockAllocator()
  nodes = expand_conv2d(alloc, "conv1", {"k": 3})
  assert len(nodes) > 0
  assert any("BEGIN Conv2d" in str(n) for n in nodes)


def test_expand_linear() -> None:
  """Verify expand linear behavior."""
  alloc = MockAllocator()
  nodes = expand_linear(alloc, "lin1", {"in_features": 64})
  assert len(nodes) > 0
  assert any("BEGIN Linear" in str(n) for n in nodes)


def test_expand_mean() -> None:
  """Verify expand mean behavior."""
  alloc = MockAllocator()
  nodes = expand_mean(alloc, "mean1", {"elements": 32})
  assert len(nodes) > 0
  assert any("BEGIN Mean" in str(n) for n in nodes)


def test_expand_flatten() -> None:
  """Verify expand flatten behavior."""
  alloc = MockAllocator()
  nodes = expand_flatten(alloc, "flat1", {})
  assert len(nodes) > 0
  assert any("BEGIN Flatten" in str(n) for n in nodes)


def test_expand_reshape() -> None:
  """Verify expand reshape behavior."""
  alloc = MockAllocator()
  nodes = expand_reshape(alloc, "res1", {})
  assert len(nodes) > 0
  assert any("BEGIN Reshape" in str(n) for n in nodes)


def test_expand_conv3d() -> None:
  """Verify expand conv3d behavior."""
  alloc = MockAllocator()
  nodes = expand_conv3d(alloc, "conv3d1", {"k": 5})
  assert len(nodes) > 0
  assert any("BEGIN Conv3d" in str(n) for n in nodes)


def test_expand_generic_macros() -> None:
  """Verify expand generic macros behavior."""
  alloc = MockAllocator()
  for func in [expand_variable, expand_transpose, expand_conv_general_dilated, expand_adam, expand_l]:
    nodes = func(alloc, "node1", {})
    assert len(nodes) > 0


def test_expand_relu() -> None:
  """Verify expand relu behavior."""
  alloc = MockAllocator()
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_relu

  nodes = expand_relu(alloc, "relu1", {})
  assert len(nodes) > 0
  assert any("BEGIN ReLU" in str(n) for n in nodes)
