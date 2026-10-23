"""Test suite for rdna backend macros."""

import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR, RdnaVGPR


class DummyAllocator:
  """Test helper class DummyAllocator."""

  def get_register(self, var_name: str) -> RdnaVGPR:  # noqa: F811
    """Execute get register helper logic.

    Args:
        var_name: The var name (str) for get register.

    Returns:
        Resulting `RdnaVGPR` from get register.
    """
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:  # noqa: F811
    """Execute get register helper logic.

    Args:
        var_name: The var name (str) for get scalar register.

    Returns:
        Resulting `RdnaSGPR` from get scalar register.
    """
    return RdnaSGPR(index=0)

  def allocate_temp(self) -> RdnaVGPR:  # noqa: F811
    """Execute allocate temp helper logic.

    Returns:
        Resulting `RdnaVGPR` from allocate temp.
    """
    return RdnaVGPR(index=1)

  def allocate_scalar_temp(self) -> RdnaSGPR:  # noqa: F811
    """Execute allocate temp helper logic.

    Returns:
        Resulting `RdnaSGPR` from allocate scalar temp.
    """
    return RdnaSGPR(index=1)


def test_expand_conv2d() -> None:
  """Verify expand conv2d behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_conv2d(allocator, "conv1", {"k": 3})
  assert len(nodes) > 0


def test_expand_linear() -> None:
  """Verify expand linear behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_linear(allocator, "lin1", {"in_features": 128})
  assert len(nodes) > 0
  nodes_with_bias = rdna_macros.expand_linear(allocator, "lin2", {"in_features": 128, "bias": True})
  assert len(nodes_with_bias) > len(nodes)


def test_expand_relu() -> None:
  """Verify expand relu behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_relu(allocator, "relu1", {})
  assert len(nodes) > 0


def test_expand_flatten() -> None:
  """Verify expand flatten behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_flatten(allocator, "flat1", {})
  assert len(nodes) > 0


def test_expand_reshape() -> None:
  """Verify expand reshape behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_reshape(allocator, "resh1", {})
  assert len(nodes) > 0


def test_expand_conv3d() -> None:
  """Verify expand conv3d behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_conv3d(allocator, "conv3d1", {})
  assert len(nodes) > 0


def test_expand_dropout() -> None:
  """Verify expand dropout behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_dropout(allocator, "drop1", {})
  assert len(nodes) > 0


def test_expand_variable() -> None:
  """Verify expand variable behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_variable(allocator, "var1", {})
  assert len(nodes) > 0


def test_expand_transpose() -> None:
  """Verify expand transpose behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_transpose(allocator, "trans1", {})
  assert len(nodes) > 0


def test_expand_conv_general_dilated() -> None:
  """Verify expand conv general dilated behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_conv_general_dilated(allocator, "conv_gen1", {})
  assert len(nodes) > 0


def test_expand_adam() -> None:
  """Verify expand adam behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_adam(allocator, "adam1", {})
  assert len(nodes) > 0


def test_expand_l() -> None:
  """Verify expand l behavior."""
  allocator: DummyAllocator = DummyAllocator()
  nodes = rdna_macros.expand_l(allocator, "l1", {})
  assert len(nodes) > 0
