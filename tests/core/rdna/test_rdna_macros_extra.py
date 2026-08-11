"""Module docstring."""

from ml_switcheroo.core.compiler.backends.rdna.macros import (
  RegisterAllocatorProtocol,
  RdnaVGPR,
  RdnaSGPR,
  expand_conv2d,
  expand_linear,
  expand_relu,
  expand_flatten,
  expand_reshape,
  expand_conv3d,
  expand_dropout,
  expand_variable,
  expand_transpose,
  expand_conv_general_dilated,
  expand_adam,
  expand_l,
)


class MockAllocator(RegisterAllocatorProtocol):
  """Mock allocator."""

  def __init__(self):
    """Init."""
    self.vc = 0
    self.sc = 0

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Get vector register."""
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Get scalar register."""
    return RdnaSGPR(index=0)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Allocate vector temp."""
    self.vc += 1
    return RdnaVGPR(index=self.vc)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocate scalar temp."""
    self.sc += 1
    return RdnaSGPR(index=self.sc)


def test_expand_all_macros():
  """Test expand all macros."""
  alloc = MockAllocator()
  funcs = [
    expand_conv2d,
    expand_linear,
    expand_relu,
    expand_flatten,
    expand_reshape,
    expand_conv3d,
    expand_dropout,
    expand_variable,
    expand_transpose,
    expand_conv_general_dilated,
    expand_adam,
    expand_l,
  ]
  for func in funcs:
    nodes = func(alloc, "node1", {})
    assert len(nodes) > 0
