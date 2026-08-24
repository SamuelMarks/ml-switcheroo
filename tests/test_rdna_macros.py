"""Docstring."""

from ml_switcheroo.core.compiler.backends.rdna.macros import (
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
  RegisterAllocatorProtocol,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR, RdnaSGPR


class DummyAllocator(RegisterAllocatorProtocol):
  """Docstring."""

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Docstring."""
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Docstring."""
    return RdnaSGPR(index=0)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Docstring."""
    return RdnaVGPR(index=1)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Docstring."""
    return RdnaSGPR(index=1)


def test_expand_macros():
  """Docstring."""
  allocator = DummyAllocator()

  nodes_conv2d = expand_conv2d(allocator, "conv", {"k": 3})
  assert len(nodes_conv2d) > 0

  nodes_linear = expand_linear(allocator, "lin", {"d_in": 10, "d_out": 10})
  assert len(nodes_linear) > 0

  nodes_relu = expand_relu(allocator, "relu", {})
  assert len(nodes_relu) > 0

  nodes_flatten = expand_flatten(allocator, "flat", {})
  assert len(nodes_flatten) > 0

  nodes_reshape = expand_reshape(allocator, "resh", {})
  assert len(nodes_reshape) > 0

  nodes_conv3d = expand_conv3d(allocator, "conv3d", {})
  assert len(nodes_conv3d) > 0

  nodes_dropout = expand_dropout(allocator, "drop", {})
  assert len(nodes_dropout) > 0

  nodes_var = expand_variable(allocator, "var", {})
  assert len(nodes_var) > 0

  nodes_trans = expand_transpose(allocator, "trans", {})
  assert len(nodes_trans) > 0

  nodes_conv_gen = expand_conv_general_dilated(allocator, "convgen", {})
  assert len(nodes_conv_gen) > 0

  nodes_adam = expand_adam(allocator, "adam", {})
  assert len(nodes_adam) > 0

  nodes_l = expand_l(allocator, "l", {})
  assert len(nodes_l) > 0


def test_expand_linear_bias():
  """Docstring."""
  allocator = DummyAllocator()
  nodes_linear = expand_linear(allocator, "lin_bias", {"d_in": 10, "d_out": 10, "bias": True})
  assert len(nodes_linear) > 0
