"""Tests for RDNA macro expansions."""

from typing import List

from ml_switcheroo.core.compiler.backends.rdna.macros import (
  RegisterAllocatorProtocol,
  expand_adam,
  expand_batchnorm,
  expand_conv2d,
  expand_conv3d,
  expand_conv_general_dilated,
  expand_dropout,
  expand_flatten,
  expand_gelu,
  expand_generic_linalg,
  expand_generic_norm,
  expand_generic_reduction,
  expand_l,
  expand_layernorm,
  expand_linear,
  expand_relu,
  expand_reshape,
  expand_rmsnorm,
  expand_sigmoid,
  expand_silu,
  expand_softmax,
  expand_tanh,
  expand_transpose,
  expand_variable,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode, RdnaSGPR, RdnaVGPR


class DummyAllocator(RegisterAllocatorProtocol):
  """Dummy register allocator implementation for unit testing macros."""

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Get or allocate vector register for a variable name.

    Args:
        var_name: Name of the variable.

    Returns:
        A mock RdnaVGPR instance.
    """
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Get or allocate scalar register for a variable name.

    Args:
        var_name: Name of the variable.

    Returns:
        A mock RdnaSGPR instance.
    """
    return RdnaSGPR(index=0)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Allocate a temporary vector register.

    Returns:
        A mock temporary RdnaVGPR instance.
    """
    return RdnaVGPR(index=1)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Allocate a temporary scalar register.

    Returns:
        A mock temporary RdnaSGPR instance.
    """
    return RdnaSGPR(index=1)


def test_expand_macros() -> None:
  """Test standard procedural macro expansions."""
  allocator: DummyAllocator = DummyAllocator()

  nodes_conv2d: List[RdnaNode] = expand_conv2d(allocator, "conv", {"k": 3})
  assert len(nodes_conv2d) > 0

  nodes_linear: List[RdnaNode] = expand_linear(allocator, "lin", {"d_in": 10, "d_out": 10})
  assert len(nodes_linear) > 0

  nodes_relu: List[RdnaNode] = expand_relu(allocator, "relu", {})
  assert len(nodes_relu) > 0

  nodes_flatten: List[RdnaNode] = expand_flatten(allocator, "flat", {})
  assert len(nodes_flatten) > 0

  nodes_reshape: List[RdnaNode] = expand_reshape(allocator, "resh", {})
  assert len(nodes_reshape) > 0

  nodes_conv3d: List[RdnaNode] = expand_conv3d(allocator, "conv3d", {})
  assert len(nodes_conv3d) > 0

  nodes_dropout: List[RdnaNode] = expand_dropout(allocator, "drop", {})
  assert len(nodes_dropout) > 0

  nodes_var: List[RdnaNode] = expand_variable(allocator, "var", {})
  assert len(nodes_var) > 0

  nodes_trans: List[RdnaNode] = expand_transpose(allocator, "trans", {})
  assert len(nodes_trans) > 0

  nodes_conv_gen: List[RdnaNode] = expand_conv_general_dilated(allocator, "convgen", {})
  assert len(nodes_conv_gen) > 0

  nodes_adam: List[RdnaNode] = expand_adam(allocator, "adam", {})
  assert len(nodes_adam) > 0

  nodes_l: List[RdnaNode] = expand_l(allocator, "l", {})
  assert len(nodes_l) > 0

  nodes_ln: List[RdnaNode] = expand_layernorm(allocator, "ln", {})
  assert len(nodes_ln) > 0

  nodes_rms: List[RdnaNode] = expand_rmsnorm(allocator, "rms", {})
  assert len(nodes_rms) > 0

  nodes_bn: List[RdnaNode] = expand_batchnorm(allocator, "bn", {})
  assert len(nodes_bn) > 0

  nodes_gn: List[RdnaNode] = expand_generic_norm(allocator, "gn", {})
  assert len(nodes_gn) > 0

  nodes_gelu: List[RdnaNode] = expand_gelu(allocator, "gelu", {})
  assert len(nodes_gelu) > 0

  nodes_silu: List[RdnaNode] = expand_silu(allocator, "silu", {})
  assert len(nodes_silu) > 0

  nodes_sig: List[RdnaNode] = expand_sigmoid(allocator, "sig", {})
  assert len(nodes_sig) > 0

  nodes_tanh: List[RdnaNode] = expand_tanh(allocator, "tanh", {})
  assert len(nodes_tanh) > 0

  nodes_sm: List[RdnaNode] = expand_softmax(allocator, "sm", {})
  assert len(nodes_sm) > 0

  nodes_red: List[RdnaNode] = expand_generic_reduction(allocator, "red", {})
  assert len(nodes_red) > 0

  nodes_linalg: List[RdnaNode] = expand_generic_linalg(allocator, "linalg", {})
  assert len(nodes_linalg) > 0


def test_expand_linear_bias() -> None:
  """Test linear macro expansion with bias."""
  allocator: DummyAllocator = DummyAllocator()
  nodes_linear: List[RdnaNode] = expand_linear(allocator, "lin_bias", {"d_in": 10, "d_out": 10, "bias": True})
  assert len(nodes_linear) > 0
