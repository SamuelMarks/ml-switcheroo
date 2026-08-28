"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaVGPR,
  RdnaSGPR,
  RdnaNode,
)
import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
from typing import List


class DummyAllocator:
  """Test element."""

  def get_vector_register(self, var_name: str) -> RdnaVGPR:
    """Test element."""
    return RdnaVGPR(index=0)

  def get_scalar_register(self, var_name: str) -> RdnaSGPR:
    """Test element."""
    return RdnaSGPR(index=0)

  def allocate_vector_temp(self) -> RdnaVGPR:
    """Test element."""
    return RdnaVGPR(index=1)

  def allocate_scalar_temp(self) -> RdnaSGPR:
    """Test element."""
    return RdnaSGPR(index=1)


def test_expand_conv2d() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv2d(allocator, "conv1", {"k": 3})
  assert len(nodes) > 0


def test_expand_linear() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_linear(allocator, "lin1", {"in_features": 128})
  assert len(nodes) > 0
  nodes_with_bias: List[RdnaNode] = rdna_macros.expand_linear(allocator, "lin2", {"in_features": 128, "bias": True})
  assert len(nodes_with_bias) > len(nodes)


def test_expand_relu() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_relu(allocator, "relu1", {})
  assert len(nodes) > 0


def test_expand_flatten() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_flatten(allocator, "flat1", {})
  assert len(nodes) > 0


def test_expand_reshape() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_reshape(allocator, "resh1", {})
  assert len(nodes) > 0


def test_expand_conv3d() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv3d(allocator, "conv3d1", {})
  assert len(nodes) > 0


def test_expand_dropout() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_dropout(allocator, "drop1", {})
  assert len(nodes) > 0


def test_expand_variable() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_variable(allocator, "var1", {})
  assert len(nodes) > 0


def test_expand_transpose() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_transpose(allocator, "trans1", {})
  assert len(nodes) > 0


def test_expand_conv_general_dilated() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv_general_dilated(allocator, "conv_gen1", {})
  assert len(nodes) > 0


def test_expand_adam() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_adam(allocator, "adam1", {})
  assert len(nodes) > 0


def test_expand_l() -> None:
  """Test element."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_l(allocator, "l1", {})
  assert len(nodes) > 0
