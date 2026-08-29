"""Test module."""

from typing import List

import ml_switcheroo.core.compiler.backends.rdna.macros as rdna_macros
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaNode, RdnaSGPR, RdnaVGPR


class DummyAllocator:
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


def test_expand_conv2d() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv2d(allocator, "conv1", {"k": 3})
  assert len(nodes) > 0


def test_expand_linear() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_linear(allocator, "lin1", {"in_features": 128})
  assert len(nodes) > 0
  nodes_with_bias: List[RdnaNode] = rdna_macros.expand_linear(allocator, "lin2", {"in_features": 128, "bias": True})
  assert len(nodes_with_bias) > len(nodes)


def test_expand_relu() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_relu(allocator, "relu1", {})
  assert len(nodes) > 0


def test_expand_flatten() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_flatten(allocator, "flat1", {})
  assert len(nodes) > 0


def test_expand_reshape() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_reshape(allocator, "resh1", {})
  assert len(nodes) > 0


def test_expand_conv3d() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv3d(allocator, "conv3d1", {})
  assert len(nodes) > 0


def test_expand_dropout() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_dropout(allocator, "drop1", {})
  assert len(nodes) > 0


def test_expand_variable() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_variable(allocator, "var1", {})
  assert len(nodes) > 0


def test_expand_transpose() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_transpose(allocator, "trans1", {})
  assert len(nodes) > 0


def test_expand_conv_general_dilated() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_conv_general_dilated(allocator, "conv_gen1", {})
  assert len(nodes) > 0


def test_expand_adam() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_adam(allocator, "adam1", {})
  assert len(nodes) > 0


def test_expand_l() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  nodes: List[RdnaNode] = rdna_macros.expand_l(allocator, "l1", {})
  assert len(nodes) > 0
