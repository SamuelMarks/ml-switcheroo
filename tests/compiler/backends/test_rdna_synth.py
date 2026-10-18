"""Test suite for RDNA synthesizer edge cases."""

import typing

from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode


class MockAllocator:
  """Mock register allocator returning v0 for RDNA synthesizer tests."""

  def get_register(self, name: str) -> typing.Any:  # noqa: F811
    """Retrieve a mock VGPR register for the given variable name.

    Returns:
        Computed test result or fixture value.
    """
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR

    return RdnaVGPR(index=0)

  def allocate_temp(self) -> typing.Any:  # noqa: F811
    """Allocate a temporary mock VGPR register.

    Returns:
        Computed test result or fixture value.
    """
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR

    return RdnaVGPR(index=0)


def test_rdna_synth_raw_opcode() -> None:
  """Verifies the behavior of RDNA synth raw opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.v_add_f32", {})
  mock_semantics.resolve_variant.return_value = {"api": "v_add_f32"}
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="rdna.v_add_f32")]})
  _res = synth.from_graph(g)
  assert len(_res) == 1
  assert getattr(_res[0], "opcode", "") == "v_add_f32"


def test_rdna_synth_invalid_opcode() -> None:
  """Verifies the behavior of RDNA synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code with spaces!"}
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="BadOp")]})
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    synth.from_graph(g)


def test_rdna_macros_coverage() -> None:
  """Verify rdna macros coverage behavior."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import (
    expand_adam,
    expand_conv2d,
    expand_conv3d,
    expand_conv_general_dilated,
    expand_dropout,
    expand_flatten,
    expand_l,
    expand_linear,
    expand_relu,
    expand_reshape,
    expand_transpose,
    expand_variable,
  )

  alloc = MockAllocator()
  assert len(expand_conv2d(alloc, "n1", {"k": 3})) > 0
  assert len(expand_linear(alloc, "n1", {"in_features": 3})) > 0
  assert len(expand_relu(alloc, "n1", {})) > 0
  assert len(expand_flatten(alloc, "n1", {})) > 0
  assert len(expand_reshape(alloc, "n1", {})) > 0
  assert len(expand_conv3d(alloc, "n1", {"k": 3})) > 0
  assert len(expand_dropout(alloc, "n1", {})) > 0
  assert len(expand_variable(alloc, "n1", {})) > 0
  assert len(expand_transpose(alloc, "n1", {})) > 0
  assert len(expand_conv_general_dilated(alloc, "n1", {})) > 0
  assert len(expand_adam(alloc, "n1", {})) > 0
  assert len(expand_l(alloc, "n1", {})) > 0


def test_rdna_macros_linear_bias() -> None:
  """Verify rdna macros linear bias behavior."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_linear

  alloc = MockAllocator()
  assert len(expand_linear(alloc, "n1", {"in_features": 3, "bias": True})) > 0
