"""Test suite for the Sass Synth Extra module."""

import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from unittest.mock import MagicMock


def test_sass_synth_invalid_opcode() -> None:
  """Verifies the behavior of SASS synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code!"}
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid SASS opcode"):
    synth.from_graph(g)


def test_sass_synth_liveness_tracking() -> None:
  """Verifies the behavior of SASS synth liveness tracking."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge

  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "FADD"}
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode(id="n1", kind="Input"), LogicalNode(id="n2", kind="Add")]
  g.edges = [LogicalEdge(source="n1", target="n2")]
  assert synth.allocator._liveness_map == {}
  synth.from_graph(g)
  assert synth.allocator._liveness_map["n1"] == 0
  assert len(synth.allocator._free_pool) == 254


def test_sass_macro_mean() -> None:
  """Verifies the behavior of SASS mean macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_mean
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_mean(alloc, "n1", {"elements": 10})
  assert any(n.opcode == "FADD" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "FMUL" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_relu() -> None:
  """Verifies the behavior of SASS relu macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_relu
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_relu(alloc, "n1", {})
  assert any(n.opcode == "FMAX" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_flatten() -> None:
  """Verifies the behavior of SASS flatten macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_flatten
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_flatten(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_reshape() -> None:
  """Verifies the behavior of SASS reshape macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_reshape
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_reshape(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_conv3d() -> None:
  """Verifies the behavior of SASS conv3d macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv3d
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_conv3d(alloc, "n1", {"k": 3})
  assert any(n.opcode == "FFMA" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "IMAD" for n in nodes if hasattr(n, "opcode"))
