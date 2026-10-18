"""Test suite for rdna backend synthesizer."""

import typing

from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import MAX_SGPR, RdnaBackend, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class MockSemanticsManager:
  """Test helper class MockSemanticsManager."""

  def get_definition(self, kind: typing.Any) -> typing.Any:
    """Execute get definition helper logic.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if kind == "known_op":
      return ["add"]
    elif kind == "macro_op":
      return ["relu"]
    elif kind == "macro_suffix_op":
      return ["some.relu"]
    return None

  def resolve_variant(self, abstract_id: typing.Any, backend: typing.Any) -> typing.Any:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: Test argument or fixture.
        backend: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if abstract_id == "add":
      return {"api": "v_add_f32"}
    return None


def test_register_allocator() -> None:
  """Verify register allocator behavior."""
  allocator: typing.Any = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))

  v1: typing.Any = allocator.get_register("x")
  v2: typing.Any = allocator.get_register("x")
  assert v1.index == v2.index

  s1: typing.Any = allocator.get_register("y")
  s2: typing.Any = allocator.get_register("y")
  assert s1.index == s2.index

  v_temp: typing.Any = allocator.allocate_temp()
  assert isinstance(v_temp, RdnaVGPR)
  assert v_temp.index != v1.index

  s_temp: typing.Any = allocator.allocate_temp()
  assert isinstance(s_temp, RdnaVGPR)
  assert s_temp.index != s1.index

  allocator.reset()
  assert len(allocator._free_pool) == 256


def test_register_allocator_overflow() -> None:
  """Verify register allocator overflow behavior."""
  allocator: typing.Any = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  allocator._free_pool = []
  with pytest.raises(ValueError, match="overflow"):
    allocator.get_register("new_var")

  allocator._next_sgpr = MAX_SGPR
  with pytest.raises(ValueError, match="overflow"):
    allocator.get_register("new_var_s")


def test_synthesizer_from_graph() -> None:
  """Verify synthesizer from graph behavior."""
  semantics: typing.Any = MockSemanticsManager()
  synthesizer: typing.Any = RdnaSynthesizer(semantics)

  n_in: typing.Any = LogicalNode(id="in1", op_type="Input", attributes={"name": "input_x"})
  n_op1: typing.Any = LogicalNode(id="op1", op_type="known_op", inputs=["in1"])
  n_op2: typing.Any = LogicalNode(id="op2", op_type="unknown_op", inputs=["op1"])
  n_out: typing.Any = LogicalNode(id="out1", op_type="Output", inputs=["op2"])
  n_macro: typing.Any = LogicalNode(id="op_macro", op_type="macro_op")
  n_macro_suffix: typing.Any = LogicalNode(id="op_macro_s", op_type="macro_suffix_op")

  graph: typing.Any = LogicalGraph(
    nodes={
      "in1": n_in,
      "op1": n_op1,
      "op2": n_op2,
      "out1": n_out,
      "op_macro": n_macro,
      "op_macro_s": n_macro_suffix,
    }
  )

  # We also test a node that resolves to an abstract_id with a variant, but no 'api'
  semantics.resolve_variant = lambda aid, b: {"api": "v_add_f32"} if aid == "add" else {}

  nodes: typing.Any = synthesizer.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Verify synthesizer to python behavior."""
  semantics: typing.Any = MockSemanticsManager()
  synthesizer: typing.Any = RdnaSynthesizer(semantics)

  inst1: typing.Any = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]
  )
  inst2: typing.Any = RdnaInstruction(opcode="store_dword", operands=[RdnaVGPR(index=0), RdnaImmediate(value=4)])
  inst3: typing.Any = RdnaInstruction(opcode="s_waitcnt", operands=[])
  inst4: typing.Any = RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabel(name="L1")])
  inst5: typing.Any = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=1.5, is_hex=False)]
  )
  inst6: typing.Any = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=0x10, is_hex=True)]
  )
  inst7: typing.Any = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "some_string"])
  inst8: typing.Any = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "array[0]"])
  inst9: typing.Any = RdnaInstruction(opcode="v_add_f32", operands=["bad_dest", "array[0]"])  # Hit 295->301
  label1: typing.Any = RdnaLabel(name="L1")
  comment: typing.Any = RdnaComment(text="; Comment")

  cst_module: typing.Any = synthesizer.to_python(
    [inst1, inst2, inst3, inst4, inst5, inst6, inst7, inst8, inst9, label1, comment]
  )
  assert isinstance(cst_module, cst.Module)


def test_rdna_backend() -> None:
  """Verify rdna backend behavior."""
  semantics: typing.Any = MockSemanticsManager()
  backend: typing.Any = RdnaBackend(semantics)

  graph: typing.Any = LogicalGraph()
  n_in: typing.Any = LogicalNode(id="in1", op_type="Input")
  graph.add_node(n_in)

  result: typing.Any = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in result


def test_rdna_backend_default_semantics() -> None:
  """Verify rdna backend default semantics behavior."""
  backend: typing.Any = RdnaBackend()
  assert backend.synthesizer.semantics is not None


# --- Merged from test_rdna_backend_synthesizer_extra.py ---


class DummySemantics:
  """Test helper class DummySemantics."""

  def resolve_variant(self, abstract_id: typing.Any, flavor: typing.Any) -> typing.Any:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: Test argument or fixture.
        flavor: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def get_definition(self, kind: typing.Any) -> typing.Any:
    """Execute get definition helper logic.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None


def test_rdna_synthesizer_branches() -> None:
  """Verify rdna synthesizer branches behavior."""
  # 1. Test when macros.json does not exist
  with patch("os.path.exists", return_value=False):
    synth: typing.Any = RdnaSynthesizer(semantics=DummySemantics())
    assert not getattr(synth, "macro_registry")

  # 2. Test when macros.json exists but is empty
  with patch("os.path.exists", return_value=True):
    with patch("builtins.open", __import__("unittest").mock.mock_open(read_data="{}")):
      synth2: typing.Any = RdnaSynthesizer(semantics=DummySemantics())
      assert not getattr(synth2, "macro_registry")

  # 3. Test multiple edges to same target (174->176)
  nodes: typing.Any = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="in2", op_type="Input"),
    LogicalNode(id="add", op_type="Add"),
    LogicalNode(id="out", op_type="Output"),  # Empty output sources?
  ]
  edges: typing.Any = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),  # multiple edges
  ]
  graph: typing.Any = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  synth3: typing.Any = RdnaSynthesizer(semantics=DummySemantics())
  synth3.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes2: typing.Any = [LogicalNode(id="out2", op_type="Output")]
  edges2: typing.Any = []
  graph2: typing.Any = LogicalGraph(nodes={n.id: n for n in nodes2}, edges=edges2)
  synth4: typing.Any = RdnaSynthesizer(semantics=DummySemantics())
  synth4.from_graph(graph2)


def test_synthesizer_init(tmp_path: typing.Any) -> None:
  """Verify synthesizer init behavior.

  Args:
      tmp_path: Test argument or fixture.
  """
  macros_json: typing.Any = tmp_path / "macros.json"
  macros_json.write_text('{"Conv2d": "expand_conv2d", "UnknownOp": "expand_unknown"}')

  semantics: typing.Any = MockSemanticsManager()

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synthesizer: typing.Any = RdnaSynthesizer(semantics)
      assert "Conv2d" in synthesizer.macro_registry
      assert "UnknownOp" not in synthesizer.macro_registry

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=False):
    synth2: typing.Any = RdnaSynthesizer(semantics)
    assert "Conv2d" not in synth2.macro_registry


def test_synthesizer_init_yaml(tmp_path: typing.Any) -> None:
  """Tests RDNA synthesizer initialization when macros.yaml is present.

  Args:
      tmp_path: Test argument or fixture.
  """
  macros_yaml: typing.Any = tmp_path / "macros.yaml"
  macros_yaml.write_text("Conv2d: expand_conv2d\nUnknownOp: expand_unknown\n")

  semantics: typing.Any = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    synth: typing.Any = RdnaSynthesizer(semantics)
    assert "Conv2d" in synth.macro_registry
    assert "UnknownOp" not in synth.macro_registry


def test_synthesizer_init_filenotfound(tmp_path: typing.Any) -> None:
  """Tests RDNA synthesizer initialization when files exist in os.path.exists check but fail on open.

  Args:
      tmp_path: Test argument or fixture.
  """
  semantics: typing.Any = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synth: typing.Any = RdnaSynthesizer(semantics)
      assert synth.macro_registry == {}


def test_unmapped_op_and_comment() -> None:
  """Verify unmapped op and comment behavior."""
  # Test Unmapped Op and RdnaComment branches
  nodes: typing.Any = [LogicalNode(id="unmapped1", op_type="TotallyUnknownOp")]
  nodes.append(LogicalNode(id="n2", op_type="", attributes={}))  # Hit 211->214
  graph: typing.Any = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])

  synth: typing.Any = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod: typing.Any = synth.from_graph(graph)
  assert cst_mod is not None
