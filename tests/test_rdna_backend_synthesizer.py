"""Test suite for rdna backend synthesizer."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import pathlib
from pydantic import JsonValue

from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import MAX_SGPR, RdnaBackend, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class MockSemanticsManager:
  """Test helper class MockSemanticsManager."""

  def get_definition(self, kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        kind: The kind (str) for get definition.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    if kind == "known_op":
      _ret_v_1: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": [getattr({"v": "add"}, "__getitem__")("v")]}, "__getitem__"
      )("v")
      return _ret_v_1
    elif kind == "macro_op":
      _ret_v_2: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": [getattr({"v": "relu"}, "__getitem__")("v")]}, "__getitem__"
      )("v")
      return _ret_v_2
    elif kind == "macro_suffix_op":
      _ret_v_3: tuple[str, dict[str, JsonValue]] | None = getattr(
        {"v": [getattr({"v": "some.relu"}, "__getitem__")("v")]}, "__getitem__"
      )("v")
      return _ret_v_3
    return None

  def resolve_variant(self, abstract_id: str, backend: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        backend: The backend (str) for resolve variant.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if abstract_id == "add":
      return {"api": "v_add_f32"}
    return None


def test_register_allocator() -> None:
  """Verify register allocator behavior."""
  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))

  v1 = allocator.get_register("x")
  v2 = allocator.get_register("x")
  assert v1.index == v2.index

  s1 = allocator.get_register("y")
  s2 = allocator.get_register("y")
  assert s1.index == s2.index

  v_temp = allocator.allocate_temp()
  assert isinstance(v_temp, RdnaVGPR)
  assert v_temp.index != v1.index

  s_temp = allocator.allocate_temp()
  assert isinstance(s_temp, RdnaVGPR)
  assert s_temp.index != s1.index

  allocator.reset()
  assert len(allocator._free_pool) == 256


def test_register_allocator_overflow() -> None:
  """Verify register allocator overflow behavior."""
  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  allocator._free_pool = []
  with pytest.raises(ValueError, match="overflow"):
    allocator.get_register("new_var")

  setattr(getattr({"v": allocator}, "__getitem__")("v"), "_next_sgpr", getattr({"v": MAX_SGPR}, "__getitem__")("v"))
  with pytest.raises(ValueError, match="overflow"):
    allocator.get_register("new_var_s")


def test_synthesizer_from_graph() -> None:
  """Verify synthesizer from graph behavior."""
  semantics = MockSemanticsManager()
  synthesizer = RdnaSynthesizer(semantics)

  n_in = LogicalNode(id="in1", op_type="Input", attributes={"name": "input_x"})
  n_op1 = LogicalNode(id="op1", op_type="known_op", inputs=["in1"])
  n_op2 = LogicalNode(id="op2", op_type="unknown_op", inputs=["op1"])
  n_out = LogicalNode(id="out1", op_type="Output", inputs=["op2"])
  n_macro = LogicalNode(id="op_macro", op_type="macro_op")
  n_macro_suffix = LogicalNode(id="op_macro_s", op_type="macro_suffix_op")

  graph = LogicalGraph(
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
  setattr(semantics, "resolve_variant", lambda aid, b: {"api": "v_add_f32"} if aid == "add" else {})

  nodes = synthesizer.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Verify synthesizer to python behavior."""
  semantics = MockSemanticsManager()
  synthesizer = RdnaSynthesizer(semantics)

  inst1 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  inst2 = RdnaInstruction(opcode="store_dword", operands=[RdnaVGPR(index=0), RdnaImmediate(value=4)])
  inst3 = RdnaInstruction(opcode="s_waitcnt", operands=[])
  inst4 = RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabel(name="L1")])

  class CustomRdnaOperand(RdnaNode):
    """Custom RDNA operand node for testing."""

    raw: str = ""

    def to_text(self) -> str:
      """Return raw string representation.

      Returns:
          str: The raw string representation.
      """
      return self.raw

  inst5 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=1.5, is_hex=False)])
  inst6 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=0x10, is_hex=True)])
  inst7 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), CustomRdnaOperand(raw="some_string")])
  inst8 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), CustomRdnaOperand(raw="array[0]")])
  inst9 = RdnaInstruction(
    opcode="v_add_f32", operands=[CustomRdnaOperand(raw="bad_dest"), CustomRdnaOperand(raw="array[0]")]
  )
  label1 = RdnaLabel(name="L1")
  comment = RdnaComment(text="; Comment")

  cst_module = synthesizer.to_python([inst1, inst2, inst3, inst4, inst5, inst6, inst7, inst8, inst9, label1, comment])
  assert isinstance(cst_module, cst.Module)


def test_rdna_backend() -> None:
  """Verify rdna backend behavior."""
  semantics = MockSemanticsManager()
  backend = getattr(
    {"v": getattr({"v": RdnaBackend}, "__getitem__")("v")(getattr({"v": semantics}, "__getitem__")("v"))}, "__getitem__"
  )("v")

  graph = LogicalGraph()
  n_in = LogicalNode(id="in1", op_type="Input")
  graph.add_node(n_in)

  result = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in result


def test_rdna_backend_default_semantics() -> None:
  """Verify rdna backend default semantics behavior."""
  backend = RdnaBackend()
  assert backend.synthesizer.semantics is not None


class DummySemantics:
  """Test helper class DummySemantics."""

  def resolve_variant(self, abstract_id: str, flavor: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        abstract_id: The abstract id (str) for resolve variant.
        flavor: The flavor (str) for resolve variant.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    return None

  def get_definition(self, kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        kind: The kind (str) for get definition.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return None


def test_rdna_synthesizer_branches() -> None:
  """Verify rdna synthesizer branches behavior."""
  # 1. Test when macros.json does not exist
  with patch("os.path.exists", return_value=False):
    synth = RdnaSynthesizer(semantics=DummySemantics())
    assert not getattr(synth, "macro_registry")

  # 2. Test when macros.json exists but is empty
  with patch("os.path.exists", return_value=True):
    with patch("builtins.open", __import__("unittest").mock.mock_open(read_data="{}")):
      synth2 = RdnaSynthesizer(semantics=DummySemantics())
      assert not getattr(synth2, "macro_registry")

  # 3. Test multiple edges to same target (174->176)
  nodes = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="in2", op_type="Input"),
    LogicalNode(id="add", op_type="Add"),
    LogicalNode(id="out", op_type="Output"),  # Empty output sources?
  ]
  edges = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),  # multiple edges
  ]
  graph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  synth3 = RdnaSynthesizer(semantics=DummySemantics())
  synth3.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes2 = [LogicalNode(id="out2", op_type="Output")]
  edges2: list[RuntimeTensorOrValue] = []
  graph2 = getattr(
    {
      "v": getattr({"v": LogicalGraph}, "__getitem__")("v")(
        nodes=getattr({"v": {getattr({"v": n}, "__getitem__")("v").id: n for n in nodes2}}, "__getitem__")("v"),
        edges=getattr({"v": edges2}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  synth4 = RdnaSynthesizer(semantics=DummySemantics())
  synth4.from_graph(graph2)


def test_synthesizer_init(tmp_path: pathlib.Path) -> None:
  """Verify synthesizer init behavior.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  macros_json = tmp_path / "macros.json"
  macros_json.write_text('{"Conv2d": "expand_conv2d", "UnknownOp": "expand_unknown"}')

  semantics = MockSemanticsManager()

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synthesizer = RdnaSynthesizer(semantics)
      assert "Conv2d" in synthesizer.macro_registry
      assert "UnknownOp" not in synthesizer.macro_registry

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=False):
    synth2 = RdnaSynthesizer(semantics)
    assert "Conv2d" not in synth2.macro_registry


def test_synthesizer_init_yaml(tmp_path: pathlib.Path) -> None:
  """Tests RDNA synthesizer initialization when macros.yaml is present.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  macros_yaml = tmp_path / "macros.yaml"
  macros_yaml.write_text("Conv2d: expand_conv2d\nUnknownOp: expand_unknown\n")

  semantics = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    synth = RdnaSynthesizer(semantics)
    assert "Conv2d" in synth.macro_registry
    assert "UnknownOp" not in synth.macro_registry


def test_synthesizer_init_filenotfound(tmp_path: pathlib.Path) -> None:
  """Tests RDNA synthesizer initialization when files exist in os.path.exists check but fail on open.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  semantics = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synth = RdnaSynthesizer(semantics)
      assert synth.macro_registry == {}


def test_unmapped_op_and_comment() -> None:
  """Verify unmapped op and comment behavior."""
  # Test Unmapped Op and RdnaComment branches
  nodes = [LogicalNode(id="unmapped1", op_type="TotallyUnknownOp")]
  nodes.append(LogicalNode(id="n2", op_type="AnotherUnknownOp", attributes={}))
  graph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])

  synth = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod = synth.from_graph(graph)
  assert cst_mod is not None
