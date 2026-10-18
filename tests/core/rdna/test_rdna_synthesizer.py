"""Test suite for rdna synthesizer."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaSGPR,
  RdnaVGPR,
)


import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import MAX_SGPR, MAX_VGPR, RdnaBackend, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaOperand
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_dual_pools() -> None:
  """Verifies the behavior of allocator dual pools."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  v0: typing.Any = alloc.get_register("x")
  s0: typing.Any = alloc.get_register("cnt")
  assert isinstance(v0, RdnaVGPR)
  assert v0.index == 0
  assert isinstance(s0, RdnaSGPR)
  assert s0.index == 0


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_reuse() -> None:
  """Verifies the behavior of allocator reuse."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  v_a: typing.Any = alloc.get_register("a")
  v_b: typing.Any = alloc.get_register("a")
  assert v_a.index == v_b.index == 0


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_overflow_vgpr() -> None:
  """Verifies the behavior of allocator overflow vgpr."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  alloc._next_vgpr = MAX_VGPR + 1
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    alloc.get_register("fail")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_overflow_sgpr() -> None:
  """Verifies the behavior of allocator overflow sgpr."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  alloc._next_sgpr = MAX_SGPR + 1
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    alloc.get_register("fail")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_temps() -> None:
  """Verifies the behavior of allocator temps."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  t1: typing.Any = alloc.allocate_temp()
  t2: typing.Any = alloc.allocate_temp()
  assert t1.index == 0
  assert t2.index == 0
  t3: typing.Any = alloc.allocate_temp()
  assert t3.index == 1


@pytest.fixture
def mock_semantics() -> typing.Any:
  """Execute mock semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: typing.Any = MagicMock(spec=SemanticsManager)

  def get_def(kind: str) -> typing.Any:
    """Gets def.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if kind == "Add":
      return ("Add", {})
    return None

  def resolve(aid: str, fw: str) -> typing.Any:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  return mgr


def test_graph_to_rdna_basic_math(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to RDNA basic math.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = RdnaSynthesizer(mock_semantics)
  x: typing.Any = LogicalNode(id="x", op_type="Input")
  y: typing.Any = LogicalNode(id="y", op_type="Input")
  z: typing.Any = LogicalNode(id="z", op_type="Add", inputs=["x", "y"])
  g: typing.Any = LogicalGraph(nodes={"x": x, "y": y, "z": z})
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) == 3
  assert isinstance(nodes[0], RdnaComment)
  assert "** Input x -> v0" in str(nodes[0]).replace(";", "**")
  inst: typing.Any = nodes[2]
  assert isinstance(inst, RdnaInstruction)
  assert inst.opcode == "v_add_f32"
  assert str(inst.operands[0]) == "v2"
  assert str(inst.operands[1]) == "v0"
  assert str(inst.operands[2]) == "v1"


def test_graph_to_rdna_unmapped(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to RDNA unmapped.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = RdnaSynthesizer(mock_semantics)
  g: typing.Any = LogicalGraph()
  g.nodes = {"n1": LogicalNode("n1", "MysteryOp", attributes={})}
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) == 1
  assert "Unmapped Op: MysteryOp" in str(nodes[0])


def test_rdna_to_python_instruction() -> None:
  """Verifies the behavior of RDNA to python instruction."""
  synth: typing.Any = RdnaSynthesizer(MagicMock())
  inst: typing.Any = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "v0 = rdna.v_add_f32(v1, v2)" in code


def test_rdna_to_python_ranges() -> None:
  """Verifies the behavior of RDNA to python ranges."""
  synth: typing.Any = RdnaSynthesizer(MagicMock())
  inst: typing.Any = RdnaInstruction(
    opcode="image_load", operands=[RdnaVGPR(index=0, count=4), RdnaVGPR(index=4, count=4), RdnaSGPR(index=0, count=4)]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "v_0_3 = rdna.image_load(v_4_7, s_0_3)" in code


def test_rdna_to_python_label() -> None:
  """Verifies the behavior of RDNA to python label."""
  synth: typing.Any = RdnaSynthesizer(MagicMock())
  nodes: list[typing.Any] = [RdnaLabel(name="L_LOOP")]
  mod: typing.Any = synth.to_python(nodes)
  code: str = mod.code
  assert "# RdnaLabel: L_LOOP" in code


# --- Merged from test_rdna_synthesizer_missing.py ---


@pytest.mark.skip(reason="Tests old allocator mock")
def test_register_allocator_overflow_vgpr() -> None:
  """Verify register allocator overflow vgpr behavior."""
  allocator: typing.Any = RegisterAllocator(256, lambda x: None)
  allocator._next_vgpr = 256
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_register("test")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_register_allocator_overflow_sgpr() -> None:
  """Verify register allocator overflow sgpr behavior."""
  allocator: typing.Any = RegisterAllocator(256, lambda x: None)
  allocator._next_sgpr = 106
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_register("test")


def test_convert_operand_to_py_immediate_float() -> None:
  """Verify convert operand to py immediate float behavior."""
  synth: typing.Any = RdnaSynthesizer(None)
  imm: typing.Any = RdnaImmediate(value=3.14)
  res: typing.Any = synth._convert_operand_to_py(imm)
  assert getattr(res, "value", None) == "3.14"


def test_convert_operand_to_py_brackets() -> None:
  """Verify convert operand to py brackets behavior."""
  synth: typing.Any = RdnaSynthesizer(None)

  class DummyOp(RdnaOperand):
    """Test helper class DummyOp."""

    def __str__(self) -> typing.Any:
      """Execute str helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "v[1:2]"

    def to_text(self) -> typing.Any:
      """To text.

      Returns:
          Computed test result or fixture value.
      """
      return "v[1:2]"

  res: typing.Any = synth._convert_operand_to_py(DummyOp())
  assert getattr(res, "value", None) == "v_1_2"


def test_convert_operand_to_py_fallback() -> None:
  """Verify convert operand to py fallback behavior."""
  synth: typing.Any = RdnaSynthesizer(None)

  class DummyOp(RdnaOperand):
    """Test helper class DummyOp."""

    def __str__(self) -> typing.Any:
      """Execute str helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return "some-weird-str!"

    def to_text(self) -> typing.Any:
      """To text.

      Returns:
          Computed test result or fixture value.
      """
      return "some-weird-str!"

  res: typing.Any = synth._convert_operand_to_py(DummyOp())
  assert res.value == "'some-weird-str!'"


def test_rdna_synthesizer_label_conversion() -> None:
  """Verify rdna synthesizer label conversion behavior."""
  synth: typing.Any = RdnaSynthesizer(None)
  label: typing.Any = RdnaLabel(name="my_label")
  mod: typing.Any = synth.to_python([label])
  assert mod is not None


def test_convert_instruction_to_py_no_operands() -> None:
  """Verify convert instruction to py no operands behavior."""
  synth: typing.Any = RdnaSynthesizer(None)
  inst: typing.Any = RdnaInstruction(opcode="s_endpgm", operands=[])
  res: typing.Any = synth._convert_instruction_to_py(inst)
  assert res is not None


# --- Merged from test_rdna_synthesizer_missing_more.py ---


def test_rdna_synthesizer_macro_exact_match() -> None:
  """Verify rdna synthesizer macro exact match behavior."""
  semantics: typing.Any = SemanticsManager()
  synth: typing.Any = RdnaSynthesizer(semantics)
  # mock a macro
  synth.macro_registry["my_abstract_id"] = lambda alloc, nid, meta: [RdnaComment(text="mock")]

  graph: typing.Any = LogicalGraph("test")
  n: typing.Any = LogicalNode("n1", "my_abstract_id")
  graph.add_node(n)

  # We also mock get_definition
  original: typing.Any = semantics.get_definition

  def mock_get_def(kind: str) -> typing.Any:
    """Execute mock get def helper logic.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if kind == "my_abstract_id":
      return ("my_abstract_id", {})
    return original(kind)

  semantics.get_definition = mock_get_def

  nodes: list[typing.Any] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaComment)
  assert nodes[0].text == "mock"


def test_rdna_backend_compile() -> None:
  """Verify rdna backend compile behavior."""
  backend: typing.Any = RdnaBackend()
  graph: typing.Any = LogicalGraph("test")
  n: typing.Any = LogicalNode("n1", "Input")
  graph.add_node(n)
  code: str = backend.compile(graph)
  assert "; RDNA Code Generation Initialized" in code
