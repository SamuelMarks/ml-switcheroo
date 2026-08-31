"""Test suite for the Synthesizer module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.sass.synthesizer import MAX_REGISTERS, RegisterAllocator, SassSynthesizer
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassComment,
  SassImmediate,
  SassInstruction,
  SassLabel,
  SassNode,
  SassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_allocator_sequential() -> None:
  """Verifies the behavior of allocator sequential."""
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("y")
  assert r1.name == "R0"
  assert r2.name == "R1"


def test_allocator_reuse() -> None:
  """Verifies the behavior of allocator reuse."""
  alloc = RegisterAllocator()
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("x")
  assert r1.name == "R0"
  assert r2.name == "R0"


def test_allocator_overflow() -> None:
  """Verifies the behavior of allocator overflow."""
  alloc = RegisterAllocator()
  alloc._free_pool = []
  with pytest.raises(ValueError, match="SassRegister overflow"):
    alloc.get_register("overflow")


def test_allocator_temp() -> None:
  """Verifies the behavior of allocator temp."""
  alloc = RegisterAllocator()
  t1 = alloc.allocate_temp()
  t2 = alloc.allocate_temp()
  assert t1.name != t2.name
  assert t1.name.startswith("R")
  assert t2.name.startswith("R")


def test_allocator_reset() -> None:
  """Verifies the behavior of allocator reset."""
  alloc = RegisterAllocator()
  alloc.get_register("x")
  assert len(alloc._free_pool) == MAX_REGISTERS - 1
  alloc.reset()
  assert len(alloc._free_pool) == MAX_REGISTERS
  assert alloc._var_to_reg == {}


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Docstring."""
  mgr = MagicMock(spec=SemanticsManager)

  def resolve(kind: str, target: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if target != "sass":
      return None
    if kind == "Add":
      return {"api": "FADD"}
    if kind == "Mul":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve

  def get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if "Conv2d" in kind:
      return ("Conv2d", {})
    if "Linear" in kind:
      return ("Linear", {})
    return None

  mgr.get_definition.side_effect = get_def
  return mgr


def test_graph_to_sass_linear_flow(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to SASS linear flow."""
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("x", "Input", {}), LogicalNode("y", "Input", {}), LogicalNode("z", "Add", {})]
  g.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]
  nodes: list[SassNode] = synth.from_graph(g)
  assert len(nodes) == 3
  assert isinstance(nodes[0], SassComment)
  assert "// Input x -> R0" in str(nodes[0])
  assert isinstance(nodes[1], SassComment)
  assert "// Input y -> R1" in str(nodes[1])
  inst: typing.Any = nodes[2]
  assert isinstance(inst, SassInstruction)
  assert inst.opcode == "FADD"
  assert getattr(inst.operands[0], "name", None) == "R2"
  assert getattr(inst.operands[1], "name", None) == "R0"
  assert getattr(inst.operands[2], "name", None) == "R1"


def test_graph_to_sass_unmapped_op(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to SASS unmapped op."""
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("n1", "UnknownOp", {})]
  nodes: list[SassNode] = synth.from_graph(g)
  assert len(nodes) == 1
  assert isinstance(nodes[0], SassComment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to SASS macro expansion."""
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("conv1", "Conv2d", {"k": 3})]
  nodes: list[SassNode] = synth.from_graph(g)
  assert len(nodes) > 10
  comments: list[str] = [typing.cast(SassComment, n).text for n in nodes if isinstance(n, SassComment)]
  assert "BEGIN Conv2d (conv1)" in comments
  labels: list[str] = [typing.cast(SassLabel, n).name for n in nodes if isinstance(n, SassLabel)]
  assert any(("L_KY" in label for label in labels))
  opcodes: list[str] = [typing.cast(SassInstruction, n).opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "IMAD" in opcodes
  assert "FFMA" in opcodes


def test_graph_to_sass_output_node(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to SASS output node."""
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("in1", "Input", {}), LogicalNode("out1", "Output", {})]
  g.edges = [LogicalEdge("in1", "out1")]
  nodes: list[SassNode] = synth.from_graph(g)
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


def test_sass_to_python_instruction() -> None:
  """Verifies the behavior of SASS to python instruction."""
  synth = SassSynthesizer(MagicMock())
  inst = SassInstruction(
    opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = sass.FADD(R1, R2)" in code


def test_sass_to_python_immediates() -> None:
  """Verifies the behavior of SASS to python immediates."""
  synth = SassSynthesizer(MagicMock())
  inst = SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassImmediate(value=16, is_hex=True)])  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = sass.MOV(0x10)" in code


def test_sass_to_python_no_dest() -> None:
  """Verifies the behavior of SASS to python no dest."""
  synth = SassSynthesizer(MagicMock())

  class RdnaLabelRef:
    """Docstring."""

    def __str__(self) -> str:
      """Helper to   string  ."""
      return "L_TARGET"

    def to_text(self) -> str:
      """To text."""
      return "L_TARGET"

  inst = SassInstruction(opcode="BRA", operands=[RdnaLabelRef()])  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_sass_to_python_complex_operand() -> None:
  """Verifies the behavior of SASS to python complex operand."""
  synth = SassSynthesizer(MagicMock())

  class ComplexMem:
    """Docstring."""

    def __str__(self) -> str:
      """Helper to   string  ."""
      return "[R1 + 0x4]"

    def to_text(self) -> str:
      """To text."""
      return "[R1 + 0x4]"

  inst = SassInstruction(opcode="LD", operands=[SassRegister(name="R0"), ComplexMem()])  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = sass.LD('[R1 + 0x4]')" in code
