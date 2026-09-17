"""Test suite for the Synthesizer module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import (
  MAX_REGISTERS,
  RegisterAllocator,
  NvidiaSassSynthesizer,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
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
  with pytest.raises(ValueError, match="NvidiaSassRegister overflow"):
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
    if target != "nvidia_sass":
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
  """Verifies the behavior of graph to NVIDIA_SASS linear flow."""
  synth = NvidiaSassSynthesizer(mock_semantics)
  x = LogicalNode(id="x", op_type="Input")
  y = LogicalNode(id="y", op_type="Input")
  z = LogicalNode(id="z", op_type="Add", inputs=["x", "y"])
  g = LogicalGraph(nodes={"x": x, "y": y, "z": z})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 3
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "// Input x -> R0" in str(nodes[0])
  assert isinstance(nodes[1], NvidiaSassComment)
  assert "// Input y -> R1" in str(nodes[1])
  inst: typing.Any = nodes[2]
  assert isinstance(inst, NvidiaSassInstruction)
  assert inst.opcode == "FADD"
  assert getattr(inst.operands[0], "name", None) == "R2"
  assert getattr(inst.operands[1], "name", None) == "R0"
  assert getattr(inst.operands[2], "name", None) == "R1"


def test_graph_to_sass_unmapped_op(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS unmapped op."""
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="UnknownOp")})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS macro expansion."""
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={"conv1": LogicalNode(id="conv1", op_type="Conv2d", attributes={"k": 3})})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) > 10
  comments: list[str] = [typing.cast(NvidiaSassComment, n).text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert "BEGIN Conv2d (conv1)" in comments
  labels: list[str] = [typing.cast(NvidiaSassLabel, n).name for n in nodes if isinstance(n, NvidiaSassLabel)]
  assert any(("L_KY" in label for label in labels))
  opcodes: list[str] = [
    typing.cast(NvidiaSassInstruction, n).opcode for n in nodes if isinstance(n, NvidiaSassInstruction)
  ]
  assert "IMAD" in opcodes
  assert "FFMA" in opcodes


def test_graph_to_sass_output_node(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS output node."""
  synth = NvidiaSassSynthesizer(mock_semantics)
  in1 = LogicalNode(id="in1", op_type="Input")
  out1 = LogicalNode(id="out1", op_type="Output", inputs=["in1"])
  g = LogicalGraph(nodes={"in1": in1, "out1": out1})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


def test_nvidia_sass_to_python_instruction() -> None:
  """Verifies the behavior of NVIDIA_SASS to python instruction."""
  synth = NvidiaSassSynthesizer(MagicMock())
  inst = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.FADD(R1, R2)" in code


def test_nvidia_sass_to_python_immediates() -> None:
  """Verifies the behavior of NVIDIA_SASS to python immediates."""
  synth = NvidiaSassSynthesizer(MagicMock())
  inst = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=16, is_hex=True)]
  )  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.MOV(0x10)" in code


def test_nvidia_sass_to_python_no_dest() -> None:
  """Verifies the behavior of NVIDIA_SASS to python no dest."""
  synth = NvidiaSassSynthesizer(MagicMock())

  class RdnaLabelRef:
    """Docstring."""

    def __str__(self) -> str:
      """Helper to   string  ."""
      return "L_TARGET"

    def to_text(self) -> str:
      """To text."""
      return "L_TARGET"

  inst = NvidiaSassInstruction(opcode="BRA", operands=[RdnaLabelRef()])  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "nvidia_sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_nvidia_sass_to_python_complex_operand() -> None:
  """Verifies the behavior of NVIDIA_SASS to python complex operand."""
  synth = NvidiaSassSynthesizer(MagicMock())

  class ComplexMem:
    """Docstring."""

    def __str__(self) -> str:
      """Helper to   string  ."""
      return "[R1 + 0x4]"

    def to_text(self) -> str:
      """To text."""
      return "[R1 + 0x4]"

  inst = NvidiaSassInstruction(opcode="LD", operands=[NvidiaSassRegister(name="R0"), ComplexMem()])  # type: ignore
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.LD('[R1 + 0x4]')" in code
