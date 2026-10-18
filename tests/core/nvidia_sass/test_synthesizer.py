"""Test suite for synthesizer."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import MAX_REGISTERS, NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassNode,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_sequential() -> None:
  """Verifies the behavior of allocator sequential."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  r1: typing.Any = alloc.get_register("x")
  r2: typing.Any = alloc.get_register("y")
  assert r1.name == "R0"
  assert r2.name == "R1"


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_reuse() -> None:
  """Verifies the behavior of allocator reuse."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  r1: typing.Any = alloc.get_register("x")
  r2: typing.Any = alloc.get_register("x")
  assert r1.name == "R0"
  assert r2.name == "R0"


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_overflow() -> None:
  """Verifies the behavior of allocator overflow."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  alloc._free_pool = []
  with pytest.raises(ValueError, match="NvidiaSassRegister overflow"):
    alloc.get_register("overflow")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_temp() -> None:
  """Verifies the behavior of allocator temp."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  t1: typing.Any = alloc.allocate_temp()
  t2: typing.Any = alloc.allocate_temp()
  assert t1.name != t2.name
  assert t1.name.startswith("R")
  assert t2.name.startswith("R")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_reset() -> None:
  """Verifies the behavior of allocator reset."""
  alloc: typing.Any = RegisterAllocator(256, lambda x: None)
  alloc.get_register("x")
  assert len(alloc._free_pool) == MAX_REGISTERS - 1
  alloc.reset()
  assert len(alloc._free_pool) == MAX_REGISTERS
  assert alloc._var_to_reg == {}


@pytest.fixture
def mock_semantics() -> typing.Any:
  """Execute mock semantics helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr: typing.Any = MagicMock(spec=SemanticsManager)

  def resolve(kind: str, target: str) -> typing.Any:
    """Resolves .

    Args:
        kind: Test argument or fixture.
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if target != "nvidia_sass":
      return None
    if kind == "Add":
      return {"api": "FADD"}
    if kind == "Mul":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve

  def get_def(kind: str) -> typing.Any:
    """Gets def.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "Conv2d" in kind:
      return ("Conv2d", {})
    if "Linear" in kind:
      return ("Linear", {})
    return None

  mgr.get_definition.side_effect = get_def
  return mgr


def test_graph_to_sass_linear_flow(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS linear flow.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = NvidiaSassSynthesizer(mock_semantics)
  x: typing.Any = LogicalNode(id="x", op_type="Input")
  y: typing.Any = LogicalNode(id="y", op_type="Input")
  z: typing.Any = LogicalNode(id="z", op_type="Add", inputs=["x", "y"])
  g: typing.Any = LogicalGraph(nodes={"x": x, "y": y, "z": z})
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
  """Verifies the behavior of graph to NVIDIA_SASS unmapped op.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = NvidiaSassSynthesizer(mock_semantics)
  g: typing.Any = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="UnknownOp")})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS macro expansion.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = NvidiaSassSynthesizer(mock_semantics)
  g: typing.Any = LogicalGraph(nodes={"conv1": LogicalNode(id="conv1", op_type="Conv2d", attributes={"k": 3})})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) > 10
  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert "BEGIN Conv2d (conv1)" in comments
  labels: list[str] = [n.name for n in nodes if isinstance(n, NvidiaSassLabel)]
  assert any(("L_KY" in label for label in labels))
  opcodes: list[str] = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "IMAD" in opcodes
  assert "FFMA" in opcodes


def test_graph_to_sass_output_node(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to NVIDIA_SASS output node.

  Args:
      mock_semantics: Test argument or fixture.
  """
  synth: typing.Any = NvidiaSassSynthesizer(mock_semantics)
  in1: typing.Any = LogicalNode(id="in1", op_type="Input")
  out1: typing.Any = LogicalNode(id="out1", op_type="Output", inputs=["in1"])
  g: typing.Any = LogicalGraph(nodes={"in1": in1, "out1": out1})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


def test_nvidia_sass_to_python_instruction() -> None:
  """Verifies the behavior of NVIDIA_SASS to python instruction."""
  synth: typing.Any = NvidiaSassSynthesizer(MagicMock())
  inst: typing.Any = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.FADD(R1, R2)" in code


def test_nvidia_sass_to_python_immediates() -> None:
  """Verifies the behavior of NVIDIA_SASS to python immediates."""
  synth: typing.Any = NvidiaSassSynthesizer(MagicMock())
  inst: typing.Any = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=16, is_hex=True)]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.MOV(0x10)" in code


def test_nvidia_sass_to_python_no_dest() -> None:
  """Verifies the behavior of NVIDIA_SASS to python no dest."""
  synth: typing.Any = NvidiaSassSynthesizer(MagicMock())

  class RdnaLabelRef:
    """Test helper class RdnaLabelRef."""

    def __str__(self) -> typing.Any:
      """Helper to   string  .

      Returns:
          Computed test result or fixture value.
      """
      return "L_TARGET"

    def to_text(self) -> typing.Any:
      """To text.

      Returns:
          Computed test result or fixture value.
      """
      return "L_TARGET"

  inst: typing.Any = NvidiaSassInstruction(opcode="BRA", operands=[RdnaLabelRef()])
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "nvidia_sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_nvidia_sass_to_python_complex_operand() -> None:
  """Verifies the behavior of NVIDIA_SASS to python complex operand."""
  synth: typing.Any = NvidiaSassSynthesizer(MagicMock())

  class ComplexMem:
    """Test helper class ComplexMem."""

    def __str__(self) -> typing.Any:
      """Helper to   string  .

      Returns:
          Computed test result or fixture value.
      """
      return "[R1 + 0x4]"

    def to_text(self) -> typing.Any:
      """To text.

      Returns:
          Computed test result or fixture value.
      """
      return "[R1 + 0x4]"

  inst: typing.Any = NvidiaSassInstruction(opcode="LD", operands=[NvidiaSassRegister(name="R0"), ComplexMem()])
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.LD('[R1 + 0x4]')" in code
