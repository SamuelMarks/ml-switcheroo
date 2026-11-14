"""Test suite for synthesizer."""

from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

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
  """Verify `RegisterAllocator` handles allocator sequential as expected."""
  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("y")
  assert r1.name == "R0"
  assert r2.name == "R1"


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_reuse() -> None:
  """Verify `RegisterAllocator` handles allocator reuse as expected."""
  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  r1 = alloc.get_register("x")
  r2 = alloc.get_register("x")
  assert r1.name == "R0"
  assert r2.name == "R0"


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_overflow() -> None:
  """Verify `RegisterAllocator` raises `ValueError` during allocator overflow."""
  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  alloc._free_pool = []
  with pytest.raises(ValueError, match="NvidiaSassRegister overflow"):
    alloc.get_register("overflow")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_temp() -> None:
  """Verify `RegisterAllocator` handles allocator temp as expected."""
  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  t1 = alloc.allocate_temp()
  t2 = alloc.allocate_temp()
  assert t1.name != t2.name
  assert t1.name.startswith("R")
  assert t2.name.startswith("R")


@pytest.mark.skip(reason="Tests old allocator mock")
def test_allocator_reset() -> None:
  """Verify `RegisterAllocator` handles allocator reset as expected."""
  alloc = RegisterAllocator(256, lambda x: NvidiaSassRegister(name=f"R{x}"))
  alloc.get_register("x")
  assert len(alloc._free_pool) == MAX_REGISTERS - 1
  alloc.reset()
  assert len(alloc._free_pool) == MAX_REGISTERS
  assert alloc._var_to_reg_id == {}


@pytest.fixture
def mock_semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for synthesizer tests.

  Returns:
      Configured `mock_semantics` fixture value (SemanticsManager).
  """
  mgr = MagicMock(spec=SemanticsManager)

  def resolve(kind: str, target: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in synthesizer tests.

    Args:
        kind: Input `kind` parameter value (`str`).
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if target != "nvidia_sass":
      return None
    if kind == "Add":
      return {"api": "FADD"}
    if kind == "Mul":
      return {"api": "FMUL"}
    return None

  mgr.resolve_variant.side_effect = resolve

  def get_def(kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        kind: Input `kind` parameter value (`str`).

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "Conv2d" in kind:
      return ("Conv2d", {})
    if "Linear" in kind:
      return ("Linear", {})
    return None

  mgr.get_definition.side_effect = get_def
  return mgr


def test_graph_to_sass_linear_flow(mock_semantics: MagicMock) -> None:
  """Verify `NvidiaSassSynthesizer` returns `3` for graph to SASS linear flow.

  Args:
      mock_semantics: Input `mock_semantics` parameter value (`MagicMock`).
  """
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
  inst = nodes[2]
  assert isinstance(inst, NvidiaSassInstruction)
  assert inst.opcode == "FADD"
  op0, op1, op2 = inst.operands[0], inst.operands[1], inst.operands[2]
  assert isinstance(op0, NvidiaSassRegister)
  assert op0.name == "R2"
  assert isinstance(op1, NvidiaSassRegister)
  assert op1.name == "R0"
  assert isinstance(op2, NvidiaSassRegister)
  assert op2.name == "R1"


def test_graph_to_sass_unmapped_op(mock_semantics: MagicMock) -> None:
  """Verify `NvidiaSassSynthesizer` returns `1` for graph to SASS unmapped operation.

  Args:
      mock_semantics: Input `mock_semantics` parameter value (`MagicMock`).
  """
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="UnknownOp")})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "Unmapped Op: UnknownOp" in str(nodes[0])


def test_graph_to_sass_macro_expansion(mock_semantics: MagicMock) -> None:
  """Verify `NvidiaSassSynthesizer` emits `BEGIN Conv2d (conv1)` for graph to SASS macro expansion.

  Args:
      mock_semantics: Input `mock_semantics` parameter value (`MagicMock`).
  """
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={"conv1": LogicalNode(id="conv1", op_type="Conv2d", attributes={"k": 3})})
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
  """Verify `NvidiaSassSynthesizer` returns `2` for graph to SASS output node.

  Args:
      mock_semantics: Input `mock_semantics` parameter value (`MagicMock`).
  """
  synth = NvidiaSassSynthesizer(mock_semantics)
  in1 = LogicalNode(id="in1", op_type="Input")
  out1 = LogicalNode(id="out1", op_type="Output", inputs=["in1"])
  g = LogicalGraph(nodes={"in1": in1, "out1": out1})
  nodes: list[NvidiaSassNode] = synth.from_graph(g)
  assert len(nodes) == 2
  assert "Return: R0" in str(nodes[1])


def test_nvidia_sass_to_python_instruction() -> None:
  """Verify `NvidiaSassSynthesizer` handles nvidia SASS to python instruction as expected."""
  synth = NvidiaSassSynthesizer(MagicMock())
  inst = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  mod = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.FADD(R1, R2)" in code


def test_nvidia_sass_to_python_immediates() -> None:
  """Verify nvidia SASS to python immediates works as expected."""
  synth = NvidiaSassSynthesizer(MagicMock())
  inst = NvidiaSassInstruction(
    opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=16, is_hex=True)]
  )
  mod = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.MOV(0x10)" in code


def test_nvidia_sass_to_python_no_dest() -> None:
  """Verify nvidia SASS to python no dest works as expected."""
  synth = NvidiaSassSynthesizer(MagicMock())

  class RdnaLabelRef(NvidiaSassNode):
    """Stub `NvidiaSassNode` subclass implementing `to_text` for synthesizer tests."""

    def __str__(self) -> str:
      """Execute `__str__` on `RdnaLabelRef` for synthesizer tests.

      Returns:
          Resulting `str` from str.
      """
      return "L_TARGET"

    def to_text(self) -> str:
      """To text.

      Returns:
          Resulting `str` from to text.
      """
      return "L_TARGET"

  inst = NvidiaSassInstruction(opcode="BRA", operands=[RdnaLabelRef()])
  mod = synth.to_python([inst])
  code: str = mod.code
  assert "nvidia_sass.BRA('L_TARGET')" in code
  assert "=" not in code


def test_nvidia_sass_to_python_complex_operand() -> None:
  """Verify `NvidiaSassSynthesizer` handles nvidia SASS to python complex operand as expected."""
  synth = NvidiaSassSynthesizer(MagicMock())

  class ComplexMem(NvidiaSassNode):
    """Stub `NvidiaSassNode` subclass implementing `to_text` for synthesizer tests."""

    def __str__(self) -> str:
      """Execute `__str__` on `ComplexMem` for synthesizer tests.

      Returns:
          Resulting `str` from str.
      """
      return "[R1 + 0x4]"

    def to_text(self) -> str:
      """To text.

      Returns:
          Resulting `str` from to text.
      """
      return "[R1 + 0x4]"

  inst = NvidiaSassInstruction(opcode="LD", operands=[NvidiaSassRegister(name="R0"), ComplexMem()])
  mod = synth.to_python([inst])
  code: str = mod.code
  assert "R0 = nvidia_sass.LD('[R1 + 0x4]')" in code
