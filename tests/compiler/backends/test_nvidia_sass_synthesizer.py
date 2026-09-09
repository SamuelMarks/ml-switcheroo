"""Unit tests for the NVIDIA_SASS Synthesizer and Register Allocator.

This module validates the correctness of:
1. `RegisterAllocator`: Ensuring correct register allocation, pooling, overflow error handling, and liveness-based register reuse/freeing.
2. `NvidiaSassSynthesizer`: Graph translation to NVIDIA_SASS CST nodes (including input/output metadata translation and unmapped op comments), Python CST generation from NVIDIA_SASS instruction sequences, and immediate/bracketed operand formatting.
3. `NvidiaSassBackend`: Instantiation with default configuration.
"""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator, NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_nvidia_sass_register_allocator_temps() -> None:
  """Verify that temporary register allocation and deallocation functions correctly.

  This test instantiates a RegisterAllocator, requests a temporary register,
  verifies it starts with the standard 'R' prefix, frees it back to the allocator's
  pool, and asserts that the pool has recovered to its full size (254 free).

  Args:
      None

  Returns:
      None
  """
  allocator = RegisterAllocator()
  reg: NvidiaSassRegister = allocator.allocate_temp()
  assert reg.name.startswith("R")

  allocator.free_register(reg.name)
  assert len(allocator._free_pool) == 254


def test_nvidia_sass_register_allocator_overflow() -> None:
  """Assert that the allocator raises a ValueError when the free pool is exhausted.

  This test simulates a register pressure overflow by emptying the allocator's
  free register pool and checking that any subsequent register request triggers
  the appropriate "NvidiaSassRegister overflow" ValueError.

  Args:
      None

  Returns:
      None
  """
  allocator = RegisterAllocator()
  allocator._free_pool = []
  with pytest.raises(ValueError, match="NvidiaSassRegister overflow"):
    allocator.get_register("overflow")


def test_nvidia_sass_register_allocator_record_usage() -> None:
  """Verify liveness-tracking and automatic register freeing based on usage.

  This test builds a liveness map from a minimal data-flow graph (Input to Output),
  checks the initial usage count for the node, gets a register, records usage on it,
  and asserts that once usage is exhausted, the register is automatically recycled
  and returned to the free register pool.

  Args:
      None

  Returns:
      None
  """
  allocator = RegisterAllocator()
  graph = LogicalGraph()
  graph.nodes = [LogicalNode("in", "Input"), LogicalNode("out", "Output")]
  graph.edges = [LogicalEdge("in", "out")]
  allocator.build_liveness(graph)

  assert allocator._liveness_map["in"] == 1

  allocator.get_register("in")
  assert len(allocator._free_pool) == 254

  allocator.record_usage("in")
  assert allocator._liveness_map["in"] == 0
  assert len(allocator._free_pool) == 255  # Freed


def test_nvidia_sass_synth_from_graph_inputs_outputs() -> None:
  """Test NVIDIA_SASS synthesizer generation of graph inputs and outputs as NVIDIA_SASS comments.

  This test sets up a LogicalGraph with one Input node and one Output node, then
  runs the synthesizer's `from_graph` method to ensure they are correctly mapped
  to a comment sequence detailing the parameter mapping and function return.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="in1", kind="Input", metadata={"name": "x"}))
  graph.nodes.append(LogicalNode(id="out1", kind="Output"))
  graph.edges.append(LogicalEdge(source="in1", target="out1"))

  nodes: list[typing.Any] = synth.from_graph(graph)
  assert len(nodes) == 2
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "Input x -> " in nodes[0].text
  assert isinstance(nodes[1], NvidiaSassComment)
  assert "Return:" in nodes[1].text


def test_nvidia_sass_synth_from_graph_unmapped() -> None:
  """Ensure unmapped operations emit fallback comment nodes in synthesized NVIDIA_SASS.

  This test feeds a LogicalGraph containing an unknown/unmapped operation to the
  NvidiaSassSynthesizer and verifies that instead of failing, it successfully produces
  a comment indicating that the operation was unmapped.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  mock_semantics.get_definition.return_value = None
  mock_semantics.resolve_variant.return_value = None
  synth = NvidiaSassSynthesizer(mock_semantics)

  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="op1", kind="UnknownOp"))

  nodes: list[typing.Any] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert isinstance(nodes[0], NvidiaSassComment)
  assert "Unmapped Op: UnknownOp" in nodes[0].text


def test_nvidia_sass_synth_to_python_label() -> None:
  """Verify that NvidiaSassLabel nodes are properly translated to Python comment CST.

  This test runs the `to_python` method with a list containing a NvidiaSassLabel and
  verifies that the resulting generated Python code correctly includes a comment
  representing the original label.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  nodes: list[typing.Any] = [NvidiaSassLabel(name="L_LOOP")]
  module: typing.Any = synth.to_python(nodes)
  code: str = module.code
  assert "# NvidiaSassLabel: L_LOOP" in code


def test_nvidia_sass_synth_to_python_comment() -> None:
  """Verify translation of NvidiaSassComment nodes to Python comments.

  This test checks that when converting a sequence of NVIDIA_SASS nodes containing
  comments and labels, the resulting Python code correctly represents comments
  preceding instructions/labels, but discards standalone comments that don't
  associate with a structural target.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  nodes: list[typing.Any] = [
    NvidiaSassComment(text="BEGIN block"),
    NvidiaSassComment(text="just a comment"),
    NvidiaSassLabel(name="L"),
  ]
  module: typing.Any = synth.to_python(nodes)
  code: str = module.code
  assert "# BEGIN block" in code

  nodes2: list[typing.Any] = [NvidiaSassComment(text="other comment")]
  module2: typing.Any = synth.to_python(nodes2)
  assert len(module2.body) == 0


def test_nvidia_sass_synth_to_python_instruction_empty_operands() -> None:
  """Verify code generation for instructions with no operands.

  This test ensures that a basic NVIDIA_SASS instruction with no operands, such as a NOP,
  is synthesized into a valid call to a Python representation under the `nvidia_sass` module
  without any arguments.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  inst = NvidiaSassInstruction(opcode="NOP", operands=[])
  module: typing.Any = synth.to_python([inst])
  code: str = module.code
  assert "nvidia_sass.NOP()" in code


def test_nvidia_sass_synth_to_python_instruction_store() -> None:
  """Verify store instruction translation does not involve register assignments.

  This test ensures that store instructions (like ST_E) which write to memory
  and do not return value to a destination register are correctly emitted as
  standalone function calls in Python (e.g., `nvidia_sass.ST_E(R0, R1)`) without assignment.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  # Store ops don't assign to a dest register
  inst = NvidiaSassInstruction(opcode="ST_E", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")])
  module: typing.Any = synth.to_python([inst])
  code: str = module.code
  assert "nvidia_sass.ST_E(R0, R1)" in code
  assert "=" not in code


def test_nvidia_sass_synth_to_python_instruction_branch() -> None:
  """Verify branch instruction translation compiles to a statement without assignment.

  This test checks that branch instructions (such as BRA), which only redirect
  control flow, compile to a basic method call under the `nvidia_sass` namespace
  without a Python variable assignment.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  inst = NvidiaSassInstruction(opcode="BRA", operands=[])
  module: typing.Any = synth.to_python([inst])
  code: str = module.code
  assert "nvidia_sass.BRA()" in code
  assert "=" not in code


def test_nvidia_sass_synth_to_python_instruction_alu() -> None:
  """Verify ALU instruction translation generates proper destination assignments.

  This test confirms that instructions yielding a value back to a register,
  such as float addition (FADD), are translated to a Python variable assignment
  (e.g., `R0 = nvidia_sass.FADD(R1, R2)`).

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  inst = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")]
  )
  module: typing.Any = synth.to_python([inst])
  code: str = module.code
  assert "R0 = nvidia_sass.FADD(R1, R2)" in code


def test_nvidia_sass_synth_to_python_instruction_predicate() -> None:
  """Verify NVIDIA_SASS instruction predicates are correctly mapped to keyword arguments.

  This test verifies that if a NVIDIA_SASS instruction is guarded by a predicate (such as P0),
  the synthesizer appends a corresponding `predicate='@P0'` keyword argument within
  the generated Python CST.

  Args:
      None

  Returns:
      None
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = NvidiaSassSynthesizer(mock_semantics)

  inst = NvidiaSassInstruction(
    opcode="FADD", operands=[NvidiaSassRegister(name="R0")], predicate=NvidiaSassPredicate(name="P0", is_guard=True)
  )
  module: typing.Any = synth.to_python([inst])
  code: str = module.code
  assert "R0 = nvidia_sass.FADD(predicate = '@P0')" in code


def test_nvidia_sass_synth_convert_operand_to_py_immediates() -> None:
  """Verify correct mapping of numeric NVIDIA_SASS immediates to Python CST nodes.

  This test checks that different formats of NVIDIA_SASS immediate values (integers,
  hexadecimals, and floating-point values) are translated into their correct
  respective `libcst.Integer` or `libcst.Float` counterparts with matching syntax.

  Args:
      None

  Returns:
      None
  """
  synth = NvidiaSassSynthesizer(MagicMock())

  imm1 = NvidiaSassImmediate(value=10, is_hex=False)
  py1: typing.Any = synth._convert_operand_to_py(imm1)
  assert isinstance(py1, cst.Integer)
  assert py1.value == "10"

  imm2 = NvidiaSassImmediate(value=255, is_hex=True)
  py2: typing.Any = synth._convert_operand_to_py(imm2)
  assert isinstance(py2, cst.Integer)
  assert py2.value == "0xff"

  imm3 = NvidiaSassImmediate(value=1.5, is_hex=False)
  py3: typing.Any = synth._convert_operand_to_py(imm3)
  assert isinstance(py3, cst.Float)
  assert py3.value == "1.5"


def test_nvidia_sass_synth_convert_operand_to_py_string() -> None:
  """Verify complex or custom operands default to bracketed string CST.

  This test checks that arbitrary operand types whose string representation contains
  brackets or special syntax are correctly mapped by the synthesizer into
  `libcst.SimpleString` nodes wrapping their stringified representation.

  Args:
      None

  Returns:
      None
  """
  synth = NvidiaSassSynthesizer(MagicMock())

  # Bracketed/Complex
  class DummyOp:
    """A dummy operand type with bracketed string representation."""

    def __str__(self) -> str:
      """Return string representation.

      Args:
          None

      Returns:
          str: The string representation.
      """
      return "[R0]"

    def to_text(self) -> str:
      """Return the text representation of the dummy operand.

      Returns:
          str: The braced string "[v0]".
      """
      return "[R0]"

  py1: typing.Any = synth._convert_operand_to_py(DummyOp())  # type: ignore
  assert isinstance(py1, cst.SimpleString)
  assert py1.value == "'[R0]'"


def test_nvidia_sass_backend_default_init() -> None:
  """Verify default initialization of the NvidiaSassBackend.

  This test instantiates NvidiaSassBackend with default arguments and asserts that
  its internal synthesizer is correctly initialized with a valid semantics manager.

  Args:
      None

  Returns:
      None
  """
  backend = NvidiaSassBackend()
  assert backend.synthesizer.semantics is not None


def test_nvidia_sass_synthesizer_empty_output() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
  from ml_switcheroo.semantics.manager import SemanticsManager

  semantics = SemanticsManager()
  synthesizer = NvidiaSassSynthesizer(semantics)

  graph = LogicalGraph(name="test")
  graph.nodes.append(LogicalNode(id="out", kind="Output"))

  res: typing.Any = synthesizer.from_graph(graph)
  assert res is not None
