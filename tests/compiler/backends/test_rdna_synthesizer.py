"""Tests for RDNA Synthesizer, Register Allocator, and the RDNA assembly backend.

This test module verifies the correctness of the RDNA (Radeon DNA) compilation backend
components including the RegisterAllocator for VGPRs and SGPRs, the RdnaSynthesizer
for translating logical graphs to low-level Pythonic representations of assembly instructions,
and the RdnaBackend high-level configuration.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer, RdnaBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaInstruction,
  RdnaVGPR,
  RdnaSGPR,
  RdnaComment,
  RdnaLabel,
  RdnaImmediate,
)
from ml_switcheroo.semantics.manager import SemanticsManager


def test_register_allocator_vgpr():
  """Test the allocation of Vector General Purpose Registers (VGPRs).

  This test verifies that:
  1. Vector registers are sequentially allocated (v0, v1, etc.).
  2. Requesting a register for the same variable multiple times returns the same
     register instance.
  3. String conversion of a VGPR produces the expected RDNA assembly format.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  reg1 = allocator.get_vector_register("var1")
  assert reg1.index == 0
  assert str(reg1) == "v0"

  reg2 = allocator.get_vector_register("var2")
  assert reg2.index == 1

  # Same var should return same register
  reg1_again = allocator.get_vector_register("var1")
  assert reg1_again.index == 0


def test_register_allocator_sgpr():
  """Test the allocation of Scalar General Purpose Registers (SGPRs).

  This test verifies that:
  1. Scalar registers are sequentially allocated (s0, s1, etc.).
  2. Requesting a register for the same variable multiple times returns the same
     register instance.
  3. String conversion of an SGPR produces the expected RDNA assembly format.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  reg1 = allocator.get_scalar_register("var1")
  assert reg1.index == 0
  assert str(reg1) == "s0"

  reg2 = allocator.get_scalar_register("var2")
  assert reg2.index == 1

  reg1_again = allocator.get_scalar_register("var1")
  assert reg1_again.index == 0


def test_register_allocator_temps():
  """Test allocation of temporary scalar and vector registers.

  This test verifies that:
  1. A temporary VGPR is allocated and indexed correctly.
  2. A temporary SGPR is allocated and indexed correctly.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  vgpr = allocator.allocate_vector_temp()
  assert vgpr.index == 0

  sgpr = allocator.allocate_scalar_temp()
  assert sgpr.index == 0


def test_register_allocator_overflow_vgpr():
  """Test that vector register allocation overflows gracefully.

  This test verifies that:
  1. Mocking the allocator to exceed its maximum limit of 256 VGPRs raises a
     ValueError containing 'RdnaVGPR overflow'.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  allocator._next_vgpr = 256
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_vector_register("overflow")


def test_register_allocator_overflow_sgpr():
  """Test that scalar register allocation overflows gracefully.

  This test verifies that:
  1. Mocking the allocator to exceed its maximum limit of 106 SGPRs raises a
     ValueError containing 'RdnaSGPR overflow'.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  allocator._next_sgpr = 106
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_scalar_register("overflow")


def test_register_allocator_reset():
  """Test resetting the register allocator state.

  This test verifies that:
  1. Allocating some scalar/vector registers and calling reset() clears all internal
     allocator counters and returns next register indices back to 0.

  Args:
      None.

  Returns:
      None.
  """
  allocator = RegisterAllocator()
  allocator.get_vector_register("var1")
  allocator.get_scalar_register("var1")
  allocator.reset()
  assert allocator._next_vgpr == 0
  assert allocator._next_sgpr == 0


def test_rdna_synth_from_graph_inputs_outputs():
  """Test assembly code synthesis from a logical graph with inputs and outputs.

  This test verifies that:
  1. A graph containing an Input node mapped to an Output node results in the expected
     RdnaComment entries with mapped input register (v0) and return notation.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = RdnaSynthesizer(mock_semantics)

  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="in1", kind="Input", metadata={"name": "x"}))
  graph.nodes.append(LogicalNode(id="out1", kind="Output"))
  graph.edges.append(LogicalEdge(source="in1", target="out1"))

  nodes = synth.from_graph(graph)
  assert len(nodes) == 2
  assert isinstance(nodes[0], RdnaComment)
  assert "Input x -> v0" in nodes[0].text
  assert isinstance(nodes[1], RdnaComment)
  assert "Return: v0" in nodes[1].text


def test_rdna_synth_from_graph_unmapped():
  """Test synthesis behavior when encountering an unknown/unmapped logical op.

  This test verifies that:
  1. If semantics cannot resolve an operation, the synthesizer creates an
     unmapped op comment block in the output assembly list, preventing crashes.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  mock_semantics.get_definition.return_value = None
  mock_semantics.resolve_variant.return_value = None
  synth = RdnaSynthesizer(mock_semantics)

  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="op1", kind="UnknownOp"))

  nodes = synth.from_graph(graph)
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaComment)
  assert "Unmapped Op: UnknownOp" in nodes[0].text


def test_rdna_synth_from_graph_valid_op():
  """Test that a valid logical node operation correctly synthesizes to an RDNA instruction.

  This test verifies that:
  1. An "Add" logical operation with two input nodes is successfully mapped to
     a three-operand RdnaInstruction (`v_add_f32`) representing destination, source1, and source2.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "v_add_f32"}
  synth = RdnaSynthesizer(mock_semantics)

  graph = LogicalGraph()
  graph.nodes.append(LogicalNode(id="in1", kind="Input"))
  graph.nodes.append(LogicalNode(id="in2", kind="Input"))
  graph.nodes.append(LogicalNode(id="add1", kind="Add"))
  graph.edges.append(LogicalEdge("in1", "add1"))
  graph.edges.append(LogicalEdge("in2", "add1"))

  nodes = synth.from_graph(graph)
  assert len(nodes) == 3
  # Two inputs, one op
  assert isinstance(nodes[2], RdnaInstruction)
  assert nodes[2].opcode == "v_add_f32"
  assert len(nodes[2].operands) == 3  # dest, src1, src2


def test_rdna_synth_to_python_label():
  """Test converting an RdnaLabel node to executable Python CST.

  This test verifies that:
  1. Translating an RdnaLabel object generates a commented RDNA assembly representation
     of the label inside the target Python source code.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = RdnaSynthesizer(mock_semantics)

  nodes = [RdnaLabel(name="L_LOOP")]
  module = synth.to_python(nodes)
  code = module.code
  assert "# RdnaLabel: L_LOOP" in code


def test_rdna_synth_to_python_instruction_empty_operands():
  """Test converting an RdnaInstruction without operands to a Python CST block.

  This test verifies that:
  1. A no-operand instruction like `s_nop` synthesizes to `rdna.s_nop()` in the
     resulting Python source code.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = RdnaSynthesizer(mock_semantics)

  inst = RdnaInstruction(opcode="s_nop", operands=[])
  module = synth.to_python([inst])
  code = module.code
  assert "rdna.s_nop()" in code


def test_rdna_synth_to_python_instruction_store():
  """Test converting a memory store instruction into Python CST format.

  This test verifies that:
  1. A `buffer_store_dword` operation with VGPR and SGPR operands synthesizes
     to a non-assignment method call like `rdna.buffer_store_dword(v0, s0)`.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = RdnaSynthesizer(mock_semantics)

  inst = RdnaInstruction(opcode="buffer_store_dword", operands=[RdnaVGPR(index=0), RdnaSGPR(index=0)])
  module = synth.to_python([inst])
  code = module.code
  assert "rdna.buffer_store_dword(v0, s0)" in code


def test_rdna_synth_to_python_instruction_alu():
  """Test converting a standard ALU/arithmetic instruction into Python CST format.

  This test verifies that:
  1. A 3-operand vector add (`v_add_f32` destination, source1, source2) is synthesized
     correctly as an assignment expression: `v0 = rdna.v_add_f32(v1, v2)`.

  Args:
      None.

  Returns:
      None.
  """
  mock_semantics = MagicMock(spec=SemanticsManager)
  synth = RdnaSynthesizer(mock_semantics)

  inst = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  module = synth.to_python([inst])
  code = module.code
  assert "v0 = rdna.v_add_f32(v1, v2)" in code


def test_rdna_synth_convert_operand_to_py_immediates():
  """Test converting RdnaImmediate operands into equivalent LibCST nodes.

  This test verifies that:
  1. Standard integers are translated to cst.Integer.
  2. Hexadecimal integers are translated to their formatted string (e.g., "0xff").
  3. Float values are translated to cst.Float.

  Args:
      None.

  Returns:
      None.
  """
  synth = RdnaSynthesizer(MagicMock())

  imm1 = RdnaImmediate(value=10, is_hex=False)
  py1 = synth._convert_operand_to_py(imm1)
  assert isinstance(py1, cst.Integer)
  assert py1.value == "10"

  imm2 = RdnaImmediate(value=255, is_hex=True)
  py2 = synth._convert_operand_to_py(imm2)
  assert isinstance(py2, cst.Integer)
  assert py2.value == "0xff"

  imm3 = RdnaImmediate(value=1.5, is_hex=False)
  py3 = synth._convert_operand_to_py(imm3)
  assert isinstance(py3, cst.Float)
  assert py3.value == "1.5"


def test_rdna_synth_convert_operand_to_py_string():
  """Test converting custom or complex string-based operands into Python LibCST nodes.

  This test verifies that:
  1. Braced operands such as `[v0]` are sanitized into valid Python names (`_v0`).
  2. Operands with special characters (e.g., `something!`) are safely converted to Python strings.

  Args:
      None.

  Returns:
      None.
  """
  synth = RdnaSynthesizer(MagicMock())

  # Bracketed
  class DummyOp:
    """A dummy RDNA operand class used to simulate braced operand formatting."""

    def __str__(self) -> str:
      """Return the string representation of the dummy operand.

      Returns:
          str: The braced string "[v0]".
      """
      return "[v0]"

    def to_text(self) -> str:
      """Return the text representation of the dummy operand.

      Returns:
          str: The braced string "[v0]".
      """
      return "[v0]"

  py1 = synth._convert_operand_to_py(DummyOp())
  assert isinstance(py1, cst.Name)
  assert py1.value == "_v0"

  # Special chars
  class DummyOp2:
    """A dummy RDNA operand class used to simulate operands with special characters."""

    def __str__(self) -> str:
      """Return the string representation of the dummy operand.

      Returns:
          str: The string "something!".
      """
      return "something!"

    def to_text(self) -> str:
      """Return the text representation of the dummy operand.

      Returns:
          str: The string "something!".
      """
      return "something!"

  py2 = synth._convert_operand_to_py(DummyOp2())
  assert isinstance(py2, cst.SimpleString)
  assert py2.value == "'something!'"


def test_rdna_backend_default_init():
  """Test that RdnaBackend is correctly initialized with default configuration.

  This test verifies that:
  1. The default backend instances are setup with active semantics mapping.
  2. The default target architecture is set to "gfx1030".

  Args:
      None.

  Returns:
      None.
  """
  backend = RdnaBackend()
  assert backend.synthesizer.semantics is not None
  assert backend.target_arch == "gfx1030"


def test_rdna_synthesizer_empty_output():
  """Test RDNA synthesizer with an Output node with no inputs."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  semantics = SemanticsManager()
  synthesizer = RdnaSynthesizer(semantics)

  graph = LogicalGraph(name="test")
  graph.nodes.append(LogicalNode(id="out", kind="Output"))

  print("I RAN!!!")
  res = synthesizer.from_graph(graph)
  assert res is not None
