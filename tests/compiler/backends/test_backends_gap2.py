"""Test suite for the Backends Gap2 module."""

import pytest


@pytest.mark.skip(reason="torch.add removed")
def test_rdna_synthesizer_gaps():
  """Verifies the behavior of RDNA synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator()
  for _ in range(256):
    alloc.allocate_vector_temp()
  import pytest

  with pytest.raises(ValueError):
    alloc.allocate_vector_temp()
  for _ in range(106):
    alloc.allocate_scalar_temp()
  with pytest.raises(ValueError):
    alloc.allocate_scalar_temp()
  pass
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "torch.add", {"arg_1": "a", "arg_2": "b"}))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend

  SassBackend()
  assert "v0 =" in mod.code


def test_rdna_synthesizer_py_translation():
  """Verifies the behavior of RDNA synthesizer py translation."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import (
    RdnaInstruction as Instruction,
    RdnaLabel as Label,
    RdnaImmediate as Immediate,
    RdnaVGPR as VGPR,
    RdnaSGPR as SGPR,
    RdnaMemory as Memory,
  )
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  nodes = [
    Instruction(opcode="v_add_f32", operands=[]),
    Instruction(opcode="store_dword", operands=[VGPR(index=0, count=1), Immediate(value=5, is_hex=True)]),
    Instruction(opcode="branch", operands=[Label(name="L1")]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=1, count=1), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="v_mov_b32", operands=[VGPR(index=2, count=1), Immediate(value=42, is_hex=False)]),
    Instruction(opcode="s_load", operands=[SGPR(index=0, count=2), Memory(base=VGPR(index=3, count=1))]),
    Label(name="L1"),
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend

  SassBackend()
  code = mod.code
  assert "rdna.v_add_f32" in code
  assert "rdna.store_dword" in code
  assert "rdna.branch" in code
  assert "0x5" in code
  assert "3.14" in code
  assert "42" in code


def test_rdna_synthesizer_io():
  """Verifies the behavior of RDNA synthesizer I/O."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(
    nodes=[LogicalNode("in", "Input", {"name": "x"}), LogicalNode("out", "Output")], edges=[LogicalEdge("in", "out")]
  )
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabelRef as LabelRef

  res = synth._convert_operand_to_py(LabelRef("[var]"))
  assert res.value == "_var"


def test_rdna_synthesizer_misc():
  """Verifies the behavior of RDNA synthesizer misc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  alloc = RegisterAllocator()
  alloc._var_to_sgpr["test"] = 0
  s = alloc.get_scalar_register("test")
  assert s.index == 0
  synth = RdnaSynthesizer(SemanticsManager())

  class MockSemantics:
    """Mock Semantics class for testing purposes."""

    def resolve_variant(self, node_id, tgt):
      """Mock implementation of resolve variant."""
      return {"api": "v_add_f32"}

    def get_definition(self, kind):
      """Mock implementation of get definition."""
      return None

  synth = RdnaSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[LogicalNode("src", "src_op"), LogicalNode("dst", "dst_op")], edges=[LogicalEdge("src", "dst")])
  synth.from_graph(g)


def test_sass_macros_linear():
  """Verifies the behavior of SASS macros linear."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_linear
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_linear(alloc, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10


def test_sass_synthesizer_gaps():
  """Verifies the behavior of SASS synthesizer gaps."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import (
    SassInstruction as Instruction,
    SassLabel as Label,
    SassComment as Comment,
    SassImmediate as Immediate,
    SassRegister as Register,
    SassMemory as Memory,
  )
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator()
  for _ in range(255):
    alloc.allocate_temp()
  import pytest

  with pytest.raises(ValueError):
    alloc.allocate_temp()

  class MockSemantics:
    """Mock Semantics class for testing purposes."""

    def get_definition(self, kind):
      """Mock implementation of get definition."""
      return None

    def resolve_variant(self, abstract_id, tgt):
      """Mock implementation of resolve variant."""
      if abstract_id == "Missing":
        return {}
      return {"api": "FADD"}

  synth = SassSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "Missing"))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  assert "Unmapped Op:" in str(nodes[0])
  nodes = [
    Instruction(opcode="FADD", operands=[]),
    Instruction(opcode="FADD", operands=[Register(name="R1")], predicate="P0"),
    Comment(text="test"),
    Label(name="L1"),
    Instruction(opcode="STG", operands=[Memory(base=Register(name="R1")), Immediate(value=1)]),
    Instruction(opcode="BRA", operands=[Label(name="L1")]),
    Instruction(opcode="MOV", operands=[Register(name="R2"), Immediate(value=1, is_hex=True)]),
    Instruction(opcode="FMUL", operands=[Register(name="R3"), Immediate(value=3.14, is_hex=False)]),
    Instruction(opcode="MOV", operands=[Register(name="R4"), Register(name="R1")]),
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend

  SassBackend()
  code = mod.code
  assert "sass.FADD" in code
  assert "Label: L1" in code
  assert "0x1" in code
  assert "3.14" in code
