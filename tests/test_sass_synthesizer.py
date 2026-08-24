"""Docstring."""

from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer
from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassRegister,
  SassInstruction,
  SassImmediate,
  SassLabel,
  SassOperand,
  SassMemory,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
import pytest


class DummySassOperand(SassOperand):
  """Docstring."""

  def __init__(self, raw):
    """Docstring."""
    self.raw = raw

  def __str__(self):
    """Docstring."""
    return self.raw


def test_register_allocator():
  """Docstring."""
  allocator = RegisterAllocator()
  reg = allocator.get_register("var1")
  assert reg.name == "R0"
  reg2 = allocator.get_register("var1")
  assert reg2.name == "R0"

  tmp_reg = allocator.allocate_temp()
  assert tmp_reg.name == "R1"

  allocator.reset()
  assert allocator.get_register("var3").name == "R0"


def test_register_allocator_overflow():
  """Docstring."""
  allocator = RegisterAllocator()
  allocator._free_pool = []
  with pytest.raises(ValueError, match="SassRegister overflow"):
    allocator.allocate_temp()


def test_synthesizer_from_graph():
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind):
      """Docstring."""
      if kind == "UnknownNode":
        return None
      if kind == "DirectMatch":
        return ("DirectMatch", {})
      if kind == "FallbackOp":
        return ("FallbackOp", {})
      return ("abstract.Linear", {})

    def resolve_variant(self, abstract_id, target):
      """Docstring."""
      if abstract_id == "FallbackOp":
        return {"api": "FADD", "args": []}
      return {"api": "NOP", "args": []}

  synth = SassSynthesizer(semantics=MockSemantics())

  def dummy_expander(alloc, nid, meta):
    """Docstring."""
    return [SassInstruction("NOP", [])]

  synth.macro_registry = {"Linear": dummy_expander, "DirectMatch": dummy_expander}

  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="conv1", kind="Conv2d", metadata={"k": 3}),
      LogicalNode(id="lin1", kind="Linear"),
      LogicalNode(id="dir1", kind="DirectMatch"),
      LogicalNode(id="fall1", kind="FallbackOp"),
      LogicalNode(id="out1", kind="Output"),
      LogicalNode(id="unkn1", kind="UnknownNode"),
    ],
    edges=[
      LogicalEdge("in1", "conv1"),
      LogicalEdge("conv1", "lin1"),
      LogicalEdge("in1", "dir1"),
      LogicalEdge("in1", "fall1"),
      LogicalEdge("lin1", "out1"),
    ],
  )

  nodes = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_from_graph_unmapped():
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind):
      """Docstring."""
      return ("abstract", {})

    def resolve_variant(self, abstract_id, target):
      """Docstring."""
      return None

  synth = SassSynthesizer(semantics=MockSemantics())
  synth.macro_registry = {}

  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="op1", kind="UnknownOp"),
    ],
    edges=[],
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) == 1
  assert "Unmapped Op" in str(nodes[0])


def test_synthesizer_to_python():
  """Docstring."""
  synth = SassSynthesizer(semantics=None)

  nodes = [
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]),
    SassLabel(name="label1"),
    SassInstruction(opcode="NOP", operands=[]),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), DummySassOperand("R[0:1]")], predicate="@P0"),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), DummySassOperand("#hashtag")]),
    SassInstruction(opcode="MOV", operands=[DummySassOperand("-R0"), SassImmediate(value=42)]),  # dest is not identifier
  ]
  tree = synth.to_python(nodes)
  code = tree.code
  assert "R0 =" in code
  assert "NOP" in code
  assert "'R[0:1]'" in code
  assert "'#hashtag'" in code
  assert "SassLabel" in code
  assert "predicate" in code


def test_synthesizer_to_python_operands():
  """Docstring."""
  synth = SassSynthesizer(semantics=None)
  nodes = [
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassImmediate(value=42)]),
    SassInstruction(opcode="BRA", operands=[SassLabel(name="label1")]),
    SassInstruction(opcode="STG", operands=[SassMemory(base=SassRegister(name="R0"), offset=4)]),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R1"), SassImmediate(value=1.5, is_hex=False)]),
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R2"), SassImmediate(value=1, is_hex=True)]),
  ]
  tree = synth.to_python(nodes)
  code = tree.code
  assert "42" in code
  assert "1.5" in code
  assert "STG" in code


def test_backend_compile():
  """Docstring."""
  backend = SassBackend()
  graph = LogicalGraph(nodes=[LogicalNode(id="in1", kind="Input")], edges=[])
  code = backend.compile(graph)
  assert "Input in1" in code
