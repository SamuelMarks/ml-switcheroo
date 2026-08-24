"""Test module."""

import pytest
import libcst as cst
from unittest.mock import patch
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode as Node, LogicalEdge as Edge
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassInstruction,
  SassRegister,
  SassImmediate,
  SassPredicate,
  SassComment,
  SassLabel,
  SassMemory,
)
from ml_switcheroo.semantics.manager import SemanticsManager


def test_register_allocator():
  """Test element."""
  alloc = RegisterAllocator()

  # get_register
  r1 = alloc.get_register("var1")
  assert r1.name == "R0"

  r1_again = alloc.get_register("var1")
  assert r1_again.name == "R0"

  r2 = alloc.get_register("var2")
  assert r2.name == "R1"

  # free_register
  alloc.free_register("var1")
  alloc.get_register("var3")

  # allocate_temp
  rtemp = alloc.allocate_temp()
  assert rtemp.name.startswith("R")

  # reset
  alloc.reset()
  r4 = alloc.get_register("var4")
  assert r4.name == "R0"


def test_register_allocator_overflow():
  """Test element."""
  alloc = RegisterAllocator()
  alloc.reset()
  for i in range(255):
    alloc.get_register(f"v{i}")
  with pytest.raises(ValueError, match="overflow"):
    alloc.get_register("v256")


def test_register_allocator_liveness():
  """Test element."""
  alloc = RegisterAllocator()
  graph = LogicalGraph()
  graph.nodes.append(Node(id="n1", kind="test", metadata={}))
  graph.nodes.append(Node(id="n2", kind="test", metadata={}))
  graph.edges.append(Edge(source="n1", target="n2"))
  graph.edges.append(Edge(source="n1", target="n2"))

  alloc.build_liveness(graph)
  assert alloc._liveness_map["n1"] == 2

  alloc.get_register("n1")
  alloc.record_usage("n1")
  assert "n1" in alloc._var_to_reg
  alloc.record_usage("n1")
  assert "n1" not in alloc._var_to_reg  # should be freed

  # Test record_usage on nonexistent var
  alloc.record_usage("nonexistent")


def test_synthesizer_init(tmp_path):
  """Test element."""
  # test reading macros.json
  macros_json = tmp_path / "macros.json"
  macros_json.write_text('{"Conv2d": "expand_conv2d"}')

  sem = SemanticsManager()

  with patch("ml_switcheroo.core.compiler.backends.sass.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.sass.synthesizer.os.path.exists", return_value=True):
      synth = SassSynthesizer(sem)
      assert "Conv2d" in synth.macro_registry


def test_synthesizer_from_graph():
  """Test element."""
  sem = SemanticsManager()
  synth = SassSynthesizer(sem)

  graph = LogicalGraph()
  n_in = Node(id="in1", kind="Input", metadata={"name": "input_x"})
  n_add = Node(id="add1", kind="Add", metadata={})
  n_conv = Node(id="conv1", kind="Conv2d", metadata={"k": 3})
  n_out = Node(id="out1", kind="Output", metadata={})

  graph.nodes.extend([n_in, n_add, n_conv, n_out])

  graph.edges.append(Edge(source="in1", target="add1"))
  graph.edges.append(Edge(source="add1", target="conv1"))
  graph.edges.append(Edge(source="conv1", target="out1"))

  def mock_get_definition(kind):
    return [kind]

  def mock_resolve_variant(abstract_id, target):
    if abstract_id == "Add":
      return {"api": "FADD"}
    return None

  # Make sure semantics resolves Add -> FADD
  with patch.object(sem, "get_definition", side_effect=mock_get_definition):
    with patch.object(sem, "resolve_variant", side_effect=mock_resolve_variant):
      # Also patch macro registry to hit Conv2d macro
      from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv2d

      synth.macro_registry = {"Conv2d": expand_conv2d}

      nodes = synth.from_graph(graph)

      assert len(nodes) > 0
      # Check input comment
      assert any(isinstance(n, SassComment) and "Input input_x" in n.text for n in nodes)
      # Check FADD instruction
      assert any(isinstance(n, SassInstruction) and n.opcode == "FADD" for n in nodes)
      # Check Conv2d macro expansion
      assert any(isinstance(n, SassComment) and "BEGIN Conv2d" in n.text for n in nodes)
      # Check Output comment
      assert any(isinstance(n, SassComment) and "Return" in n.text for n in nodes)


def test_synthesizer_from_graph_unmapped_op():
  """Test element."""
  sem = SemanticsManager()
  synth = SassSynthesizer(sem)

  graph = LogicalGraph()
  graph.nodes.append(Node(id="n1", kind="UnknownOp", metadata={}))

  with patch.object(sem, "get_definition", return_value=None):
    with patch.object(sem, "resolve_variant", return_value=None):
      nodes = synth.from_graph(graph)
      assert any(isinstance(n, SassComment) and "Unmapped Op: UnknownOp" in n.text for n in nodes)


def test_synthesizer_from_graph_method_suffix():
  """Test element."""
  sem = SemanticsManager()
  synth = SassSynthesizer(sem)

  graph = LogicalGraph()
  # test suffix macro match e.g. "reshape"
  graph.nodes.append(Node(id="n1", kind="tensor.reshape", metadata={}))

  with patch.object(sem, "get_definition", return_value=["tensor.reshape"]):

    def fake_expand(alloc, node_id, meta):
      return [SassComment(text="fake_reshape")]

    synth.macro_registry = {"reshape": fake_expand}
    nodes = synth.from_graph(graph)
    assert any(isinstance(n, SassComment) and "fake_reshape" in n.text for n in nodes)


def test_synthesizer_to_python():
  """Test element."""
  sem = SemanticsManager()
  synth = SassSynthesizer(sem)

  nodes = [
    # basic inst
    SassInstruction(opcode="FADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]),
    # branch (no dest)
    SassInstruction(opcode="BRA", operands=[SassLabel(name="L1")], predicate=SassPredicate(name="P0")),
    # store without dots to avoid CSTValidationError
    SassInstruction(opcode="STG", operands=[SassMemory(base=SassRegister(name="R0")), SassRegister(name="R1")]),
    # immediate hex
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassImmediate(value=255, is_hex=True)]),
    # immediate float
    SassInstruction(opcode="MOV", operands=[SassRegister(name="R0"), SassImmediate(value=1.5)]),
    # NOP
    SassInstruction(opcode="NOP", operands=[]),
    # Labels and Comments
    SassLabel(name="L1"),
    SassComment(text="BEGIN Loop"),
    SassComment(text="END Loop"),
  ]

  py_mod = synth.to_python(nodes)
  code = cst.Module(body=py_mod.body).code

  assert "R0 = sass.FADD(R1, R2)" in code
  assert "sass.BRA('L1:', predicate = 'P0')" in code
  assert "sass.STG('[R0]', R1)" in code
  assert "sass.MOV(0xff)" in code
  assert "sass.MOV(1.5)" in code
  assert "sass.NOP()" in code
