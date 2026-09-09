"""Test module."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator, NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassNode,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.graph import LogicalEdge as Edge
from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.core.graph import LogicalNode as Node
from ml_switcheroo.semantics.manager import SemanticsManager


def test_register_allocator() -> None:
  """Docstring."""
  alloc: RegisterAllocator = RegisterAllocator()

  # get_register
  r1: NvidiaSassRegister = alloc.get_register("var1")
  assert r1.name == "R0"

  r1_again: NvidiaSassRegister = alloc.get_register("var1")
  assert r1_again.name == "R0"

  r2: NvidiaSassRegister = alloc.get_register("var2")
  assert r2.name == "R1"

  alloc.free_register("var1")
  assert len(alloc._free_pool) == 254
  # freeing already freed
  alloc.free_register("var1")

  temp = alloc.allocate_temp()
  assert temp.name == "R2"

  # free_register
  alloc.free_register("var1")
  alloc.get_register("var3")

  # allocate_temp
  rtemp: NvidiaSassRegister = alloc.allocate_temp()
  assert rtemp.name.startswith("R")

  # reset
  alloc.reset()
  r4: NvidiaSassRegister = alloc.get_register("var4")
  assert r4.name == "R0"


def test_register_allocator_overflow() -> None:
  """Docstring."""
  alloc: RegisterAllocator = RegisterAllocator()
  alloc.reset()
  for i in range(255):
    alloc.get_register(f"v{i}")
  with pytest.raises(ValueError, match="overflow"):
    alloc.get_register("v256")


def test_register_allocator_liveness() -> None:
  """Docstring."""
  alloc: RegisterAllocator = RegisterAllocator()
  graph: LogicalGraph = LogicalGraph()
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


def test_synthesizer_init(tmp_path: Path) -> None:
  """Docstring."""
  # test reading macros.json
  macros_json: Path = tmp_path / "macros.json"
  macros_json.write_text('{"Conv2d": "expand_conv2d", "UnknownOp": "expand_unknown"}')

  sem: SemanticsManager = SemanticsManager()

  with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.exists", return_value=True):
      synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)
      assert "Conv2d" in synth.macro_registry
      assert "UnknownOp" not in synth.macro_registry

  with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.exists", return_value=False):
    synth2: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)
    assert "Conv2d" not in synth2.macro_registry


def test_synthesizer_init_yaml(tmp_path: Path) -> None:
  """Tests synthesizer initialization when macros.yaml is present."""
  macros_yaml: Path = tmp_path / "macros.yaml"
  macros_yaml.write_text("Conv2d: expand_conv2d\nUnknownOp: expand_unknown\n")

  sem: SemanticsManager = SemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)
    assert "Conv2d" in synth.macro_registry
    assert "UnknownOp" not in synth.macro_registry


def test_synthesizer_init_filenotfound(tmp_path: Path) -> None:
  """Tests synthesizer initialization when files exist in os.path.exists check but fail on open."""
  sem: SemanticsManager = SemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer.os.path.exists", return_value=True):
      synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)
      assert synth.macro_registry == {}


def test_synthesizer_from_graph() -> None:
  """Docstring."""
  sem: SemanticsManager = SemanticsManager()
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)

  graph: LogicalGraph = LogicalGraph()
  n_in: Node = Node(id="in1", kind="Input", metadata={"name": "input_x"})
  n_add: Node = Node(id="add1", kind="Add", metadata={})
  n_conv: Node = Node(id="conv1", kind="Conv2d", metadata={"k": "3"})
  n_out: Node = Node(id="out1", kind="Output", metadata={})

  graph.nodes.extend([n_in, n_add, n_conv, n_out])

  graph.edges.append(Edge(source="in1", target="add1"))
  graph.edges.append(Edge(source="add1", target="conv1"))
  graph.edges.append(Edge(source="conv1", target="out1"))

  def mock_get_definition(kind: str) -> List[str]:
    """Docstring."""
    return [kind]

  def mock_resolve_variant(abstract_id: str, target: str) -> Optional[Dict[str, str]]:
    """Docstring."""
    if abstract_id == "Add":
      return {"api": "FADD"}
    return None

  # Make sure semantics resolves Add -> FADD
  with patch.object(sem, "get_definition", side_effect=mock_get_definition):
    with patch.object(sem, "resolve_variant", side_effect=mock_resolve_variant):
      # Also patch macro registry to hit Conv2d macro
      from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_conv2d

      synth.macro_registry = {"Conv2d": expand_conv2d}

      nodes: List[NvidiaSassNode] = synth.from_graph(graph)

      assert len(nodes) > 0
      # Check input comment
      assert any(isinstance(n, NvidiaSassComment) and "Input input_x" in n.text for n in nodes)
      # Check FADD instruction
      assert any(isinstance(n, NvidiaSassInstruction) and n.opcode == "FADD" for n in nodes)
      # Check Conv2d macro expansion
      assert any(isinstance(n, NvidiaSassComment) and "BEGIN Conv2d" in n.text for n in nodes)
      # Check Output comment
      assert any(isinstance(n, NvidiaSassComment) and "Return" in n.text for n in nodes)


def test_synthesizer_from_graph_unmapped_op() -> None:
  """Docstring."""
  sem: SemanticsManager = SemanticsManager()
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)

  graph: LogicalGraph = LogicalGraph()
  # Hit 216->218: edge source to a target already seen
  graph.nodes.append(Node(id="n1", kind="UnknownOp", metadata={}))
  graph.nodes.append(Node(id="n2", kind="UnknownOp2", metadata={}))
  graph.edges.append(Edge(source="n1", target="n2"))
  graph.edges.append(Edge(source="n1", target="n2"))  # Duplicate edge

  # Hit 231->220: Output with no sources
  graph.nodes.append(Node(id="n3", kind="Output", metadata={}))

  # Hit 275->278: Node without abstract_id
  graph.nodes.append(Node(id="n4", kind="", metadata={}))

  with patch.object(sem, "get_definition", return_value=None):
    with patch.object(sem, "resolve_variant", return_value=None):
      nodes: List[NvidiaSassNode] = synth.from_graph(graph)
      assert any(isinstance(n, NvidiaSassComment) and "Unmapped Op: UnknownOp" in n.text for n in nodes)


def test_synthesizer_from_graph_method_suffix() -> None:
  """Docstring."""
  sem: SemanticsManager = SemanticsManager()
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)

  graph: LogicalGraph = LogicalGraph()
  # test suffix macro match e.g. "reshape"
  graph.nodes.append(Node(id="n0", kind="input", metadata={}))
  graph.nodes.append(Node(id="n1", kind="tensor.reshape", metadata={}))
  graph.edges.append(Edge(source="n0", target="n1"))

  with patch.object(sem, "get_definition", return_value=["tensor.reshape"]):

    def fake_expand(alloc: RegisterAllocator, node_id: str, meta: Dict[str, Any]) -> List[NvidiaSassNode]:
      """Docstring."""
      return [NvidiaSassComment(text="fake_reshape")]

    synth.macro_registry = {"reshape": fake_expand}
    nodes: List[NvidiaSassNode] = synth.from_graph(graph)
    assert any(isinstance(n, NvidiaSassComment) and "fake_reshape" in n.text for n in nodes)


def test_synthesizer_to_python() -> None:
  """Docstring."""
  sem: SemanticsManager = SemanticsManager()
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(sem)

  nodes: List[NvidiaSassNode] = [
    # basic inst
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")],
    ),
    # branch (no dest)
    NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassLabel(name="L1")], predicate=NvidiaSassPredicate(name="P0")),
    # store without dots to avoid CSTValidationError
    NvidiaSassInstruction(
      opcode="STG", operands=[NvidiaSassMemory(base=NvidiaSassRegister(name="R0")), NvidiaSassRegister(name="R1")]
    ),
    # immediate hex
    NvidiaSassInstruction(
      opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=255, is_hex=True)]
    ),
    # immediate float
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=1.5)]),
    # NOP
    NvidiaSassInstruction(opcode="NOP", operands=[]),
    # Labels and Comments
    NvidiaSassLabel(name="L1"),
    NvidiaSassComment(text="BEGIN Loop"),
    NvidiaSassComment(text="END Loop"),
    # Missing comment
    NvidiaSassComment(text="Just a normal comment"),
    # Dest not identifier
    NvidiaSassInstruction(
      opcode="FADD", operands=[NvidiaSassMemory(base=NvidiaSassRegister(name="R0")), NvidiaSassRegister(name="R1")]
    ),
    # Dest identifier but not register
    NvidiaSassInstruction(opcode="FADD", operands=[NvidiaSassPredicate(name="P0"), NvidiaSassRegister(name="R1")]),
    # Immediate int
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=123)]),
  ]

  class FakeNode(NvidiaSassNode):
    """Fake node for testing."""

    def __str__(self) -> str:
      """Return string representation."""
      return "fake"

  nodes.append(FakeNode())

  py_mod: cst.Module = synth.to_python(nodes)
  code: str = cst.Module(body=py_mod.body).code

  assert "R0 = nvidia_sass.FADD(R1, R2)" in code
  assert "nvidia_sass.BRA('L1:', predicate = 'P0')" in code
  assert "nvidia_sass.STG('[R0]', R1)" in code
  assert "nvidia_sass.MOV(0xff)" in code
  assert "nvidia_sass.MOV(1.5)" in code
  assert "nvidia_sass.NOP()" in code
  assert "nvidia_sass.FADD(R1)" in code
  assert "P0 = nvidia_sass.FADD(R1)" in code
  assert "nvidia_sass.MOV(123)" in code
