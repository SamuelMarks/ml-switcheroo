"""Test suite for the Rdna Synthesizer module."""

import typing
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import (
  MAX_SGPR,
  MAX_VGPR,
  RdnaBackend,
  RdnaSynthesizer,
  RegisterAllocator,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaOperand,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager


def test_allocator_dual_pools() -> None:
  """Verifies the behavior of allocator dual pools."""
  alloc = RegisterAllocator()
  v0: RdnaVGPR = alloc.get_vector_register("x")
  s0: RdnaSGPR = alloc.get_scalar_register("cnt")
  assert isinstance(v0, RdnaVGPR)
  assert v0.index == 0
  assert isinstance(s0, RdnaSGPR)
  assert s0.index == 0


def test_allocator_reuse() -> None:
  """Verifies the behavior of allocator reuse."""
  alloc = RegisterAllocator()
  v_a: RdnaVGPR = alloc.get_vector_register("a")
  v_b: RdnaVGPR = alloc.get_vector_register("a")
  assert v_a.index == v_b.index == 0


def test_allocator_overflow_vgpr() -> None:
  """Verifies the behavior of allocator overflow vgpr."""
  alloc = RegisterAllocator()
  alloc._next_vgpr = MAX_VGPR + 1
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    alloc.get_vector_register("fail")


def test_allocator_overflow_sgpr() -> None:
  """Verifies the behavior of allocator overflow sgpr."""
  alloc = RegisterAllocator()
  alloc._next_sgpr = MAX_SGPR + 1
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    alloc.get_scalar_register("fail")


def test_allocator_temps() -> None:
  """Verifies the behavior of allocator temps."""
  alloc = RegisterAllocator()
  t1: RdnaVGPR = alloc.allocate_vector_temp()
  t2: RdnaSGPR = alloc.allocate_scalar_temp()
  assert t1.index == 0
  assert t2.index == 0
  t3: RdnaVGPR = alloc.allocate_vector_temp()
  assert t3.index == 1


@pytest.fixture
def mock_semantics() -> MagicMock:
  """Docstring."""
  mgr = MagicMock(spec=SemanticsManager)

  def get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if kind == "Add":
      return ("Add", {})
    return None

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  return mgr


def test_graph_to_rdna_basic_math(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to RDNA basic math."""
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("x", "Input", {}), LogicalNode("y", "Input", {}), LogicalNode("z", "Add", {})]
  g.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) == 3
  assert isinstance(nodes[0], RdnaComment)
  assert "** Input x -> v0" in str(nodes[0]).replace(";", "**")
  inst: typing.Any = nodes[2]
  assert isinstance(inst, RdnaInstruction)
  assert inst.opcode == "v_add_f32"
  assert str(inst.operands[0]) == "v2"
  assert str(inst.operands[1]) == "v0"
  assert str(inst.operands[2]) == "v1"


def test_graph_to_rdna_unmapped(mock_semantics: MagicMock) -> None:
  """Verifies the behavior of graph to RDNA unmapped."""
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("n1", "MysteryOp", {})]
  nodes: list[typing.Any] = synth.from_graph(g)
  assert len(nodes) == 1
  assert "Unmapped Op: MysteryOp" in str(nodes[0])


def test_rdna_to_python_instruction() -> None:
  """Verifies the behavior of RDNA to python instruction."""
  synth = RdnaSynthesizer(MagicMock())
  inst = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "v0 = rdna.v_add_f32(v1, v2)" in code


def test_rdna_to_python_ranges() -> None:
  """Verifies the behavior of RDNA to python ranges."""
  synth = RdnaSynthesizer(MagicMock())
  inst = RdnaInstruction(
    opcode="image_load", operands=[RdnaVGPR(index=0, count=4), RdnaVGPR(index=4, count=4), RdnaSGPR(index=0, count=4)]
  )
  mod: typing.Any = synth.to_python([inst])
  code: str = mod.code
  assert "v_0_3 = rdna.image_load(v_4_7, s_0_3)" in code


def test_rdna_to_python_label() -> None:
  """Verifies the behavior of RDNA to python label."""
  synth = RdnaSynthesizer(MagicMock())
  nodes: list[typing.Any] = [RdnaLabel(name="L_LOOP")]
  mod: typing.Any = synth.to_python(nodes)
  code: str = mod.code
  assert "# RdnaLabel: L_LOOP" in code


# --- Merged from test_rdna_synthesizer_missing.py ---


def test_register_allocator_overflow_vgpr() -> None:
  """Docstring."""
  allocator = RegisterAllocator()
  allocator._next_vgpr = 256
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_vector_register("test")


def test_register_allocator_overflow_sgpr() -> None:
  """Docstring."""
  allocator = RegisterAllocator()
  allocator._next_sgpr = 106
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_scalar_register("test")


def test_convert_operand_to_py_immediate_float() -> None:
  """Docstring."""
  synth = RdnaSynthesizer(None)  # type: ignore
  imm = RdnaImmediate(value=3.14)  # type: ignore
  res: typing.Any = synth._convert_operand_to_py(imm)
  assert getattr(res, "value", None) == "3.14"


def test_convert_operand_to_py_brackets() -> None:
  """Docstring."""
  synth = RdnaSynthesizer(None)  # type: ignore

  class DummyOp(RdnaOperand):
    """Docstring."""

    def __str__(self) -> str:
      """Docstring."""
      return "v[1:2]"

    def to_text(self) -> str:
      """To text."""
      return "v[1:2]"

  res: typing.Any = synth._convert_operand_to_py(DummyOp())
  assert getattr(res, "value", None) == "v_1_2"


def test_convert_operand_to_py_fallback() -> None:
  """Docstring."""
  synth = RdnaSynthesizer(None)  # type: ignore

  class DummyOp(RdnaOperand):
    """Docstring."""

    def __str__(self) -> str:
      """Docstring."""
      return "some-weird-str!"

    def to_text(self) -> str:
      """To text."""
      return "some-weird-str!"

  res: typing.Any = synth._convert_operand_to_py(DummyOp())
  assert res.value == "'some-weird-str!'"


def test_rdna_synthesizer_label_conversion() -> None:
  """Docstring."""
  synth = RdnaSynthesizer(None)  # type: ignore
  label = RdnaLabel(name="my_label")
  mod: typing.Any = synth.to_python([label])
  assert mod is not None


def test_convert_instruction_to_py_no_operands() -> None:
  """Docstring."""
  synth = RdnaSynthesizer(None)  # type: ignore
  inst = RdnaInstruction(opcode="s_endpgm", operands=[])
  res: typing.Any = synth._convert_instruction_to_py(inst)
  assert res is not None


# --- Merged from test_rdna_synthesizer_missing_more.py ---


def test_rdna_synthesizer_macro_exact_match() -> None:
  """Docstring."""
  semantics = SemanticsManager()
  synth = RdnaSynthesizer(semantics)
  # mock a macro
  synth.macro_registry["my_abstract_id"] = lambda alloc, nid, meta: [RdnaComment(text="mock")]

  graph = LogicalGraph("test")
  n = LogicalNode("n1", "my_abstract_id")
  graph.nodes.append(n)

  # We also mock get_definition
  original = semantics.get_definition

  def mock_get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Docstring."""
    if kind == "my_abstract_id":
      return ("my_abstract_id", {})
    return original(kind)

  semantics.get_definition = mock_get_def  # type: ignore

  nodes: list[typing.Any] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert isinstance(nodes[0], RdnaComment)
  assert nodes[0].text == "mock"


def test_rdna_backend_compile() -> None:
  """Docstring."""
  backend = RdnaBackend()
  graph = LogicalGraph("test")
  n = LogicalNode("n1", "Input")
  graph.nodes.append(n)
  code: str = backend.compile(graph)
  assert "; RDNA Code Generation Initialized" in code
