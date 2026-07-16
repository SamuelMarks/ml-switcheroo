"""Auto-generated doc."""


def test_rdna_synthesizer_gaps():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  # RegisterAllocator fallbacks
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

  # 84: release scalar
  pass

  # RdnaSynthesizer compilation gaps
  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())
  g = LogicalGraph(nodes=[], edges=[])
  # 172-193: Unmapped node
  g.nodes.append(LogicalNode("n1", "torch.add", {"arg_1": "a", "arg_2": "b"}))
  # 201-216: rdna_nodes to cst conversion with label
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  # Also compile to python explicitly to hit 201-216 block
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend

  SassBackend()
  assert "v0 =" in mod.code


def test_rdna_synthesizer_py_translation():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import Instruction, Label, Immediate, VGPR, SGPR, Memory

  from ml_switcheroo.semantics.manager import SemanticsManager

  synth = RdnaSynthesizer(SemanticsManager())

  # 201-216, 225-257, 266-281
  nodes = [
    Instruction("v_add_f32", []),  # 226
    Instruction("store_dword", [VGPR(0, 1), Immediate(5, True)]),  # 236 is_store
    Instruction("branch", [Label("L1")]),  # 236 is_branch
    Instruction("v_mov_b32", [VGPR(1, 1), Immediate(3.14, False)]),  # float immediate
    Instruction("v_mov_b32", [VGPR(2, 1), Immediate(42, False)]),  # int immediate
    Instruction("s_load", [SGPR(0, 2), Memory(VGPR(3, 1))]),  # bracket string
    Label("L1"),
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
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  synth = RdnaSynthesizer(SemanticsManager())

  # 140-142 edges
  # 147-150 Input
  # 154-158 Output

  g = LogicalGraph(
    nodes=[LogicalNode("in", "Input", {"name": "x"}), LogicalNode("out", "Output")], edges=[LogicalEdge("in", "out")]
  )
  nodes = synth.from_graph(g)
  assert len(nodes) > 0

  # 275-276 convert operand string with bracket
  from ml_switcheroo.core.compiler.frontends.rdna.nodes import LabelRef

  # "LabelRef" __str__ might return `str(self.name)`? Let's use something that returns a string with `[`
  res = synth._convert_operand_to_py(LabelRef("[var]"))
  assert res.value == "_var"


def test_rdna_synthesizer_misc():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RegisterAllocator, RdnaSynthesizer
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  alloc = RegisterAllocator()
  # populate var_to_sgpr
  alloc._var_to_sgpr["test"] = 0
  s = alloc.get_scalar_register("test")
  assert s.index == 0

  synth = RdnaSynthesizer(SemanticsManager())

  # 189-190: op with a mapped variant and sources.
  # we need a node that is mapped to rdna so variant gets found. Let's mock the semantics manager.
  class MockSemantics:
    """Auto-generated doc."""

    def resolve_variant(self, node_id, tgt):
      """Auto-generated doc."""
      return {"api": "v_add_f32"}

    def get_definition(self, kind):
      """Auto-generated doc."""
      return None

  synth = RdnaSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[LogicalNode("src", "src_op"), LogicalNode("dst", "dst_op")], edges=[LogicalEdge("src", "dst")])
  synth.from_graph(g)


def test_sass_macros_linear():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_linear
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_linear(alloc, "test_lin", {"in_features": 64, "bias": True})
  assert len(nodes) > 10


def test_sass_synthesizer_gaps():
  """Auto-generated doc."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator, SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.nodes import Instruction, Label, Comment, Immediate, Register, Memory
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode

  alloc = RegisterAllocator()
  for _ in range(255):
    alloc.allocate_temp()

  import pytest

  with pytest.raises(ValueError):
    alloc.allocate_temp()

  class MockSemantics:
    """Auto-generated doc."""

    def get_definition(self, kind):
      """Auto-generated doc."""
      return None

    def resolve_variant(self, abstract_id, tgt):
      """Auto-generated doc."""
      if abstract_id == "Missing":
        return {}
      return {"api": "FADD"}

  synth = SassSynthesizer(MockSemantics())
  g = LogicalGraph(nodes=[], edges=[])
  g.nodes.append(LogicalNode("n1", "Missing"))
  nodes = synth.from_graph(g)
  assert len(nodes) > 0
  assert "Unmapped Op:" in str(nodes[0])

  # 267-289: Python emission
  nodes = [
    Instruction("FADD", []),
    Instruction("FADD", [Register("R1")], predicate="P0"),
    Comment("test"),
    Label("L1"),
    Instruction("STG", [Memory(Register("R1")), Immediate(1)]),
    Instruction("BRA", [Label("L1")]),
    Instruction("MOV", [Register("R2"), Immediate(1, True)]),  # 380 Hex
    Instruction("FMUL", [Register("R3"), Immediate(3.14, False)]),  # 382 float
    Instruction("MOV", [Register("R4"), Register("R1")]),  # alphanumeric
  ]
  mod = synth.to_python(nodes)
  from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend

  SassBackend()
  code = mod.code
  assert "sass.FADD" in code
  assert "Label: L1" in code
  assert "0x1" in code
  assert "3.14" in code
