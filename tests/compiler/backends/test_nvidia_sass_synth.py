"""Test suite for nvidia sass synth."""

import pathlib
import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode  # noqa: F811
from ml_switcheroo.semantics.manager import SemanticsManager


def test_nvidia_sass_synth_invalid_opcode() -> None:
  """Verifies the behavior of NVIDIA_SASS synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code!"}
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="BadOp")]})
  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    synth.from_graph(g)


def test_nvidia_sass_synth_liveness_tracking() -> None:
  """Verifies the behavior of NVIDIA_SASS synth liveness tracking."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "FADD"}
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(
    nodes={
      "n1": LogicalNode(id="n1", op_type="Input"),
      "n2": LogicalNode(id="n2", op_type="Add", inputs=["n1"]),
    }
  )
  assert synth.allocator._liveness_map == {}
  synth.from_graph(g)
  assert synth.allocator._liveness_map["n1"] == 0
  assert len(synth.allocator._free_pool) == 254


def test_nvidia_sass_macro_mean() -> None:
  """Verifies the behavior of NVIDIA_SASS mean macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_mean

  alloc = RegisterAllocator(255, lambda x: None)
  nodes = expand_mean(alloc, "n1", {"elements": 10})  # type: ignore[arg-type]
  assert any(n.opcode == "FADD" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "FMUL" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_relu() -> None:
  """Verifies the behavior of NVIDIA_SASS relu macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_relu

  alloc = RegisterAllocator(255, lambda x: None)
  nodes = expand_relu(alloc, "n1", {})  # type: ignore[arg-type]
  assert any(n.opcode == "FMAX" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_flatten() -> None:
  """Verifies the behavior of NVIDIA_SASS flatten macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_flatten

  alloc = RegisterAllocator(255, lambda x: None)
  nodes = expand_flatten(alloc, "n1", {})  # type: ignore[arg-type]
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_reshape() -> None:
  """Verifies the behavior of NVIDIA_SASS reshape macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_reshape

  alloc = RegisterAllocator(255, lambda x: None)
  nodes = expand_reshape(alloc, "n1", {})  # type: ignore[arg-type]
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_conv3d() -> None:
  """Verifies the behavior of NVIDIA_SASS conv3d macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_conv3d

  alloc = RegisterAllocator(255, lambda x: None)
  nodes = expand_conv3d(alloc, "n1", {"k": 3})  # type: ignore[arg-type]
  assert any(n.opcode == "FFMA" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "IMAD" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macros_coverage() -> None:
  """Verify nvidia sass macros coverage behavior."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
    expand_adam,
    expand_conv_general_dilated,
    expand_l,
    expand_transpose,
    expand_variable,
  )

  allocator = None
  assert len(expand_variable(allocator, "n", {})) == 2  # type: ignore[arg-type]
  assert len(expand_transpose(allocator, "n", {})) == 2  # type: ignore[arg-type]
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2  # type: ignore[arg-type]
  assert len(expand_adam(allocator, "n", {})) == 2  # type: ignore[arg-type]
  assert len(expand_l(allocator, "n", {})) == 2  # type: ignore[arg-type]


def test_nvidia_sass_synth_suffix_macro() -> None:
  """Verify nvidia sass synth suffix macro behavior."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("nvidia_sass.l", {})
  synth = NvidiaSassSynthesizer(mock_semantics)
  n1 = LogicalNode(id="n1", op_type="dummy")
  n2 = LogicalNode(id="n2", op_type="nvidia_sass.l")
  edge = LogicalEdge(source="n1", target="n2")
  g = LogicalGraph(nodes={n.id: n for n in [n1, n2]}, edges=[edge])
  res = synth.from_graph(g)
  assert len(res) >= 1


def test_nvidia_sass_synth_liveness_existing() -> None:
  """Verify nvidia sass synth liveness existing behavior."""
  # Hit 130->132
  alloc = RegisterAllocator(255, lambda x: None)
  nodes = {
    "s": LogicalNode("s", op_type="Input"),
    "t1": LogicalNode("t1", op_type="Op"),
    "t2": LogicalNode("t2", op_type="Op"),
  }
  edges = [
    LogicalEdge(source="s", target="t1"),
    LogicalEdge(source="s", target="t2"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  alloc.build_liveness(g)
  assert alloc._liveness_map["s"] == 2


def test_nvidia_sass_synth_record_usage_missing() -> None:
  """Verify nvidia sass synth record usage missing behavior."""
  # Hit 141->exit (var_name not in _liveness_map)
  alloc = RegisterAllocator(255, lambda x: None)
  alloc.record_usage("missing")  # should not throw
  res = None
  assert res is None
  assert "missing" not in alloc._liveness_map


def test_nvidia_sass_synth_record_usage_not_zero() -> None:
  """Verify nvidia sass synth record usage not zero behavior."""
  # Hit 143->exit
  alloc = RegisterAllocator(255, lambda x: None)
  alloc._liveness_map["x"] = 2
  alloc.record_usage("x")
  assert alloc._liveness_map["x"] == 1


def test_nvidia_sass_synth_macros_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify nvidia sass synth macros missing behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  # Hit 173->exit
  import os

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = NvidiaSassSynthesizer(None)
  assert synth.macro_registry == {}


def test_nvidia_sass_synth_macros_attr_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
  """Verify nvidia sass synth macros attr missing behavior.

  Args:
      monkeypatch: Test argument or fixture.
      tmp_path: Test argument or fixture.
  """
  # Hit 178->177
  import json
  import os

  f = tmp_path / "macros.json"
  f.write_text(json.dumps({"test_op": "missing_macro_func"}))

  def mock_dirname(p: str) -> str:
    """Mock dirname.

    Args:
        p: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return str(tmp_path)

  def mock_exists(p: str) -> bool:
    """Mock exists.

    Args:
        p: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return p == str(f)

  monkeypatch.setattr(os.path, "dirname", mock_dirname)
  monkeypatch.setattr(os.path, "exists", mock_exists)

  synth = NvidiaSassSynthesizer(None)
  assert synth.macro_registry == {}


def test_nvidia_sass_synth_existing_input_map() -> None:
  """Verify nvidia sass synth existing input map behavior."""
  # Hit 218->220
  synth = NvidiaSassSynthesizer(None)
  g = LogicalGraph(
    "Test",
    nodes={
      "s": LogicalNode(id="s", op_type="Input"),
      "s2": LogicalNode(id="s2", op_type="Input"),
      "t": LogicalNode(id="t", op_type="Output", inputs=["s", "s2"]),
    },
  )
  res = synth.from_graph(g)
  assert isinstance(res, list)
  assert len(res) >= 1


def test_nvidia_sass_synth_output_no_sources() -> None:
  """Verify nvidia sass synth output no sources behavior."""
  # Hit 234->238
  synth = NvidiaSassSynthesizer(None)
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="out", op_type="Output"))
  res = synth.from_graph(g)
  assert res == []


def test_nvidia_sass_synth_abstract_id_none() -> None:
  """Verify nvidia sass synth abstract id none behavior."""
  # Hit 278->281

  class FakeSemantics:
    """Fake semantics."""

    def get_definition(self, k: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Get definition.

      Args:
          k: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return ("", {})

  synth = NvidiaSassSynthesizer(FakeSemantics())
  g = LogicalGraph("Test")
  g.add_node(LogicalNode(id="n", op_type="not_found"))
  res = synth.from_graph(g)
  assert "Unmapped Op:" in res[0].text  # type: ignore[attr-defined]


def test_nvidia_sass_synth_to_cst_other_node() -> None:
  """Verify nvidia sass synth to cst other node behavior."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment

  synth = NvidiaSassSynthesizer(None)
  mod = synth.to_python([NvidiaSassComment("test comment")])
  assert len(mod.body) == 0


def test_nvidia_sass_synth_to_cst_label() -> None:
  """Verify nvidia sass synth to cst label behavior."""
  synth = NvidiaSassSynthesizer(None)
  mod = synth.to_python([NvidiaSassLabel("lbl")])
  assert len(mod.body) == 1


def test_nvidia_sass_synth_to_cst_comment_no_begin_end() -> None:
  """Verify nvidia sass synth to cst comment no begin end behavior."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment

  synth = NvidiaSassSynthesizer(None)
  mod = synth.to_python([NvidiaSassComment("just a regular comment")])
  assert len(mod.body) == 0


def test_nvidia_sass_synth_instruction_dest_not_register() -> None:
  """Verify nvidia sass synth instruction dest not register behavior."""
  # Hit 408->412
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate

  synth = NvidiaSassSynthesizer(None)
  # create instruction with immediate as dest
  inst = NvidiaSassInstruction(
    opcode="OP",
    operands=[NvidiaSassImmediate("target_var"), NvidiaSassImmediate("2")],  # type: ignore[arg-type]
  )
  stmt = synth._convert_instruction_to_py(inst)
  import libcst as cst

  code = cst.Module(body=[stmt]).code
  assert "nvidia_sass.OP" in code


def test_nvidia_sass_synth_to_cst_other_non_label_non_comment() -> None:
  """Verify nvidia sass synth to cst other non label non comment behavior."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode

  class CustomNode(NvidiaSassNode):
    """Custom node."""

    pass

  synth = NvidiaSassSynthesizer(None)
  mod = synth.to_python([CustomNode()])
  assert len(mod.body) == 0


def test_nvidia_sass_allocator_overflow() -> None:
  """Verify nvidia sass allocator overflow behavior."""
  allocator = RegisterAllocator(255, lambda x: None)
  # It has 255 registers. Allocate them all
  with pytest.raises(ValueError):
    for i in range(256):
      allocator.allocate_temp()


def test_nvidia_sass_synth_float_immediate() -> None:
  """Verify nvidia sass synth float immediate behavior."""
  import libcst as cst

  synth = NvidiaSassSynthesizer(None)
  res = synth._convert_operand_to_py(NvidiaSassImmediate(value=3.14))
  assert isinstance(res, cst.Float)
  assert res.value == "3.14"


def test_nvidia_sass_synth_expr_statement() -> None:
  """Verify nvidia sass synth expr statement behavior."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate

  synth = NvidiaSassSynthesizer(None)
  # Give it an instruction with operands, but where dest is not a Register
  # So it hits the else branch "Expression Statement"
  inst = NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassImmediate(value=0), NvidiaSassImmediate(value=1)])
  stmt = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_no_operands() -> None:
  """Verify nvidia sass synth no operands behavior."""
  import libcst as cst

  synth = NvidiaSassSynthesizer(None)
  inst = NvidiaSassInstruction(opcode="NOP", operands=[])
  stmt = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_nop() -> None:
  """Verify nvidia sass synth nop behavior."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate

  synth = NvidiaSassSynthesizer(None)
  inst = NvidiaSassInstruction(opcode="NOP", operands=[NvidiaSassImmediate(value=0)])
  stmt = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_hex_immediate() -> None:
  """Verify nvidia sass synth hex immediate behavior."""
  import libcst as cst

  synth = NvidiaSassSynthesizer(None)
  # Hex immediate
  res = synth._convert_operand_to_py(NvidiaSassImmediate(value=10, is_hex=True))
  assert isinstance(res, cst.Integer)
  assert res.value == hex(10)


def test_nvidia_sass_synth_output_with_sources() -> None:
  """Verify nvidia sass synth output with sources behavior."""
  synth = NvidiaSassSynthesizer(None)
  src_node = LogicalNode(id="src_node", op_type="Input")
  out_node = LogicalNode(id="out", op_type="Output", inputs=["src_node"])
  g = LogicalGraph("Test", nodes={"src_node": src_node, "out": out_node})
  res = synth.from_graph(g)
  assert len(res) > 0


# --- Merged from test_nvidia_sass_synth_missing.py ---


def test_nvidia_sass_synthesizer_macro_exact_match() -> None:
  """Verify nvidia sass synthesizer macro exact match behavior."""
  semantics = SemanticsManager()
  synth = NvidiaSassSynthesizer(semantics)

  synth.macro_registry["my_macro"] = lambda alloc, nid, meta: [NvidiaSassComment(text="mock_my_macro")]

  n1 = LogicalNode(id="n1", op_type="tensor")
  n2 = LogicalNode(id="n2", op_type="my_macro", inputs=["n1"])
  graph = LogicalGraph("test", nodes={"n1": n1, "n2": n2})

  original = semantics.get_definition

  def mock_get_def(kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Execute mock get def helper logic.

    Args:
        kind: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if kind == "my_macro":
      return ("my_macro", {})
    return original(kind)

  setattr(semantics, "get_definition", mock_get_def)

  nodes = synth.from_graph(graph)

  found = False
  for n in nodes:
    if isinstance(n, NvidiaSassComment) and n.text == "mock_my_macro":
      found = True
  assert found


def test_nvidia_sass_backend_semantics_provided() -> None:
  """Verify nvidia sass backend semantics provided behavior."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
  from ml_switcheroo.semantics.manager import SemanticsManager

  sem = SemanticsManager()
  backend = NvidiaSassBackend(semantics=sem)
  assert backend.synthesizer.semantics is sem


def test_nvidia_sass_misc_and_isa_cst_coverage() -> None:
  """Verify coverage of nvidia_sass/misc.py helpers and core/compiler/isa/cst.py methods."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.misc import (
    _make_generic_expand,
    expand_adam,
    expand_l,
    expand_mseloss,
    expand_transpose,
    expand_variable,
  )
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister
  from ml_switcheroo.core.compiler.isa.cst import IsaImmediate, IsaInstruction, IsaMemory
  from ml_switcheroo.core.cst.base import Trivia
  from ml_switcheroo.core.markup.cst import MarkupNode, TextNode, TriviaNode
  from ml_switcheroo.core.markup.parser import BaseTokenizer, RecursiveDescentParser, Token
  from ml_switcheroo.utils.formatting import MarkupVisitor

  alloc = RegisterAllocator(255, lambda x: NvidiaSassRegister(name=f"R{x}"))
  mse_none = expand_mseloss(alloc, "loss_node", {"reduction": "none", "elements": 8})
  assert any("END MSELoss" in str(n) for n in mse_none)

  generic_fn = _make_generic_expand("CustomOp")
  generic_nodes = generic_fn(alloc, "custom_1", {})
  assert len(generic_nodes) == 3
  assert "BEGIN CustomOp" in str(generic_nodes[0])

  assert len(expand_variable(alloc, "var_1", {})) == 2
  assert len(expand_transpose(alloc, "tr_1", {})) == 2
  assert len(expand_adam(alloc, "adam_1", {})) == 2
  assert len(expand_l(alloc, "l_1", {})) == 2

  # IsaMemory and IsaInstruction to_text()
  reg = NvidiaSassRegister(name="R0", leading_trivia=[Trivia(" ")], trailing_trivia=[Trivia(" ")])
  mem_int = IsaMemory(base_reg=reg, offset=4)
  assert "[R0 + 4]" in mem_int.to_text()
  mem_reg = IsaMemory(base_reg=None, offset=reg)
  assert "[ + R0]" in mem_reg.to_text()
  mem_plain = IsaMemory(base_reg=None, offset=None)
  assert mem_plain.to_text() == "[]"

  inst_empty = IsaInstruction(opcode="NOP", operands=[])
  assert inst_empty.to_text() == "NOP"
  inst_ops = IsaInstruction(opcode="MOV", operands=[IsaImmediate(value=1), IsaImmediate(value=16, is_hex=True)])
  assert inst_ops.to_text() == "MOV 1, 0x10"

  # MarkupNode, TriviaNode, TextNode
  base_markup = MarkupNode()
  with pytest.raises(NotImplementedError):
    base_markup.to_text()

  tn = TriviaNode(content="  \n")
  assert tn.emit() == "  \n"

  txt = TextNode(
    content="hello",
    bold=True,
    italic=True,
    leading_trivia=[Trivia("<")],
    trailing_trivia=[Trivia(">")],
  )
  assert txt.emit() == "<***hello***>"
  txt_plain = TextNode(content="plain")
  assert txt_plain.emit() == "plain"

  # Markup parser Token, BaseTokenizer, RecursiveDescentParser
  tok = Token("IDENT", "foo", 0)
  assert repr(tok) == "Token(IDENT, 'foo')"
  tokenizer = BaseTokenizer([("SKIP", r"\s+"), ("IDENT", r"[a-z]+"), ("NUM", r"\d+")])
  tokens = tokenizer.tokenize("  abc 123 ")
  assert len(tokens) == 2
  rdp = RecursiveDescentParser(tokens)
  assert rdp.match("NUM") is False
  assert rdp.match("IDENT") is True
  num_tok = rdp.expect("NUM")
  assert num_tok.value == "123"
  assert rdp.current is None
  assert rdp.advance() is None
  with pytest.raises(ValueError, match="Expected IDENT"):
    rdp.expect("IDENT")

  # MarkupVisitor
  visited = []

  class ChildNode:
    """Test helper class ChildNode."""

    pass

  class ParentNode:
    """Test helper class ParentNode."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.children = [ChildNode()]

  class CustomVisitor(MarkupVisitor):
    """Test helper class CustomVisitor."""

    def visit_ChildNode(self, node: ChildNode) -> None:
      """Record visited ChildNode.

      Args:
          node: Test argument or fixture.
      """
      visited.append("child")

  v = CustomVisitor()
  v.visit(ParentNode())
  v.visit(ChildNode())
  assert visited == ["child", "child"]

  import ml_switcheroo.core.exceptions as core_exceptions
  import ml_switcheroo.exceptions as root_exceptions

  assert issubclass(core_exceptions.TranspilationError, root_exceptions.MLSwitcherooError)
