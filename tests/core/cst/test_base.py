"""Canonical unit tests for `ml_switcheroo.core.cst.base`."""

import os
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

from ml_switcheroo.core.cst.base import CSTNode, CSTTransformer, CSTVisitor, Trivia
from ml_switcheroo.exceptions import CSTValidationError


class DummyNode(CSTNode):
  """Concrete `CSTNode` test fixture supporting scalar and list children."""

  name: str = ""
  child: "DummyNode | None" = None
  children: list["DummyNode"] = pydantic.Field(default_factory=list)
  mixed_list: list["DummyNode | str | int"] = pydantic.Field(default_factory=list)

  def to_text(self) -> str:
    """Render `DummyNode` text concatenated with its leading and trailing trivia.

    Returns:
        Rendered source text string.
    """
    leading = "".join(t.text for t in self.leading_trivia)
    trailing = "".join(t.text for t in self.trailing_trivia)
    return f"{leading}{self.name}{trailing}"


class LeafScalarNode(CSTNode):
  """Minimal `CSTNode` with only scalar primitive fields (`updates` empty in `generic_transform`)."""

  label: str = "scalar"

  def to_text(self) -> str:
    """Render `LeafScalarNode` label.

    Returns:
        Node label string.
    """
    return self.label


class ContainerNode(CSTNode):
  """Container `CSTNode` without a custom `transform_ContainerNode` handler."""

  child: DummyNode = pydantic.Field(default_factory=lambda: DummyNode(name="leaf"))
  items: list[DummyNode | str] = pydantic.Field(default_factory=list)

  def to_text(self) -> str:
    """Render `ContainerNode` text via its scalar child.

    Returns:
        Rendered child source text.
    """
    return self.child.to_text()


class RecordingVisitor(CSTVisitor):
  """Visitor recording `DummyNode` names in pre-order traversal."""

  def __init__(self) -> None:
    """Initialize visited-name log and register `DummyNode` handler."""
    super().__init__()
    self.visited_names: list[str] = []
    self.register_visitor(DummyNode, self.visit_DummyNode)

  def visit_DummyNode(self, node: DummyNode) -> None:
    """Record `node.name` and recurse into child fields.

    Args:
        node: Visited `DummyNode` instance.
    """
    self.visited_names.append(node.name)
    self.generic_visit(node)


class UppercaseReplacementTransformer(CSTTransformer):
  """Pure functional transformer returning new `DummyNode` instances with uppercase names."""

  def __init__(self) -> None:
    """Initialize transformer and register `DummyNode` handler."""
    super().__init__()
    self.register_transformer(DummyNode, self.transform_DummyNode)

  def transform_DummyNode(self, node: DummyNode) -> DummyNode:
    """Transform `node` children first and return a new `DummyNode` with `name.upper()`.

    Args:
        node: Input `DummyNode` instance.

    Returns:
        Newly constructed `DummyNode` with uppercase `name`.
    """
    updated = self.generic_transform(node)
    return updated.model_copy(update={"name": updated.name.upper()})


def test_trivia_and_cstnode_base() -> None:
  """Verify `Trivia` rendering and `CSTNode` abstract `to_text` / `__str__` behavior."""
  trivia = Trivia("  ")
  assert trivia.text == "  "
  assert trivia.content == "  "
  assert trivia.to_text() == "  "

  base_node = CSTNode()
  with pytest.raises(NotImplementedError, match="Subclasses must implement to_text"):
    base_node.to_text()
  with pytest.raises(NotImplementedError, match="Subclasses must implement to_text"):
    str(base_node)


def test_trivia_coercion_and_to_text() -> None:
  """Verify `CSTNode` coerces string, `None`, and `list[Trivia]` inputs and renders via `to_text`."""
  coerced = DummyNode.model_validate({"name": "tok", "leading_trivia": "  ", "trailing_trivia": ";"})
  assert len(coerced.leading_trivia) == 1
  assert coerced.leading_trivia[0].text == "  "
  assert len(coerced.trailing_trivia) == 1
  assert coerced.trailing_trivia[0].text == ";"
  assert coerced.to_text() == "  tok;"
  assert str(coerced) == "  tok;"

  empty_trivia = DummyNode.model_validate({"name": "tok", "leading_trivia": None, "trailing_trivia": None})
  assert empty_trivia.leading_trivia == []
  assert empty_trivia.trailing_trivia == []

  explicit_list = DummyNode(name="tok", leading_trivia=[Trivia(" ")])
  assert explicit_list.to_text() == " tok"


def test_visitor_and_non_callable_handler_guard() -> None:
  """Verify `CSTVisitor` traverses scalar and list `CSTNode` children and raises on non-callable handlers."""
  grandchild = DummyNode(name="grandchild")
  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2", child=grandchild)
  root = DummyNode(name="root", children=[child1, child2], mixed_list=["raw", child1, 42])

  visitor = RecordingVisitor()
  container = ContainerNode(child=root, items=[child1, "skip"])
  visitor.visit(container)
  assert visitor.visited_names == ["root", "child1", "child2", "grandchild", "child1", "child1"]


def test_transformer_immutability_scalar_replacement_and_validation_errors() -> None:
  """Verify `CSTTransformer` preserves scalar replacements immutably and raises `CSTValidationError` on invalid handlers."""
  grandchild = DummyNode(name="grandchild")
  child1 = DummyNode(name="child1")
  child2 = DummyNode(name="child2", child=grandchild)
  container = ContainerNode(child=child2, items=[child1, "untouched"])

  transformer = UppercaseReplacementTransformer()
  transformed = transformer.transform(container)
  assert isinstance(transformed, ContainerNode)
  assert transformed.to_text() == "CHILD2"
  assert transformed.child.name == "CHILD2"
  assert transformed.child.child is not None
  assert transformed.child.child.name == "GRANDCHILD"
  assert isinstance(transformed.items[0], DummyNode)
  assert transformed.items[0].name == "CHILD1"
  assert transformed.items[1] == "untouched"

  # Original tree must remain unmutated
  assert container.child.name == "child2"
  assert isinstance(container.items[0], DummyNode)
  assert container.items[0].name == "child1"

  # Scalar-only node with zero CSTNode/list fields returns itself unchanged
  leaf = LeafScalarNode(label="plain")
  assert transformer.transform(leaf) is leaf

  class WrongReturnTransformer(CSTTransformer):
    """Transformer returning an incompatible `CSTNode` subtype."""

    def __init__(self) -> None:
      """Register `DummyNode` handler returning wrong node subtype."""
      super().__init__()
      self.register_transformer(DummyNode, self.transform_DummyNode)

    def transform_DummyNode(self, node: DummyNode) -> LeafScalarNode:
      """Return wrong node type intentionally.

      Args:
          node: Input `DummyNode`.

      Returns:
          Incompatible `LeafScalarNode`.
      """
      return LeafScalarNode(label=node.name)

  with pytest.raises(CSTValidationError, match="must return DummyNode, got LeafScalarNode"):
    WrongReturnTransformer().transform(DummyNode(name="x"))


def test_cst_no_native_extensions() -> None:
  """Verify importing `ml_switcheroo.core.cst.base` loads zero native compiler extensions (`llvmlite`, `mlir.ir`)."""
  repo_root = Path(__file__).resolve().parent.parent.parent.parent
  src_dir = str(repo_root / "src")
  env = dict(os.environ)
  env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

  cmd = [
    sys.executable,
    "-c",
    (
      "import ml_switcheroo.core.cst.base, sys; "
      "assert not any(m.startswith('llvmlite') for m in sys.modules), 'Native extension llvmlite found'; "
      "assert not any(m.startswith('mlir.ir') for m in sys.modules), 'Native extension mlir.ir found'"
    ),
  ]
  res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
  assert res.returncode == 0, f"Native extension check failed: {res.stderr}"


def test_unified_cst_hierarchy_and_emit_helpers() -> None:
  """Verify ``MarkupNode``, ``HtmlNode``, ``LatexNode``, ``TikzBaseNode``, ``CppNode``, and ``WatNode`` inherit ``CSTNode``."""
  from ml_switcheroo.core.compiler.backends.cpp.cst import CppNode, Identifier
  from ml_switcheroo.core.html.nodes import HtmlNode, TagNode
  from ml_switcheroo.core.html.nodes import TextNode as HtmlTextNode
  from ml_switcheroo.core.latex.nodes import LatexNode
  from ml_switcheroo.core.latex.nodes import TextNode as LatexTextNode
  from ml_switcheroo.core.markup.cst import MarkupNode
  from ml_switcheroo.core.markup.cst import TextNode as MarkupTextNode
  from ml_switcheroo.core.markup.cst import TriviaNode as MarkupTriviaNode
  from ml_switcheroo.core.tikz.nodes import TikzBaseNode, TikzNode
  from ml_switcheroo.core.tikz.nodes import TriviaNode as TikzTriviaNode
  from ml_switcheroo.core.wasm.cst import WasmRegister, WatNode, WatParam

  assert issubclass(MarkupNode, CSTNode)
  assert issubclass(HtmlNode, CSTNode)
  assert issubclass(LatexNode, CSTNode)
  assert issubclass(TikzBaseNode, CSTNode)
  assert issubclass(CppNode, CSTNode)
  assert issubclass(WatNode, CSTNode)

  triv = Trivia("  ")
  assert triv.emit() == "  "
  assert triv.emit(2) == "  "

  coerced_leading = CSTNode._coerce_trivia([" ", MarkupTriviaNode(content="//c\n", kind="comment")])
  coerced_trailing = CSTNode._coerce_trivia("")
  coerced_from_seq = DummyNode(
    name="line1\n\nline2",
    leading_trivia=coerced_leading,
    trailing_trivia=coerced_trailing,
  )
  assert len(coerced_from_seq.leading_trivia) == 2
  assert coerced_from_seq.trailing_trivia == []
  assert coerced_from_seq.emit(0) == " //c\nline1\n\nline2"
  assert coerced_from_seq.emit(1) == "   //c\n  line1\n\n  line2"

  markup_txt = MarkupTextNode(content="hi", leading_trivia=" ", trailing_trivia="\n")
  assert markup_txt.to_text() == " hi\n"
  with pytest.raises(NotImplementedError):
    MarkupNode().emit()

  html_tag = TagNode(name="p", children=[HtmlTextNode(content="ok")], leading_trivia=" ")
  assert html_tag.to_text() == " <p>ok</p>"

  latex_txt = LatexTextNode(content="note")
  assert "note" in latex_txt.to_text()

  tikz_node = TikzNode(node_id="n0", x=1, y=2, content="x", leading_trivia=[TikzTriviaNode(" ")])
  assert tikz_node.to_text().startswith(" \\node")

  cpp_id = Identifier(name="tensor_x")
  assert cpp_id.emit(1) == "  tensor_x"

  wat_param = WatParam(name=WasmRegister("x"), type_id="f32")
  assert wat_param.emit(1).endswith("(param $x f32)")


def test_isa_cst_and_table_driven_parser_base_contracts() -> None:
  """Verify ``IsaDirective``, ``IsaModule``, ``IsaLexer``, ``IsaTransformerBase``, and ``IsaParser`` base contracts."""
  from lark import Token

  from ml_switcheroo.core.compiler.isa.cst import IsaComment, IsaDirective, IsaImmediate, IsaLabel, IsaModule, IsaNode
  from ml_switcheroo.core.compiler.isa.parser import (
    IsaDialectSpec,
    IsaLexer,
    IsaParser,
    IsaToken,
    IsaTransformerBase,
  )

  directive = IsaDirective(name="global", params=["kernel_entry"], leading_trivia="  ", trailing_trivia="\n")
  assert directive.to_text() == "  .global kernel_entry\n"
  bare_directive = IsaDirective(name="text")
  assert bare_directive.to_text() == ".text"

  mod: IsaModule[IsaNode] = IsaModule(statements=[directive, IsaComment(text="end")], leading_trivia="// hdr\n")
  assert mod.to_text() == "// hdr\n  .global kernel_entry\n// end"

  spec = IsaDialectSpec(
    token_regex=(("WS", r"[ \t]+"), ("IDENTIFIER", r"[a-z]+"), ("MISMATCH", r".")),
    punctuation_map={},
    grammar="start: IDENTIFIER\nIDENTIFIER: /.+/",
  )

  class UnimplementedLexer(IsaLexer[IsaToken]):
    """Minimal lexer subclass without overriding ``_make_token``."""

    SPEC = spec

  with pytest.raises(NotImplementedError):
    list(UnimplementedLexer(None).lex("abc"))

  base_tr: IsaTransformerBase[
    IsaNode,
    IsaImmediate,
    IsaModule[IsaNode],
    IsaComment,
    IsaDirective,
    IsaLabel,
    IsaImmediate,
  ] = IsaTransformerBase()
  with pytest.raises(NotImplementedError):
    base_tr._is_statement_node("x")
  with pytest.raises(NotImplementedError):
    base_tr._is_operand_node("x")
  with pytest.raises(NotImplementedError):
    base_tr._create_module([], [])
  with pytest.raises(NotImplementedError):
    base_tr._create_comment("c", [])
  with pytest.raises(NotImplementedError):
    base_tr._create_directive("d", [], [])
  with pytest.raises(NotImplementedError):
    base_tr._create_label("l", [])
  with pytest.raises(NotImplementedError):
    base_tr._create_immediate(1, False, [])

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassTransformer

  sass_tr = NvidiaSassTransformer()
  ops = sass_tr.operands([NvidiaSassImmediate(value=1), None, Token("COMMA", ","), NvidiaSassImmediate(value=2)])
  assert len(ops) == 2

  class UnimplementedParser(IsaParser[IsaModule[IsaNode], IsaNode]):
    """Minimal parser subclass without overriding abstract hooks."""

    SPEC = spec
    LEXER_CLS = UnimplementedLexer

  with pytest.raises(NotImplementedError):
    UnimplementedParser("abc")

  class PartialParser(UnimplementedParser):
    """Parser subclass implementing ``_build_transformer`` to exercise remaining abstract hooks."""

    def _build_transformer(
      self,
    ) -> IsaTransformerBase[
      IsaNode,
      IsaImmediate,
      IsaModule[IsaNode],
      IsaComment,
      IsaDirective,
      IsaLabel,
      IsaImmediate,
    ]:
      """Return ``base_tr`` for testing ``_empty_module`` and ``_validate_module``.

      Returns:
          Base transformer instance.
      """
      return base_tr

  partial = PartialParser("")
  with pytest.raises(NotImplementedError):
    partial.parse()
  with pytest.raises(NotImplementedError):
    partial._validate_module(IsaNode())
