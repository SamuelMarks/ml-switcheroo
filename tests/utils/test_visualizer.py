"""Test suite for the Visualizer module."""

import libcst as cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.utils.visualizer import MermaidGenerator


def test_visualizer_basic_flow() -> None:
  """Verify `MermaidGenerator` emits `graph TD` for visualizer basic flow."""
  code: str = "x = 1"
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "graph TD" in mermaid
  assert "classDef" in mermaid
  assert "::modNode" in mermaid
  assert "Assign (=)" in mermaid
  assert "::stmtNode" in mermaid
  assert "-->" in mermaid


def test_visualizer_function_def() -> None:
  """Verify `MermaidGenerator` emits `Def: f` for visualizer function def."""
  code: str = "def f(a, b=2): pass"
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "Def: f" in mermaid
  assert "::funcNode" in mermaid
  assert mermaid.count(":::funcNode") == 1


def test_visualizer_call_structure() -> None:
  """Verify `MermaidGenerator` emits `Call` for visualizer call structure."""
  code: str = "fn(x, y=z)"
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "Call" in mermaid
  assert "fn()" in mermaid
  assert "::callNode" in mermaid
  assert "arg=" in mermaid or "arg" in mermaid


def test_visualizer_truncated_labels() -> None:
  """Verify `MermaidGenerator` emits `...` for visualizer truncated labels."""
  long_str: str = "A" * 100
  code: str = f"x = '{long_str}'"
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "..." in mermaid


def test_visualizer_escapes_quotes() -> None:
  """Verify `MermaidGenerator` emits `quote` for visualizer escapes quotes."""
  code: str = 'x = "quote"'
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "quote" in mermaid


def test_node_to_str_robustness() -> None:
  """Verify `MermaidGenerator` handles node to str robustness as expected."""
  gen: MermaidGenerator = MermaidGenerator()
  assert gen._node_to_str(cst.Name("x")) == "x"
  attr: cst.Attribute = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  assert gen._node_to_str(attr) == "a.b"
  assert gen._node_to_str(cst.Integer("1")) == "1"
  assert gen._node_to_str(cst.Float("1.5")) == "1.5"
  tup: cst.Tuple = cst.Tuple(elements=[])
  res: str = gen._node_to_str(tup)
  assert res == "()"


def test_visualizer_exceptions() -> None:
  """Verify `MermaidGenerator` emits `<BadNode>` for visualizer exceptions."""
  import libcst as cst

  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen: MermaidGenerator = MermaidGenerator()

  class BadNode(cst.CSTNode):
    """Test double `BadNode` used in visualizer tests."""

    def _codegen_impl(self, state: CodegenState) -> None:
      """Execute `RuntimeError` to support codegen impl in visualizer tests.

      Args:
          state: Input `state` parameter value (`CodegenState`).
      """
      raise RuntimeError("fail")

    def _visit_and_replace_children(self, v: cst.CSTVisitorT) -> "BadNode":
      """Execute `_visit_and_replace_children` on `BadNode` for visualizer tests.

      Args:
          v: Input `v` parameter value (`cst.CSTVisitorT`).

      Returns:
          Resulting `'BadNode'` from visit and replace children.
      """
      return self

  from unittest import mock

  assert "<BadNode>" in gen._node_to_str(BadNode())
  with mock.patch.object(gen, "_node_to_str", side_effect=RuntimeError("fail")):
    call_node: cst.Call = cst.Call(func=cst.Name("foo"))
    gen.visit_Call(call_node)
  with mock.patch.object(gen, "_node_to_str", side_effect=RuntimeError("fail")):
    arg_node: cst.Arg = cst.Arg(value=cst.Name("foo"))
    gen.visit_Arg(arg_node)
  gen.stack.clear()
  gen.leave_Assign(cst.Assign(targets=[cst.AssignTarget(cst.Name("a"))], value=cst.Name("b")))
  gen.stack.clear()
  gen.visit_SimpleString(cst.SimpleString('""'))


def test_visualizer_more_nodes() -> None:
  """Verify `MermaidGenerator` handles visualizer more nodes as expected."""
  import libcst as cst

  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen: MermaidGenerator = MermaidGenerator()
  cls_node: cst.ClassDef = cst.ClassDef(name=cst.Name("Foo"), body=cst.IndentedBlock([]))
  gen.visit_ClassDef(cls_node)
  gen.leave_ClassDef(cls_node)
  imp_node: cst.Import = cst.Import(names=[cst.ImportAlias(name=cst.Name("foo")), cst.ImportAlias(name=cst.Name("bar"))])
  gen.visit_Import(imp_node)
  imp_from_node: cst.ImportFrom = cst.ImportFrom(
    module=cst.Name("foo"),
    names=[
      cst.ImportAlias(name=cst.Name("a")),
      cst.ImportAlias(name=cst.Name("b")),
      cst.ImportAlias(name=cst.Name("c")),
      cst.ImportAlias(name=cst.Name("d")),
    ],
  )
  gen.visit_ImportFrom(imp_from_node)
  imp_from_star: cst.ImportFrom = cst.ImportFrom(module=cst.Name("foo"), names=cst.ImportStar())
  res = gen.visit_ImportFrom(imp_from_star)
  assert res is False
  assert any("Foo" in line for line in gen.nodes)


def test_visualizer_more_fallbacks() -> None:
  """Verify `MermaidGenerator` handles visualizer more fallbacks as expected."""
  import libcst as cst

  from ml_switcheroo.utils.visualizer import MermaidGenerator

  gen: MermaidGenerator = MermaidGenerator()
  call: cst.Call = cst.Call(func=cst.Call(func=cst.Name("a")))
  gen.visit_Call(call)

  class DummyImportAlias(cst.ImportAlias):
    """Minimal `cst.ImportAlias` test double used in visualizer tests."""

  imp: cst.Import = cst.Import(names=[DummyImportAlias(name=cst.Attribute(cst.Name("a"), cst.Name("b")))])
  res = gen.visit_Import(imp)
  assert res is False
  assert len(gen.nodes) > 0


def test_visualizer_complex_arg() -> None:
  """Verify `MermaidGenerator` emits `sub_call()` for visualizer complex arg."""
  code: str = "fn(k=sub_call(1))"
  tree: cst.Module = cst.parse_module(code)
  gen: MermaidGenerator = MermaidGenerator()
  mermaid: str = gen.generate(tree)
  assert "sub_call()" in mermaid


def test_visualizer_empty_stack_and_import_from_edge_cases() -> None:
  """Verifies leave methods with empty stack and ImportFrom edge cases."""
  from unittest.mock import MagicMock

  gen: MermaidGenerator = MermaidGenerator()
  gen.stack.clear()
  gen.leave_Module(cst.Module([]))
  gen.leave_ClassDef(cst.ClassDef(name=cst.Name("Foo"), body=cst.IndentedBlock([])))
  gen.leave_FunctionDef(cst.FunctionDef(name=cst.Name("foo"), params=cst.Parameters(), body=cst.IndentedBlock([])))
  gen.leave_Call(cst.Call(func=cst.Name("foo")))
  gen.leave_Arg(cst.Arg(value=cst.Name("x")))

  mock_imp_from: MagicMock = MagicMock(spec=cst.ImportFrom)
  mock_imp_from.module = None
  mock_imp_from.names = [
    cst.ImportAlias(name=cst.Name("valid")),
    cst.MaybeSentinel.DEFAULT,
  ]
  res = gen.visit_ImportFrom(mock_imp_from)
  assert res is False
  assert gen.stack == []
