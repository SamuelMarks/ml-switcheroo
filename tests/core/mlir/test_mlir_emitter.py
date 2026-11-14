"""Unit tests for `PythonToMlirEmitter` and `SSAContext` (`ml_switcheroo.core.mlir.emitter`)."""

import collections
from unittest import mock

import libcst as cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.core.mlir.cst import ModuleNode, OperationNode
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter, SSAContext

_NL = chr(10)


def convert_code(code: str) -> str:
  """Parse Python `code` and lower it to MLIR text via `PythonToMlirEmitter`.

  Args:
      code: Python source code snippet.

  Returns:
      Serialized MLIR text representation.
  """
  tree = cst.parse_module(code.strip())
  mlir_node: ModuleNode = PythonToMlirEmitter().convert(tree)
  return mlir_node.to_text()


def test_ssa_context_scopes_and_lookups() -> None:
  """Verify `SSAContext` lexical scope nesting, SSA allocation, and root-scope guard."""
  ctx = SSAContext()
  ctx.enter_scope()
  val1 = ctx.allocate_ssa()
  assert val1.name == "%0"
  ctx.declare("foo", val1)
  assert ctx.lookup("foo") == val1
  assert ctx.lookup("bar") is None
  ctx.exit_scope()
  assert ctx.lookup("foo") is None
  ctx.exit_scope()
  assert len(ctx._scopes) == 1


def test_emitter_modules_trivia_and_blocks() -> None:
  """Verify `PythonToMlirEmitter.convert`, `_extract_trivia`, and `_emit_block` across trivia shapes."""
  emitter = PythonToMlirEmitter()
  assert emitter.convert(cst.parse_module("")).body.operations == []

  header_only = emitter.convert(cst.parse_module(f"# header{_NL}# another{_NL}"))
  assert header_only.body.operations == []

  with_header_and_footer = emitter.convert(cst.parse_module(f"# header{_NL}{_NL}x = 1  # inline{_NL}# footer{_NL}"))
  ops = with_header_and_footer.body.operations
  assert len(ops) == 1
  assert any(t.text.startswith("// header") for t in ops[0].leading_trivia)
  assert any(t.text.startswith("// inline") for t in ops[0].leading_trivia)

  # Module with blank header EmptyLine (comment=None)
  blank_hdr_mod = cst.Module(
    header=[cst.EmptyLine(comment=None, newline=cst.Newline(_NL)), cst.EmptyLine(comment=None, newline=cst.Newline(""))],
    body=[],
    footer=[],
  )
  assert emitter.convert(blank_hdr_mod).body.operations == []

  # Node without `leading_lines` attribute and custom `_LeadingLinesProtocol` shapes
  assert emitter._extract_trivia(cst.Pass()) == []

  stmt_blank_nl = cst.SimpleStatementLine(
    body=[cst.Pass()],
    leading_lines=[
      cst.EmptyLine(indent=False, comment=None, newline=cst.Newline(value="")),
      cst.EmptyLine(indent=False, comment=None, newline=cst.Newline(value=_NL)),
    ],
  )
  extracted = emitter._extract_trivia(stmt_blank_nl)
  assert len(extracted) == 1
  assert extracted[0].text == _NL

  # Sequence that is neither Suite nor list/tuple hits the non-matching `_emit_block` branch cleanly
  assert emitter._emit_block([]).operations == []
  assert emitter._emit_block(collections.deque[cst.CSTNode]()).operations == []


def test_emitter_imports_assignments_and_returns() -> None:
  """Verify `sw.import`, `sw.setattr`, `sw.return`, and multi-target/unresolved assignments."""
  import_mlir = convert_code(
    _NL.join(
      [
        "import numpy as np",
        "import os",
        "from math import pi as p, sqrt",
        "from some_module import *",
        "from . import local_sym",
      ]
    )
  )
  assert import_mlir.count("sw.import") == 5

  emitter = PythonToMlirEmitter()
  alias = cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))
  with mock.patch.object(cst.AsName, "name", cst.Tuple([])):
    assert emitter._emit_import(cst.Import(names=[alias])).name == "sw.import"

  # Assignments: local, chained, subscript, unresolved attribute, dynamic base attribute, known `self` attribute
  assert len(emitter.convert(cst.parse_module("x = y = 1")).body.operations) == 1
  assert emitter.ctx.lookup("x") is not None
  assert len(emitter.convert(cst.parse_module("a[0] = 1")).body.operations) == 1
  assert "sw.setattr" not in convert_code("self.unknown.attr = 5")
  assert "sw.setattr" not in convert_code("foo().attr = 1")

  known_emitter = PythonToMlirEmitter()
  known_emitter.ctx.declare("self", known_emitter.ctx.allocate_ssa())
  known_mod = known_emitter.convert(cst.parse_module("self.x = 1"))
  assert [op.name for op in known_mod.body.operations] == ["sw.constant", "sw.setattr"]

  assert [op.name for op in PythonToMlirEmitter().convert(cst.parse_module("return 42")).body.operations] == [
    "sw.constant",
    "sw.return",
  ]
  assert [op.name for op in PythonToMlirEmitter().convert(cst.parse_module("return")).body.operations] == ["sw.return"]


def test_emitter_declarations_control_flow_and_expressions() -> None:
  """Verify class/function declarations, control-flow stubs, binary ops, calls, and unhandled expressions."""
  cls_mlir = convert_code(
    _NL.join(
      [
        "# A leading comment",
        "class MyLayer(nn.Module, Base, dynamic_base()):",
        "    pass",
        "class BareLayer:",
        "    pass",
        "class DynamicOnly(dynamic_base()):",
        "    pass",
      ]
    )
  )
  assert "// A leading comment" in cls_mlir
  assert "sw.module" in cls_mlir
  assert 'bases = ["nn.Module", "Base"]' in cls_mlir

  fn_mlir = convert_code(
    _NL.join(
      [
        "def f(x: torch.Tensor, y: List[int], z):",
        "    # Statement leading comment",
        "    print(x)",
        "    return torch.nn.functional.relu(x, inplace=True)",
      ]
    )
  )
  assert '!sw.type<"torch.Tensor">' in fn_mlir
  assert "// Statement leading comment" in fn_mlir
  assert "arg_keywords" in fn_mlir

  assert "sw.call" in convert_code(_NL.join(["def apply_func(func, x):", "    return func(x)"]))
  assert "sw.call" in convert_code(_NL.join(["def chained(a):", "    return a().attr()"]))
  assert "sw.return" in convert_code(_NL.join(["def chained_attr(a):", "    return a().attr"]))
  assert "%error" in convert_code(_NL.join(["def lam():", "    return lambda y: y"]))

  emitter = PythonToMlirEmitter()
  assert emitter.convert(cst.parse_module(f"if True:{_NL}    pass")).body.operations == []
  assert emitter.convert(cst.parse_module(f"while True:{_NL}    pass")).body.operations == []
  assert emitter._emit_statement(cst.SimpleStatementLine(body=[])) == []
  assert emitter._dispatch_small_stmt(cst.Pass()) == []
  assert emitter._flatten_attr(cst.Pass()) is None

  # Non-Name parameter defensive branch in `_emit_func_def`
  parsed_fn = cst.ensure_type(cst.parse_statement("def f(x): pass"), cst.FunctionDef)
  with mock.patch.object(cst.Param, "name", cst.Tuple([])):
    assert emitter._emit_func_def(parsed_fn).name == "sw.func"

  # Binary operators & custom fallback
  binops_code = _NL.join(
    [
      "def math_ops(a, b):",
      "    v0 = a + b",
      "    v1 = a - b",
      "    v2 = a * b",
      "    v3 = a / b",
      "    v4 = a // b",
      "    v5 = a % b",
      "    v6 = a ** b",
      "    v7 = a @ b",
      "    v8 = a << b",
      "    v9 = a >> b",
      "    v10 = a & b",
      "    v11 = a | b",
      "    v12 = a ^ b",
      "    return v12",
    ]
  )
  bin_mlir = convert_code(binops_code)
  assert "binop.add" in bin_mlir
  assert "binop.sub" in bin_mlir
  assert "binop.xor" in bin_mlir

  class _CustomBinaryOp(cst.BaseBinaryOp):
    """Custom `BaseBinaryOp` subclass exercising the `'unknown'` operator fallback."""

    def _codegen_impl(self, state: CodegenState, default_indicator: str | None = None) -> None:
      """No-op codegen hook required by `BaseBinaryOp`.

      Args:
          state: LibCST codegen state.
          default_indicator: Optional whitespace indicator.
      """

    def _visit_and_replace_children(self, visitor: cst.CSTVisitorT) -> "_CustomBinaryOp":
      """Return `self` unchanged during child traversal.

      Args:
          visitor: LibCST visitor instance.

      Returns:
          Unchanged `_CustomBinaryOp` instance.
      """
      return self

  assert emitter._get_binop_str(_CustomBinaryOp()) == "unknown"

  class _CustomStmtEmitter(PythonToMlirEmitter):
    """Subclass returning an `OperationNode` so statement trivia prepending is exercised."""

    def _dispatch_small_stmt(self, node: cst.CSTNode) -> list[OperationNode]:
      """Return a single dummy `OperationNode`.

      Args:
          node: Inner small statement CST node.

      Returns:
          Single-element `OperationNode` list.
      """
      return [OperationNode(name="dummy", leading_trivia=[])]

  triv_ops = _CustomStmtEmitter()._emit_statement(cst.parse_statement(f"# comment{_NL}dummy()"))
  assert len(triv_ops) == 1
  assert any("// comment" in t.text for t in triv_ops[0].leading_trivia)
