"""Test suite for MLIR emitter coverage with explicit assertions."""

import typing
from typing import List, Optional

import libcst as cst

from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter, SSAContext


def test_everything_emitter() -> None:
  """Verify MLIR emitter statement, trivia, and expression lowering."""
  ctx: SSAContext = SSAContext()
  ctx.enter_scope()
  ctx.exit_scope()
  ctx.exit_scope()
  ctx.declare("x", ValueNode(name="%x"))
  assert ctx.lookup("x") is not None
  ctx.enter_scope()
  assert ctx.lookup("y") is None

  emitter: PythonToMlirEmitter = PythonToMlirEmitter()
  mod: cst.Module = cst.parse_module("# header\nx = 1\n# footer\n")
  mlir_mod1 = emitter.convert(mod)
  assert "sw.constant" in mlir_mod1.to_text()

  mod2: cst.Module = cst.parse_module("# head\n\nx = 1")
  mlir_mod2 = emitter.convert(mod2)
  assert "sw.constant" in mlir_mod2.to_text()

  class FakeLine:
    """Test helper class FakeLine."""

    def __init__(self, comment: Optional[cst.Comment], newline: Optional[cst.Newline]) -> None:
      """Initialize test helper instance."""
      self.comment: Optional[cst.Comment] = comment
      self.newline: Optional[cst.Newline] = newline

  class FakeMod:
    """Test helper class FakeMod."""

    header: List[FakeLine] = [
      FakeLine(cst.Comment("# c1"), None),
      FakeLine(None, cst.Newline("\n")),
      FakeLine(None, None),
    ]
    body: List[cst.BaseStatement] = []
    footer: List[FakeLine] = []

  assert emitter.convert(typing.cast(cst.Module, FakeMod())) is not None

  class FakeNode:
    """Test helper class FakeNode."""

    leading_lines: List[FakeLine] = [
      FakeLine(cst.Comment("# c1"), None),
      FakeLine(None, cst.Newline("\n")),
      FakeLine(None, cst.Newline("")),
      FakeLine(None, None),
    ]

  class FakeNodeEmptyNewlineEnd:
    """Test helper class FakeNodeEmptyNewlineEnd."""

    leading_lines: List[FakeLine] = [
      FakeLine(None, cst.Newline("")),
    ]

  trivia1 = emitter._extract_trivia(FakeNode())
  assert isinstance(trivia1, list)

  trivia2 = emitter._extract_trivia(FakeNodeEmptyNewlineEnd())
  assert isinstance(trivia2, list)

  trivia3 = emitter._extract_trivia(object())
  assert trivia3 == []

  block_none = emitter._emit_block(typing.cast(cst.BaseSuite, None))
  assert len(block_none.operations) == 0

  assert len(emitter._emit_statement(cst.parse_statement("class A: pass"))) == 1
  assert len(emitter._emit_statement(cst.parse_statement("def f(): pass"))) == 1
  assert emitter._emit_statement(cst.parse_statement("if True: pass")) == []
  assert emitter._emit_statement(cst.parse_statement("while True: pass")) == []
  assert emitter._emit_statement(cst.parse_statement("pass")) == []
  assert emitter._emit_statement(cst.SimpleStatementLine(body=[])) == []

  class DummyEmitter(PythonToMlirEmitter):
    """Test helper class DummyEmitter."""

    def _dispatch_small_stmt(self, node: cst.BaseSmallStatement) -> List[OperationNode]:
      """Execute dispatch small stmt helper logic."""
      return [OperationNode(name="dummy", leading_trivia=[])]

  dummy: DummyEmitter = DummyEmitter()
  dummy_ops = dummy._emit_statement(cst.parse_statement("# comment\ndummy()"))
  assert len(dummy_ops) == 1

  assert isinstance(
    emitter._dispatch_small_stmt(cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("y"))), list
  )
  assert isinstance(emitter._dispatch_small_stmt(cst.Return(cst.Name("x"))), list)
  assert isinstance(emitter._dispatch_small_stmt(cst.Expr(cst.Name("x"))), list)
  assert len(emitter._dispatch_small_stmt(cst.Import(names=[cst.ImportAlias(cst.Name("math"))]))) == 1
  assert emitter._dispatch_small_stmt(cst.Pass()) == []

  emitter._emit_import(cst.Import(names=[cst.ImportAlias(cst.Name("math"))]))
  emitter._emit_import(cst.ImportFrom(module=cst.Name("math"), names=cst.ImportStar()))
  emitter._emit_import(cst.Import(names=[cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))]))

  alias: cst.ImportAlias = cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))

  class FakeName:
    """Test helper class FakeName."""

    pass

  object.__setattr__(alias.asname, "name", FakeName())
  node4: cst.Import = cst.Import(names=[alias])
  emitter._emit_import(node4)

  emitter._emit_import(cst.ImportFrom(module=cst.Name("math"), names=[cst.ImportAlias(cst.Name("sin"))]))
  emitter._emit_import(cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(cst.Name("sin"))]))

  def parse_assign(code: str) -> cst.Assign:
    """Helper to parse an assign statement."""
    stmt = cst.parse_statement(code)
    assert isinstance(stmt, cst.SimpleStatementLine)
    assign = stmt.body[0]
    assert isinstance(assign, cst.Assign)
    return assign

  assert len(emitter._emit_assign(parse_assign("x = y = 1"))) == 1
  assert len(emitter._emit_assign(parse_assign("a[0] = 1"))) == 1
  assert len(emitter._emit_assign(parse_assign("obj.attr = 1"))) == 1
  assert len(emitter._emit_assign(parse_assign("foo().attr = 1"))) == 1
  emitter.ctx.declare("self", ValueNode(name="%self"))
  assert len(emitter._emit_assign(parse_assign("self.attr = 1"))) >= 1

  assert len(emitter._emit_return(cst.Return(cst.Name("x")))) >= 1
  assert len(emitter._emit_return(cst.Return(None))) == 1

  assert emitter._flatten_attr(cst.Name("x")) == "x"
  assert emitter._flatten_attr(cst.Attribute(cst.Name("obj"), cst.Name("attr"))) == "obj.attr"
  assert emitter._flatten_attr(cst.Attribute(typing.cast(cst.BaseExpression, cst.Pass()), cst.Name("attr"))) is None
  assert emitter._flatten_attr(cst.Pass()) is None

  assert emitter._get_binop_str(cst.Add()) == "add"
  assert emitter._get_binop_str(cst.Subtract()) == "sub"
  assert emitter._get_binop_str(cst.Multiply()) == "mul"
  assert emitter._get_binop_str(cst.Divide()) == "div"
  assert emitter._get_binop_str(cst.FloorDivide()) == "floordiv"
  assert emitter._get_binop_str(cst.Modulo()) == "mod"
  assert emitter._get_binop_str(cst.Power()) == "pow"
  assert emitter._get_binop_str(cst.MatrixMultiply()) == "matmul"
  assert emitter._get_binop_str(cst.LeftShift()) == "lshift"
  assert emitter._get_binop_str(cst.RightShift()) == "rshift"
  assert emitter._get_binop_str(cst.BitAnd()) == "and"
  assert emitter._get_binop_str(cst.BitOr()) == "or"
  assert emitter._get_binop_str(cst.BitXor()) == "xor"

  class FakeOp(cst.BaseBinaryOp):
    """Test helper class FakeOp."""

    def _codegen_impl(self, state: object, default_semi: bool = False) -> None:
      """Execute codegen impl helper logic."""
      pass

    def _visit_and_replace_children(self, visitor: object) -> cst.CSTNode:
      """Execute visit and replace children helper logic."""
      return self

  assert emitter._get_binop_str(FakeOp()) == "unknown"


def test_emitter_missing_branches() -> None:
  """Verify emitter trivia extraction on comment-only modules and empty leading lines."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()
  mod_empty: cst.Module = cst.parse_module("# header\n# another\n")
  res_mod = emitter.convert(mod_empty)
  assert res_mod is not None

  class EmptyFakeLine:
    """Test helper class EmptyFakeLine."""

    comment: None = None
    newline: None = None

  class FakeNodeEmptyLines:
    """Test helper class FakeNodeEmptyLines."""

    leading_lines: List[EmptyFakeLine] = [EmptyFakeLine()]

  assert emitter._extract_trivia(FakeNodeEmptyLines()) == []


def test_emitter_decl_class_and_func() -> None:
  """Verify class and function declaration MLIR emission with bases and type annotations."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()

  stmt = cst.parse_statement("class A(B, C.D, obj.attr): pass")
  assert isinstance(stmt, cst.ClassDef)
  op_cls = emitter._emit_class_def(stmt)
  assert op_cls.name == "sw.module"

  stmt_bad_base = cst.parse_statement("class A(foo()): pass")
  assert isinstance(stmt_bad_base, cst.ClassDef)
  op_bad = emitter._emit_class_def(stmt_bad_base)
  assert op_bad.name == "sw.module"

  stmt_func = cst.parse_statement("def f(x: int, y: list[int]): pass")
  assert isinstance(stmt_func, cst.FunctionDef)
  op_fn = emitter._emit_func_def(stmt_func)
  assert op_fn.name == "sw.func"


def test_emitter_decl_class_and_func2() -> None:
  """Verify class and function declaration MLIR emission without bases or annotations."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()

  stmt = cst.parse_statement("class A: pass")
  assert isinstance(stmt, cst.ClassDef)
  assert emitter._emit_class_def(stmt).name == "sw.module"

  stmt_func = cst.parse_statement("def f(x, y): pass")
  assert isinstance(stmt_func, cst.FunctionDef)
  assert emitter._emit_func_def(stmt_func).name == "sw.func"


def test_emitter_decl_class_and_func3() -> None:
  """Verify full module conversion for classes and functions."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()
  m1 = emitter.convert(cst.parse_module("class A(B, C.D, obj.attr): pass\nclass A: pass\nclass A(foo()): pass"))
  assert "sw.module" in m1.to_text()
  m2 = emitter.convert(cst.parse_module("def f(x, y): pass\ndef f(x: int, y: list[int]): pass"))
  assert "sw.func" in m2.to_text()


def test_emitter_extract_trivia_loop() -> None:
  """Verify _extract_trivia loop across empty, blank, and newline FakeLine objects."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()

  class FakeLine:
    """Test helper class FakeLine."""

    comment: None = None
    newline: None = None

  class FakeLine2:
    """Test helper class FakeLine2."""

    comment: None = None
    newline: cst.Newline = cst.Newline(value="")

  class FakeNodeEmptyLines:
    """Test helper class FakeNodeEmptyLines."""

    leading_lines: List[FakeLine] = [FakeLine()]

  assert emitter._extract_trivia(FakeNodeEmptyLines()) == []

  class FakeNodeEmptyLines2:
    """Test helper class FakeNodeEmptyLines2."""

    leading_lines: List[FakeLine2] = [FakeLine2()]

  assert emitter._extract_trivia(FakeNodeEmptyLines2()) == []

  class FakeLine3:
    """Test helper class FakeLine3."""

    comment: None = None
    newline: cst.Newline = cst.Newline(value="\n")

  class FakeNodeEmptyLines3:
    """Test helper class FakeNodeEmptyLines3."""

    leading_lines: List[FakeLine3] = [FakeLine3()]

  assert len(emitter._extract_trivia(FakeNodeEmptyLines3())) == 1


def test_emitter_decl_class_and_func4() -> None:
  """Verify _emit_func_def handles non-Name parameter CST nodes gracefully."""
  emitter: PythonToMlirEmitter = PythonToMlirEmitter()

  parsed = cst.parse_statement("def f(x): pass")
  assert isinstance(parsed, cst.FunctionDef)
  stmt_func: cst.FunctionDef = parsed

  class Mutator(cst.CSTTransformer):
    """Test helper class Mutator."""

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
      """Execute leave Param helper logic."""
      return updated_node.with_changes(name=typing.cast(cst.Name, cst.Tuple([cst.Element(cst.Name("x"))])))

  mutated = stmt_func.visit(Mutator())
  assert isinstance(mutated, cst.FunctionDef)
  op = emitter._emit_func_def(mutated)
  assert op.name == "sw.func"
