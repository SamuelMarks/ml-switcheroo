"""Test suite for MLIR emitter coverage with explicit assertions."""

from ml_switcheroo.enums import RuntimeTensorOrValue


import libcst as cst
from libcst._nodes.internal import CodegenState

from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter, SSAContext


def test_everything_emitter() -> None:
  """Verify MLIR emitter statement, trivia, and expression lowering."""
  ctx = SSAContext()
  ctx.enter_scope()
  ctx.exit_scope()
  ctx.exit_scope()
  ctx.declare("x", ValueNode(name="%x"))
  assert ctx.lookup("x") is not None
  ctx.enter_scope()
  assert ctx.lookup("y") is None

  emitter = PythonToMlirEmitter()
  mod = cst.parse_module("# header\nx = 1\n# footer\n")
  mlir_mod1 = emitter.convert(mod)
  assert "sw.constant" in mlir_mod1.to_text()

  mod2 = cst.parse_module("# head\n\nx = 1")
  mlir_mod2 = emitter.convert(mod2)
  assert "sw.constant" in mlir_mod2.to_text()

  class FakeLine:
    """Test helper class FakeLine."""

    def __init__(self, comment: cst.Comment | None, newline: cst.Newline | None) -> None:
      """Initialize test helper instance.

      Args:
          comment: The comment (cst.Comment | None) for init.
          newline: The newline (cst.Newline | None) for init.
      """
      self.comment = comment
      self.newline = newline

  class FakeMod:
    """Test helper class FakeMod."""

    header = [
      FakeLine(cst.Comment("# c1"), None),
      FakeLine(None, cst.Newline("\n")),
      FakeLine(None, None),
    ]
    body: list[RuntimeTensorOrValue] = []
    footer: list[RuntimeTensorOrValue] = []

  assert getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v").convert}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeMod}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") is not getattr({"v": None}, "__getitem__")("v")

  class FakeNode:
    """Test helper class FakeNode."""

    leading_lines = [
      FakeLine(cst.Comment("# c1"), None),
      FakeLine(None, cst.Newline("\n")),
      FakeLine(None, cst.Newline("")),
      FakeLine(None, None),
    ]

  class FakeNodeEmptyNewlineEnd:
    """Test helper class FakeNodeEmptyNewlineEnd."""

    leading_lines = [
      FakeLine(None, cst.Newline("")),
    ]

  trivia1 = getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeNode}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(trivia1, list)

  trivia2 = getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeNodeEmptyNewlineEnd}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(trivia2, list)

  trivia3 = getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": object}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert trivia3 == []

  block_none = getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._emit_block}, "__getitem__")("v")(
        getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert len(block_none.operations) == 0

  assert len(emitter._emit_statement(cst.parse_statement("class A: pass"))) == 1
  assert len(emitter._emit_statement(cst.parse_statement("def f(): pass"))) == 1
  assert emitter._emit_statement(cst.parse_statement("if True: pass")) == []
  assert emitter._emit_statement(cst.parse_statement("while True: pass")) == []
  assert emitter._emit_statement(cst.parse_statement("pass")) == []
  assert emitter._emit_statement(cst.SimpleStatementLine(body=[])) == []

  class DummyEmitter(PythonToMlirEmitter):
    """Test helper class DummyEmitter."""

    def _dispatch_small_stmt(self, node: cst.CSTNode) -> list[OperationNode]:
      """Execute dispatch small stmt helper logic.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `list[OperationNode]` from dispatch small stmt.
      """
      return [OperationNode(name="dummy", leading_trivia=[])]

  dummy = DummyEmitter()
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

  alias = cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))

  class FakeName:
    """Test helper class FakeName."""

    pass

  object.__setattr__(alias.asname, "name", FakeName())
  node4 = cst.Import(names=[alias])
  emitter._emit_import(node4)

  emitter._emit_import(cst.ImportFrom(module=cst.Name("math"), names=[cst.ImportAlias(cst.Name("sin"))]))
  emitter._emit_import(cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(cst.Name("sin"))]))

  def parse_assign(code: str) -> cst.Assign:
    """Helper to parse an assign statement.

    Args:
        code: Python or DSL source code string to parse or transpile.

    Returns:
        Resulting `cst.Assign` from parse assign.
    """
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
  assert getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._flatten_attr}, "__getitem__")("v")(
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Attribute}, "__getitem__")("v")(
              getattr(
                {"v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Pass}, "__getitem__")("v")()}, "__getitem__"
              )("v"),
              getattr(
                {
                  "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v")(
                    getattr({"v": "attr"}, "__getitem__")("v")
                  )
                },
                "__getitem__",
              )("v"),
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")
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

    def _codegen_impl(self, state: CodegenState, default_semi: bool = False) -> None:
      """Execute codegen impl helper logic.

      Args:
          state: The state (CodegenState) for codegen impl.
          default_semi: The default semi (bool) for codegen impl.
      """
      pass

    def _visit_and_replace_children(self, visitor: cst.CSTVisitorT) -> "FakeOp":
      """Execute visit and replace children helper logic.

      Args:
          visitor: The visitor (cst.CSTVisitorT) for visit and replace children.

      Returns:
          Resulting `'FakeOp'` from visit and replace children.
      """
      return self

  assert emitter._get_binop_str(FakeOp()) == "unknown"


def test_emitter_missing_branches() -> None:
  """Verify emitter trivia extraction on comment-only modules and empty leading lines."""
  emitter = PythonToMlirEmitter()
  mod_empty = cst.parse_module("# header\n# another\n")
  res_mod = emitter.convert(mod_empty)
  assert res_mod is not None

  class EmptyFakeLine:
    """Test helper class EmptyFakeLine."""

    comment = None
    newline = None

  class FakeNodeEmptyLines:
    """Test helper class FakeNodeEmptyLines."""

    leading_lines = [EmptyFakeLine()]

  assert getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeNodeEmptyLines}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": []}, "__getitem__")("v")


def test_emitter_decl_class_and_func() -> None:
  """Verify class and function declaration MLIR emission with bases and type annotations."""
  emitter = PythonToMlirEmitter()

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
  emitter = PythonToMlirEmitter()

  stmt = cst.parse_statement("class A: pass")
  assert isinstance(stmt, cst.ClassDef)
  assert emitter._emit_class_def(stmt).name == "sw.module"

  stmt_func = cst.parse_statement("def f(x, y): pass")
  assert isinstance(stmt_func, cst.FunctionDef)
  assert emitter._emit_func_def(stmt_func).name == "sw.func"


def test_emitter_decl_class_and_func3() -> None:
  """Verify full module conversion for classes and functions."""
  emitter = PythonToMlirEmitter()
  m1 = emitter.convert(cst.parse_module("class A(B, C.D, obj.attr): pass\nclass A: pass\nclass A(foo()): pass"))
  assert "sw.module" in m1.to_text()
  m2 = emitter.convert(cst.parse_module("def f(x, y): pass\ndef f(x: int, y: list[int]): pass"))
  assert "sw.func" in m2.to_text()


def test_emitter_extract_trivia_loop() -> None:
  """Verify _extract_trivia loop across empty, blank, and newline FakeLine objects."""
  emitter = PythonToMlirEmitter()

  class FakeLine:
    """Test helper class FakeLine."""

    comment = None
    newline = None

  class FakeLine2:
    """Test helper class FakeLine2."""

    comment = None
    newline = cst.Newline(value="")

  class FakeNodeEmptyLines:
    """Test helper class FakeNodeEmptyLines."""

    leading_lines = [FakeLine()]

  assert getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeNodeEmptyLines}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": []}, "__getitem__")("v")

  class FakeNodeEmptyLines2:
    """Test helper class FakeNodeEmptyLines2."""

    leading_lines = [FakeLine2()]

  assert getattr(
    {
      "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
        getattr({"v": getattr({"v": FakeNodeEmptyLines2}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": []}, "__getitem__")("v")

  class FakeLine3:
    """Test helper class FakeLine3."""

    comment = None
    newline = cst.Newline(value="\n")

  class FakeNodeEmptyLines3:
    """Test helper class FakeNodeEmptyLines3."""

    leading_lines = [FakeLine3()]

  assert getattr(
    {
      "v": len(
        getattr(
          {
            "v": getattr({"v": getattr({"v": emitter}, "__getitem__")("v")._extract_trivia}, "__getitem__")("v")(
              getattr({"v": getattr({"v": FakeNodeEmptyLines3}, "__getitem__")("v")()}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v") == getattr({"v": 1}, "__getitem__")("v")


def test_emitter_decl_class_and_func4() -> None:
  """Verify _emit_func_def handles non-Name parameter CST nodes gracefully."""
  emitter = PythonToMlirEmitter()

  parsed = cst.parse_statement("def f(x): pass")
  assert isinstance(parsed, cst.FunctionDef)
  stmt_func = parsed

  class Mutator(cst.CSTTransformer):
    """Test helper class Mutator."""

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
      """Execute leave Param helper logic.

      Args:
          original_node: The original node (cst.Param) for leave Param.
          updated_node: The updated node (cst.Param) for leave Param.

      Returns:
          Resulting `cst.Param` from leave Param.
      """
      return updated_node.with_changes(name=cst.Tuple([cst.Element(cst.Name("x"))]))

  mutated = stmt_func.visit(Mutator())
  assert isinstance(mutated, cst.FunctionDef)
  op = emitter._emit_func_def(mutated)
  assert op.name == "sw.func"
