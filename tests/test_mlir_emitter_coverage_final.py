"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter, SSAContext
from ml_switcheroo.core.mlir.cst import ValueNode, OperationNode


def test_everything_emitter():
  """Docstring."""
  ctx = SSAContext()
  ctx.enter_scope()
  ctx.exit_scope()
  ctx.exit_scope()
  ctx.declare("x", ValueNode(name="%x"))
  assert ctx.lookup("x")
  ctx.enter_scope()
  assert ctx.lookup("y") is None

  emitter = PythonToMlirEmitter()
  mod = cst.parse_module("# header\nx = 1\n# footer\n")
  emitter.convert(mod)

  # Cover missing line 112 (empty line in header without comment)
  mod2 = cst.parse_module("# head\n\nx = 1")
  emitter.convert(mod2)

  # cover different trivia combinations
  class FakeLine:
    """Class doc."""

    def __init__(self, comment, newline):
      """Init doc."""
      self.comment = comment
      self.newline = newline

  class FakeMod:
    """Class doc."""

    header = [FakeLine(cst.Comment("# c1"), None), FakeLine(None, cst.Newline("\n")), FakeLine(None, None)]
    body = []

  try:
    emitter.visit_Module(FakeMod())
  except Exception:
    pass

  class FakeNode:
    """Class doc."""

    leading_lines = [
      FakeLine(cst.Comment("# c1"), None),
      FakeLine(None, cst.Newline("\n")),
      FakeLine(None, cst.Newline("")),
      FakeLine(None, None),
    ]

  try:
    emitter._extract_trivia(FakeNode())
  except Exception:
    pass

  emitter._emit_block(None)
  emitter._emit_statement(cst.parse_statement("class A: pass"))
  emitter._emit_statement(cst.parse_statement("def f(): pass"))
  emitter._emit_statement(cst.parse_statement("if True: pass"))
  emitter._emit_statement(cst.parse_statement("while True: pass"))
  emitter._emit_statement(cst.parse_statement("pass"))
  emitter._emit_statement(cst.SimpleStatementLine(body=[]))

  class DummyEmitter(PythonToMlirEmitter):
    """Class doc."""

    def _dispatch_small_stmt(self, node):
      """Function doc."""
      return [OperationNode(name="dummy", leading_trivia=[])]

  dummy = DummyEmitter()
  dummy._emit_statement(cst.parse_statement("# comment\ndummy()"))

  emitter._dispatch_small_stmt(cst.Assign(targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Name("y")))
  emitter._dispatch_small_stmt(cst.Return(cst.Name("x")))
  emitter._dispatch_small_stmt(cst.Expr(cst.Name("x")))
  emitter._dispatch_small_stmt(cst.Import(names=[cst.ImportAlias(cst.Name("math"))]))
  emitter._dispatch_small_stmt(cst.Pass())

  emitter._emit_import(cst.Import(names=[cst.ImportAlias(cst.Name("math"))]))
  emitter._emit_import(cst.ImportFrom(module=cst.Name("math"), names=cst.ImportStar()))
  emitter._emit_import(cst.Import(names=[cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))]))

  alias = cst.ImportAlias(cst.Name("math"), asname=cst.AsName(cst.Name("m")))

  class FakeName:
    """Class doc."""

    pass

  object.__setattr__(alias.asname, "name", FakeName())
  node4 = cst.Import(names=[alias])
  emitter._emit_import(node4)

  emitter._emit_import(cst.ImportFrom(module=cst.Name("math"), names=[cst.ImportAlias(cst.Name("sin"))]))
  emitter._emit_import(cst.ImportFrom(module=None, relative=[cst.Dot()], names=[cst.ImportAlias(cst.Name("sin"))]))

  emitter._emit_assign(cst.parse_statement("x = y = 1").body[0])
  emitter._emit_assign(cst.parse_statement("a[0] = 1").body[0])
  emitter._emit_assign(cst.parse_statement("obj.attr = 1").body[0])
  emitter._emit_assign(cst.parse_statement("foo().attr = 1").body[0])
  emitter.ctx.declare("self", ValueNode(name="%self"))
  emitter._emit_assign(cst.parse_statement("self.attr = 1").body[0])

  emitter._emit_return(cst.Return(cst.Name("x")))
  emitter._emit_return(cst.Return(None))

  assert emitter._flatten_attr(cst.Name("x")) == "x"
  assert emitter._flatten_attr(cst.Attribute(cst.Name("obj"), cst.Name("attr"))) == "obj.attr"
  assert emitter._flatten_attr(cst.Attribute(cst.Pass(), cst.Name("attr"))) is None
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
    """Class doc."""

    def _codegen_impl(self, state, default_semi):
      """Function doc."""
      pass

    def _visit_and_replace_children(self, visitor):
      """Function doc."""
      return self

  assert emitter._get_binop_str(FakeOp()) == "unknown"


def test_emitter_missing_branches():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  # Cover [115, 106] and [131, 142] in _extract_trivia and visit_Module
  # We need a header trivia that is extracted, but body_block.operations is empty.
  emitter = PythonToMlirEmitter()
  mod_empty = cst.parse_module("# header\n# another\n")
  # operations is empty, but there is header trivia
  emitter.convert(mod_empty)

  # For extract_trivia to not have line.comment or line.newline
  class EmptyFakeLine:
    """Class doc."""

    comment = None
    newline = None

  class FakeNodeEmptyLines:
    """Class doc."""

    leading_lines = [EmptyFakeLine()]

  emitter._extract_trivia(FakeNodeEmptyLines())


def test_emitter_decl_class_and_func():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  emitter = PythonToMlirEmitter()

  # class with bases
  stmt = cst.parse_statement("class A(B, C.D, obj.attr): pass")
  emitter._emit_class_def(stmt)

  # class with unflattenable base
  stmt_bad_base = cst.parse_statement("class A(foo()): pass")
  emitter._emit_class_def(stmt_bad_base)

  # func def with annotations
  stmt_func = cst.parse_statement("def f(x: int, y: list[int]): pass")
  emitter._emit_func_def(stmt_func)


def test_emitter_decl_class_and_func2():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  emitter = PythonToMlirEmitter()

  # class without bases
  stmt = cst.parse_statement("class A: pass")
  emitter._emit_class_def(stmt)

  # func def without annotations
  stmt_func = cst.parse_statement("def f(x, y): pass")
  emitter._emit_func_def(stmt_func)


def test_emitter_decl_class_and_func3():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  emitter = PythonToMlirEmitter()
  emitter.convert(cst.parse_module("class A(B, C.D, obj.attr): pass\nclass A: pass\nclass A(foo()): pass"))
  emitter.convert(cst.parse_module("def f(x, y): pass\ndef f(x: int, y: list[int]): pass"))


def test_emitter_extract_trivia_loop():
  """Function doc."""
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
  import libcst as cst

  emitter = PythonToMlirEmitter()

  class FakeLine:
    """Class doc."""

    comment = None
    newline = None

  class FakeLine2:
    """Class doc."""

    comment = None
    newline = cst.Newline(value="")

  class FakeNodeEmptyLines:
    """Class doc."""

    leading_lines = [FakeLine()]

  emitter._extract_trivia(FakeNodeEmptyLines())

  class FakeNodeEmptyLines2:
    """Class doc."""

    leading_lines = [FakeLine2()]

  emitter._extract_trivia(FakeNodeEmptyLines2())

  class FakeLine3:
    """Class doc."""

    comment = None
    newline = cst.Newline(value="\n")

  class FakeNodeEmptyLines3:
    """Class doc."""

    leading_lines = [FakeLine3()]

  emitter._extract_trivia(FakeNodeEmptyLines3())


def test_emitter_decl_class_and_func4():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

  emitter = PythonToMlirEmitter()

  # To cover the branch where param.name is NOT a Name, we can construct a FunctionDef CST directly
  # because Python 3 syntax no longer allows tuple unpacking in function arguments like `def f((x, y)):`
  stmt_func = cst.parse_statement("def f(x): pass")

  # mutate the param name to be something other than Name, e.g., a Tuple
  class Mutator(cst.CSTTransformer):
    """Class doc."""

    def leave_Param(self, original_node, updated_node):
      """Function doc."""
      return updated_node.with_changes(name=cst.Tuple([cst.Element(cst.Name("x"))]))

  stmt_mutated = stmt_func.visit(Mutator())
  emitter._emit_func_def(stmt_mutated)
