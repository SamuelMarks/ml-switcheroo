"""Test suite for plugin mlx extras."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_extras import _create_dotted_name, transform_compiler, transform_synchronize


def _parse_expr(code: str) -> cst.BaseExpression:
  """Parse a single-statement expression into a LibCST BaseExpression node.

  Args:
      code: Python expression source string.

  Returns:
      Parsed `cst.BaseExpression` AST node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return expr.value


def test_create_dotted_name() -> None:
  """Verify `_create_dotted_name` produces expected results across plugin MLX extras inputs."""
  node: cst.BaseExpression = _create_dotted_name("mlx.core.compile")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "compile"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "core"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "mlx"


def test_transform_compiler_decorator() -> None:
  """Verify `transform_compiler` handles decorator as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  code: str = "@torch.compile(fullgraph=True)\ndef foo(): pass"
  module: cst.Module = cst.parse_module(code)
  func_def = cst.ensure_type(module.body[0], cst.FunctionDef)
  decorator_node: cst.Decorator = func_def.decorators[0]

  transformed_node: cst.CSTNode = transform_compiler(decorator_node, ctx)
  assert isinstance(transformed_node, cst.Decorator)
  assert isinstance(transformed_node.decorator, cst.Attribute)
  assert transformed_node.decorator.attr.value == "compile"


def test_transform_compiler_call_with_args() -> None:
  """Verify `transform_compiler` returns `1` for call with args."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  call_node: cst.BaseExpression = _parse_expr("torch.compile(fn, backend='inductor')")

  transformed: cst.CSTNode = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "compile"
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].value, cst.Name)
  assert transformed.args[0].value.value == "fn"


def test_transform_compiler_call_without_args() -> None:
  """Verify `transform_compiler` returns `0` for call without args."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  call_node: cst.BaseExpression = _parse_expr("torch.compile()")

  transformed: cst.CSTNode = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_compiler_not_decorator_or_call() -> None:
  """Verify `transform_compiler` handles not decorator or call as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  node: cst.BaseExpression = _parse_expr("torch.compile")

  transformed: cst.CSTNode = transform_compiler(node, ctx)
  assert transformed is node


def test_transform_synchronize() -> None:
  """Verify `ensure_type` returns `1` for transform synchronize."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  call_node: cst.Call = cst.ensure_type(_parse_expr("torch.cuda.synchronize()"), cst.Call)

  transformed: cst.CSTNode = transform_synchronize(call_node, ctx)

  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "print"
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].value, cst.SimpleString)
  assert "Global sync requires explicit tensor args" in transformed.args[0].value.value


def test_transform_compiler_call_keyword_fun() -> None:
  """Verify transform_compiler extracts callable passed via keyword fun=."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"
  node = cst.ensure_type(cst.parse_expression("compile(fun=my_fn, fullgraph=True)"), cst.Call)
  transformed = transform_compiler(node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert transformed.args[0].keyword is None
  assert isinstance(transformed.args[0].value, cst.Name)
  assert transformed.args[0].value.value == "my_fn"
