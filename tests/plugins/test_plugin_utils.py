"""Test suite for plugin utils."""

import typing
from unittest.mock import MagicMock

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import _extract_root_name, create_dotted_name, is_framework_module_node


def parse_expr(code: str) -> cst.BaseExpression:
  """Execute `parse_expr` helper in plugin utils tests.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `cst.BaseExpression` from parse expr.
  """
  return cst.parse_expression(code)


def test_create_dotted_name() -> None:
  """Verify `create_dotted_name` produces expected results across plugin utils inputs."""
  node: cst.BaseExpression = create_dotted_name("a.b.c")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code
  assert code.strip() == "a.b.c"


def test_extract_root_name_attribute() -> None:
  """Verify `parse_expr` handles extract root name attribute as expected."""
  node: cst.BaseExpression = parse_expr("a.b.c")
  assert _extract_root_name(node) == "a"
  assert _extract_root_name(parse_expr("1 + 1")) is None


def test_is_framework_module_node_no_name() -> None:
  """Verify `parse_expr` handles is framework module node no name as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  node: cst.BaseExpression = parse_expr("1 + 1")
  assert not is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_framework() -> None:
  """Verify `parse_expr` handles is framework module node source framework as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "torch"
  ctx.target_framework = None
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_target_framework() -> None:
  """Verify `parse_expr` handles is framework module node target framework as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "jax"
  ctx.target_framework = "torch"
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_direct() -> None:
  """Verify `parse_expr` handles is framework module node configs direct as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {}}
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_dict() -> None:
  """Verify `parse_expr` handles is framework module node configs alias dict as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {"alias": {"name": "jnp"}}}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object() -> None:
  """Verify `parse_expr` handles is framework module node configs alias object as expected."""

  class AliasInfo:
    """Test double `AliasInfo` implementing `model_dump` for plugin utils tests."""

    def model_dump(self) -> dict[str, JsonValue]:
      """Execute `model_dump` on `AliasInfo` for plugin utils tests.

      Returns:
          Resulting `dict[str, JsonValue]` from model dump.
      """
      return {"name": "jnp"}

  class Conf:
    """Sentinel `Conf` fixture class used in plugin utils tests."""

    alias = AliasInfo()

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object_no_dump() -> None:
  """Verify `parse_expr` handles is framework module node configs alias object no dump as expected."""

  class Conf:
    """Sentinel `Conf` fixture class used in plugin utils tests."""

    alias: typing.ClassVar[dict[str, str]] = {"name": "jnp"}

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_registry() -> None:
  """Verify `parse_expr` handles is framework module node source registry as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {"np.random": None}
  node: cst.BaseExpression = parse_expr("np")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_no_match() -> None:
  """Verify `parse_expr` handles is framework module node no match as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_framework = "a"
  ctx.target_framework = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {}
  node: cst.BaseExpression = parse_expr("unknown_framework")
  assert not is_framework_module_node(node, ctx)
