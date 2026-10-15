"""Test suite for plugin utils."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import _extract_root_name, create_dotted_name, is_framework_module_node


def parse_expr(code: str) -> cst.BaseExpression:
  """Execute parse expr helper logic."""
  return cst.parse_expression(code)


def test_create_dotted_name() -> None:
  """Verify create dotted name behavior."""
  node: cst.BaseExpression = create_dotted_name("a.b.c")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code
  assert code.strip() == "a.b.c"


def test_extract_root_name_attribute() -> None:
  """Verify extract root name attribute behavior."""
  node: cst.BaseExpression = parse_expr("a.b.c")
  assert _extract_root_name(node) == "a"
  assert _extract_root_name(parse_expr("1 + 1")) is None


def test_is_framework_module_node_no_name() -> None:
  """Verify is framework module node no name behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  node: cst.BaseExpression = parse_expr("1 + 1")
  assert not is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_fw() -> None:
  """Verify is framework module node source fw behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "torch"
  ctx.target_fw = None
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_target_fw() -> None:
  """Verify is framework module node target fw behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "jax"
  ctx.target_fw = "torch"
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_direct() -> None:
  """Verify is framework module node configs direct behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {}}
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_dict() -> None:
  """Verify is framework module node configs alias dict behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {"alias": {"name": "jnp"}}}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object() -> None:
  """Verify is framework module node configs alias object behavior."""

  class AliasInfo:
    """Test helper class AliasInfo."""

    def model_dump(self) -> dict:
      """Execute model dump helper logic."""
      return {"name": "jnp"}

  class Conf:
    """Test helper class Conf."""

    alias = AliasInfo()

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object_no_dump() -> None:
  """Verify is framework module node configs alias object no dump behavior."""

  class Conf:
    """Test helper class Conf."""

    alias = {"name": "jnp"}

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_registry() -> None:
  """Verify is framework module node source registry behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {"np.random": None}
  node: cst.BaseExpression = parse_expr("np")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_no_match() -> None:
  """Verify is framework module node no match behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {}
  node: cst.BaseExpression = parse_expr("unknown_framework")
  assert not is_framework_module_node(node, ctx)
