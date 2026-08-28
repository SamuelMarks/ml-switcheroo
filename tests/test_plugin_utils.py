"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node, _extract_root_name


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  return cst.parse_expression(code)


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = create_dotted_name("a.b.c")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code
  assert code.strip() == "a.b.c"


def test_extract_root_name_attribute() -> None:
  """Docstring."""
  node: cst.BaseExpression = parse_expr("a.b.c")
  assert _extract_root_name(node) == "a"
  assert _extract_root_name(parse_expr("1 + 1")) is None


def test_is_framework_module_node_no_name() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  node: cst.BaseExpression = parse_expr("1 + 1")
  assert not is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_fw() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "torch"
  ctx.target_fw = None
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_target_fw() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "jax"
  ctx.target_fw = "torch"
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_direct() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {}}
  node: cst.BaseExpression = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_dict() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {"alias": {"name": "jnp"}}}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object() -> None:
  """Docstring."""

  class AliasInfo:
    """Docstring."""

    def model_dump(self) -> dict:
      """Docstring."""
      return {"name": "jnp"}

  class Conf:
    """Docstring."""

    alias = AliasInfo()

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object_no_dump() -> None:
  """Docstring."""

  class Conf:
    """Docstring."""

    alias = {"name": "jnp"}

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node: cst.BaseExpression = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_registry() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {"np.random": None}
  node: cst.BaseExpression = parse_expr("np")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_no_match() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {}
  node: cst.BaseExpression = parse_expr("unknown_framework")
  assert not is_framework_module_node(node, ctx)
