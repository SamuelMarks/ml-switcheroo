"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.utils import create_dotted_name, is_framework_module_node, _extract_root_name


def parse_expr(code: str) -> cst.BaseExpression:
  """Docstring."""
  return cst.parse_expression(code)


def test_create_dotted_name():
  """Docstring."""
  node = create_dotted_name("a.b.c")
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(node)])]).code
  assert code.strip() == "a.b.c"


def test_extract_root_name_attribute():
  """Docstring."""
  node = parse_expr("a.b.c")
  assert _extract_root_name(node) == "a"
  assert _extract_root_name(parse_expr("1 + 1")) is None


def test_is_framework_module_node_no_name():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  node = parse_expr("1 + 1")
  assert not is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_fw():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "torch"
  ctx.target_fw = None
  node = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_target_fw():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "jax"
  ctx.target_fw = "torch"
  node = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_direct():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {}}
  node = parse_expr("torch")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_dict():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": {"alias": {"name": "jnp"}}}
  node = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object():
  """Docstring."""

  class AliasInfo:
    """Docstring."""

    def model_dump(self):
      """Docstring."""
      return {"name": "jnp"}

  class Conf:
    """Docstring."""

    alias = AliasInfo()

  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_configs_alias_object_no_dump():
  """Docstring."""

  class Conf:
    """Docstring."""

    alias = {"name": "jnp"}

  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {"torch": Conf()}
  node = parse_expr("jnp")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_source_registry():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {"np.random": None}
  node = parse_expr("np")
  assert is_framework_module_node(node, ctx)


def test_is_framework_module_node_no_match():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.source_fw = "a"
  ctx.target_fw = "b"
  ctx.semantics = MagicMock()
  ctx.semantics.framework_configs = {}
  ctx.semantics._source_registry = {}
  node = parse_expr("unknown_framework")
  assert not is_framework_module_node(node, ctx)
