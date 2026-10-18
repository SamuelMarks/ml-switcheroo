"""Test suite for rewriter api helpers."""


# --- Merged from test_rewriter_api_helpers_extra.py ---

import typing
from typing import Any

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map: typing.Any = {"t": "torch"}
    self._key_origins: typing.Any = {"abs_id": "neural"}
    self.framework_configs: typing.Any = {}

  def get_definition(self, func_name: typing.Any) -> typing.Any:
    """Execute get definition helper logic.

    Args:
        func_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def get_framework_config(self, fw: typing.Any) -> typing.Any:
    """Execute get framework config helper logic.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(fw, {"alias": {}})

  def resolve_op_id(self, fw: typing.Any, name: typing.Any) -> typing.Any:
    """Execute resolve op id helper logic.

    Args:
        fw: Test argument or fixture.
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def is_verified(self, abs_id: typing.Any) -> typing.Any:
    """Execute is verified helper logic.

    Args:
        abs_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def resolve_variant(self, a: Any, b: Any) -> typing.Any:
    """Execute resolve variant helper logic.

    Args:
        a: Test argument or fixture.
        b: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None


def test_api_helpers_module_alias() -> None:
  """Verify api helpers module alias behavior."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: typing.Any = DummySemantics()
  context: typing.Any = RewriterContext(semantics=semantics, config=config)
  transformer: typing.Any = ApiTransformer(context)

  semantics.framework_configs = {
    "fw1": {"alias": {"module": "mod1"}},
    "fw2": {"alias": {}},  # empty dict
  }

  assert transformer._is_module_alias(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("mod1.Linear").body[0]).body[0]).value
  )
  assert not transformer._is_module_alias(
    getattr(
      typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("mod3.Linear").body[0]).body[0]),
      "value",
    )
  )


def test_api_helpers_branches_missing() -> None:
  """Verify api helpers branches missing behavior."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: typing.Any = DummySemantics()
  context: typing.Any = RewriterContext(semantics=semantics, config=config)
  transformer: typing.Any = ApiTransformer(context)

  # 177->182: _inject_stmts_to_body branch when empty body
  stmt_empty: typing.Any = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  transformer._inject_stmts_to_body(stmt_empty, [getattr(cst.parse_module("a = 1"), "body")[0]])

  # 223->225: get_mapping silent=False
  setattr(semantics, "get_definition", lambda x: ("abs_id", {}))
  setattr(semantics, "resolve_variant", lambda a, b: None)
  res_map: typing.Any = transformer._get_mapping("foo", silent=False)
  assert res_map is None

  # 294->292, 296->292: resolve_tensor_methods
  # Wait, the branches are in _is_framework_base!
  # lines 294, 296 are in _is_framework_base.
  # 293: base = traits.get(...)
  # 294: if base:
  # 296: if name in self._known_module_bases:
  # Let's hit the branches where base is None
  semantics.framework_configs = {"fw1": {"traits": {}}}
  transformer._known_module_bases = None
  transformer._is_framework_base("foo")

  # 361->365: check_version_constraints
  # Mock context config properly to have a version we can control
  setattr(semantics, "get_framework_config", lambda x: {"version": "1.0"})
  assert transformer.check_version_constraints("0.5", "2.0") is None

  res2: typing.Any = transformer.check_version_constraints("1.5", None)
  assert res2 is not None

  res3: typing.Any = transformer.check_version_constraints(None, "0.5")
  assert res3 is not None
