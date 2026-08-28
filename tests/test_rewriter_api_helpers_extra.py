"""Test module."""

import typing

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from typing import Dict, Any, Optional, Tuple


class DummySemantics:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.alias_map: Dict[str, str] = {"t": "torch"}
    self._key_origins: Dict[str, str] = {"abs_id": "neural"}
    self.framework_configs: Dict[str, Any] = {}

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Test element."""
    return None

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Test element."""
    return self.framework_configs.get(fw, {"alias": {}})

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Test element."""
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Test element."""
    return True

  def resolve_variant(self, a: Any, b: Any) -> Optional[Dict[str, Any]]:
    """Test element."""
    return None


def test_api_helpers_module_alias() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

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
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 177->182: _inject_stmts_to_body branch when empty body
  stmt_empty: cst.FunctionDef = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  transformer._inject_stmts_to_body(stmt_empty, [getattr(cst.parse_module("a = 1"), "body")[0]])

  # 223->225: get_mapping silent=False
  semantics.get_definition = lambda x: ("abs_id", {})
  semantics.resolve_variant = lambda a, b: None
  res_map: Optional[Dict[str, Any]] = transformer._get_mapping("foo", silent=False)
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
  semantics.get_framework_config = lambda x: {"version": "1.0"}
  assert transformer.check_version_constraints("0.5", "2.0") is None

  res2: Optional[Dict[str, Any]] = transformer.check_version_constraints("1.5", None)
  assert res2 is not None

  res3: Optional[Dict[str, Any]] = transformer.check_version_constraints(None, "0.5")
  assert res3 is not None
