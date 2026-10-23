"""Test suite for rewriter api helpers."""

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map = {"t": "torch"}
    self._key_origins = {"abs_id": "neural"}
    self.framework_configs: dict[str, dict[str, JsonValue]] = {}

  def get_definition(self, func_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        func_name: Function or qualified API name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return None

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(fw, {"alias": {}})

  def resolve_op_id(self, fw: str, name: str) -> str | None:
    """Execute resolve op id helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).
        name: Identifier or operation name string.

    Returns:
        Resolved canonical operation ID, or None if unmapped.
    """
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Execute is verified helper logic.

    Args:
        abs_id: Abstract operation identifier string.

    Returns:
        True if the operation is verified, False otherwise.
    """
    return True

  def resolve_variant(self, a: str, b: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        a: The a (str) for resolve variant.
        b: The b (str) for resolve variant.

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    return None


def test_api_helpers_module_alias() -> None:
  """Verify api helpers module alias behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  semantics.framework_configs = {
    "fw1": {"alias": {"module": "mod1"}},
    "fw2": {"alias": {}},  # empty dict
  }

  assert transformer._is_module_alias(cst.parse_expression("mod1.Linear"))
  assert not transformer._is_module_alias(cst.parse_expression("mod3.Linear"))


def test_api_helpers_branches_missing() -> None:
  """Verify api helpers branches missing behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 177->182: _inject_stmts_to_body branch when empty body
  stmt_empty = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  transformer._inject_stmts_to_body(stmt_empty, [getattr(cst.parse_module("a = 1"), "body")[0]])

  # 223->225: get_mapping silent=False
  setattr(semantics, "get_definition", lambda x: ("abs_id", {}))
  setattr(semantics, "resolve_variant", lambda a, b: None)
  res_map = transformer._get_mapping("foo", silent=False)
  assert res_map is None

  # 294->292, 296->292: resolve_tensor_methods
  # Wait, the branches are in _is_framework_base!
  # lines 294, 296 are in _is_framework_base.
  # 293: base = traits.get(...)
  # 294: if base:
  # 296: if name in self._known_module_bases:
  # Let's hit the branches where base is None
  semantics.framework_configs = {"fw1": {"traits": {}}}
  setattr(transformer, "_known_module_bases", None)
  transformer._is_framework_base("foo")

  # 361->365: check_version_constraints
  # Mock context config properly to have a version we can control
  setattr(semantics, "get_framework_config", lambda x: {"version": "1.0"})
  assert transformer.check_version_constraints("0.5", "2.0") is None

  res2 = transformer.check_version_constraints("1.5", None)
  assert res2 is not None

  res3 = transformer.check_version_constraints(None, "0.5")
  assert res3 is not None
