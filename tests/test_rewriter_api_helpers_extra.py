"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig


class DummySemantics:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.alias_map = {"t": "torch"}
    self._key_origins = {"abs_id": "neural"}

  def get_definition(self, func_name):
    """Test element."""
    return None

  def get_framework_config(self, fw):
    """Test element."""
    return {"alias": {}}

  def resolve_op_id(self, fw, name):
    """Test element."""
    return None

  def is_verified(self, abs_id):
    """Test element."""
    return True


def test_api_helpers_module_alias():
  """Test element."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  semantics.framework_configs = {
    "fw1": {"alias": {"module": "mod1"}},
    "fw2": {"alias": {}},  # empty dict
  }

  assert transformer._is_module_alias(cst.parse_expression("mod1.Linear"))
  assert not transformer._is_module_alias(cst.parse_expression("mod3.Linear"))


def test_api_helpers_branches_missing():
  """Test element."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 177->182: _inject_stmts_to_body branch when empty body
  stmt_empty = cst.parse_statement("def foo(): pass")
  transformer._inject_stmts_to_body(stmt_empty, [cst.parse_statement("a = 1")])

  # 223->225: get_mapping silent=False
  semantics.get_definition = lambda x: ("abs_id", {})
  semantics.resolve_variant = lambda a, b: None
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
  transformer._known_module_bases = None
  transformer._is_framework_base("foo")

  # 361->365: check_version_constraints
  # Mock context config properly to have a version we can control
  semantics.get_framework_config = lambda x: {"version": "1.0"}
  assert transformer.check_version_constraints("0.5", "2.0") is None

  res2 = transformer.check_version_constraints("1.5", None)
  assert res2 is not None

  res3 = transformer.check_version_constraints(None, "0.5")
  assert res3 is not None
