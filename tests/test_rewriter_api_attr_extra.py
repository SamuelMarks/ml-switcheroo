"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from typing import Optional, Dict, Any, Tuple


class DummySemantics:
  """Test element."""

  def __init__(self) -> None:
    """Test element."""
    self.alias_map: Dict[str, str] = {"t": "torch"}
    self._key_origins: Dict[str, str] = {"abs_id": "neural"}
    self.get_definition_called: bool = False

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Test element."""
    self.get_definition_called = True
    return ("abs_id", {"variants": {}})

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Test element."""
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Test element."""
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Test element."""
    return True

  def resolve_variant(self, abs_id: str, target: str) -> Dict[str, str]:
    """Test element."""
    return {"api": "jax.func"}


def test_api_attr_mixin_branches() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 61->78: track variable init
  mod: cst.Module = cst.parse_module("a = t.func()\n")
  assign: cst.Assign = getattr(getattr(mod, "body")[0], "body")[0]
  res: cst.CSTNode = transformer.leave_Assign(assign, assign)
  assert res is assign
  assert semantics.get_definition_called

  # 139->144: leave_Attribute function with empty std_args
  semantics.get_definition_called = False
  semantics.get_definition = lambda x: ("abs_id", {"op_type": "function", "std_args": {}})
  attr: cst.BaseExpression = cst.parse_expression("t.func")
  transformer.leave_Attribute(attr, attr)

  semantics.get_definition = lambda x: ("abs_id", {"op_type": "function", "std_args": {"a": "b"}})
  res_attr2: cst.CSTNode = transformer.leave_Attribute(attr, attr)
  assert res_attr2 is attr
