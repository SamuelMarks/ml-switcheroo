"""Test module."""

from typing import Any, Dict, Optional, Tuple

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.schema import SemanticTier


def test_api_attr_mixin_branches() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    framework_configs: Dict[str, Any] = {}
    _key_origins: Dict[str, str] = {"dummy_call": SemanticTier.NEURAL.value}

    def get_definition(self, name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      if name == "dummy.call":
        return ("dummy_call", {"op_type": "function", "std_args": []})
      if name == "dummy.call2":
        return ("dummy_call2", {"op_type": "function"})  # No std_args
      return None

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
      return {}

    def resolve_variant(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
      return {"target": "jax.numpy.float32", "api": "jax.numpy.float32"}

    def is_verified(self, name: str) -> bool:
      return True

  context: RewriterContext = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  transformer.__dict__["source_traits"] = type("StructuralTraits", (), {"functional_execution_method": "apply"})()
  # 61->78 branch in leave_Assign
  # When target_name is falsy
  stmt: cst.Assign = getattr(cst.parse_statement("a, b = dummy.call()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)

  # 139->144 branch in leave_Attribute
  # "std_args" in details and details["std_args"] is false
  attr_node: cst.Attribute = getattr(getattr(cst.parse_statement("dummy.call"), "body")[0], "value")
  transformer.leave_Attribute(attr_node, attr_node)

  attr_node2: cst.Attribute = getattr(getattr(cst.parse_statement("dummy.call2"), "body")[0], "value")
  transformer.leave_Attribute(attr_node2, attr_node2)


# --- Merged from test_rewriter_api_attr_extra.py ---


class DummySemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.alias_map: Dict[str, str] = {"t": "torch"}
    self._key_origins: Dict[str, str] = {"abs_id": "neural"}
    self.get_definition_called: bool = False

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Docstring."""
    self.get_definition_called = True
    return ("abs_id", {"variants": {}})

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Docstring."""
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Docstring."""
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Docstring."""
    return True

  def resolve_variant(self, abs_id: str, target: str) -> Dict[str, str]:
    """Docstring."""
    return {"api": "jax.func"}


def test_api_attr_mixin_branches_extra() -> None:
  """Docstring."""
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
