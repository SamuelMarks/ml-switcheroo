"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.schema import SemanticTier
from typing import Optional, Tuple, Dict, Any


def test_api_attr_mixin_branches() -> None:
  """Test element."""
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
