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
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id):
    """Test element."""
    return True

  def resolve_variant(self, abs_id, target):
    """Test element."""
    return {"api": "jax.func"}


def test_api_call_mixin_branches2():
  """Test element."""
  config = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 97->100: is_super_call returns updated_node
  call = cst.parse_expression("super().func()")
  res = transformer.leave_Call(call, call)
  assert res is call

  # 110->114: abstract_id resolving
  call2 = cst.parse_expression("torch.nn.Linear()")
  # We must patch strict_mode to True via config
  config.strict_mode = True
  transformer.leave_Call(call2, call2)
  # 140->142: get details skipped if deprecated is False
  semantics.get_definition = lambda x: ("Linear", {"deprecated": True, "replaced_by": "Something"})
  call3 = cst.parse_expression("torch.nn.Linear()")
  transformer.leave_Call(call3, call3)
