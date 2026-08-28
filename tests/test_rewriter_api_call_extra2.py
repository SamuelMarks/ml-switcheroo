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

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Test element."""
    return None

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Test element."""
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Test element."""
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Test element."""
    return True

  def resolve_variant(self, abs_id: str, target: str) -> Dict[str, str]:
    """Test element."""
    return {"api": "jax.func"}


def test_api_call_mixin_branches2() -> None:
  """Test element."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 97->100: is_super_call returns updated_node
  call: cst.Call = typing.cast(
    cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("super().func()").body[0]).body[0]
  ).value
  res: cst.CSTNode = transformer.leave_Call(call, call)
  assert res is call

  # 110->114: abstract_id resolving
  call2: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("torch.nn.Linear()").body[0]).body[0]),
    "value",
  )
  # We must patch strict_mode to True via config
  config.strict_mode = True
  transformer.leave_Call(call2, call2)
  # 140->142: get details skipped if deprecated is False
  semantics.get_definition = lambda x: ("Linear", {"deprecated": True, "replaced_by": "Something"})
  call3: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("torch.nn.Linear()").body[0]).body[0]),
    "value",
  )
  transformer.leave_Call(call3, call3)
