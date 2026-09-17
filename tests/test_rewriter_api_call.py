"""Test module."""


# --- Merged from test_rewriter_api_call_extra.py ---

import typing
from typing import Any, Dict, Optional, Tuple

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


class DummySemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.alias_map: Dict[str, str] = {"t": "torch"}
    self._key_origins: Dict[str, str] = {"abs_id": "neural"}

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Docstring."""
    return None

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Docstring."""
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Docstring."""
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Docstring."""
    return True

  def resolve_variant(self, abs_id: str, target: str) -> Dict[str, str]:
    """Docstring."""
    return {"api": "jax.func"}


def test_api_call_mixin_branches() -> None:
  """Docstring."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 97->100: is_super_call returns updated_node
  call: cst.Call = cst.parse_expression("super().func()")
  res: cst.CSTNode = transformer.leave_Call(call, call)
  assert res is call

  # 110->114: abstract_id resolving
  call2: cst.Call = cst.parse_expression("torch.nn.Linear()")
  res2: cst.CSTNode = transformer.leave_Call(call2, call2)
  assert res2 is call2  # actually it reports failure since tier is neural

  # 140->142: get details skipped if deprecated is False
  semantics.get_definition = lambda x: ("Linear", {"deprecated": True, "replaced_by": "Something"})
  call3: cst.Call = cst.parse_expression("torch.nn.Linear()")
  transformer.leave_Call(call3, call3)


# --- Merged from test_rewriter_api_call_extra2.py ---


class DummySemanticsExtra:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.alias_map: Dict[str, str] = {"t": "torch"}
    self._key_origins: Dict[str, str] = {"abs_id": "neural"}

  def get_definition(self, func_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Docstring."""
    return None

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Docstring."""
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> Optional[str]:
    """Docstring."""
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Docstring."""
    return True

  def resolve_variant(self, abs_id: str, target: str) -> Dict[str, str]:
    """Docstring."""
    return {"api": "jax.func"}


def test_api_call_mixin_branches2() -> None:
  """Docstring."""
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


def test_api_call_mixin_remaining_branches(monkeypatch: typing.Any) -> None:
  """Test remaining branches in ApiTransformerCallMixin."""
  config: RuntimeConfig = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 109->112: resolve_implicit_method returns guessed name, but _get_mapping returns None
  monkeypatch.setattr(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method",
    lambda *args, **kwargs: "torch.unmapped_guessed",
  )
  call_guess: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("x.func()").body[0]).body[0]),
    "value",
  )
  assert transformer.leave_Call(call_guess, call_guess) is call_guess

  # 116->119: is_builtin returns True
  call_builtin: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("len([1, 2])").body[0]).body[0]),
    "value",
  )
  assert transformer.leave_Call(call_builtin, call_builtin) is call_builtin

  # 129->133 and 136: lookup_id from abstract_id is not empty, neural tier to pure math jax
  config.strict_mode = True
  semantics.resolve_op_id = lambda fw, name: "nn_linear"
  semantics._key_origins = {"nn_linear": "neural"}
  call_strict: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("torch.nn_linear()").body[0]).body[0]),
    "value",
  )
  transformer.leave_Call(call_strict, call_strict)
  assert any("Cannot map neural network abstraction" in err for err in transformer.context.current_stmt_errors)
  transformer.context.current_stmt_errors.clear()

  # 159->161: deprecated operation without replaced_by
  semantics.get_definition = lambda x: ("DeprecatedOp", {"deprecated": True})
  monkeypatch.setattr(transformer, "_get_mapping", lambda name, **kwargs: {"api": "jax.some_func"})
  call_dep: cst.Call = getattr(
    typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, cst.parse_module("torch.dep()").body[0]).body[0]),
    "value",
  )
  transformer.leave_Call(call_dep, call_dep)
