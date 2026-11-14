"""Test suite for the Nnx To Torch Params module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.nnx_to_torch_params import transform_nnx_param


def test_nnx_param_trainable(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` returns `1` for NNX param trainable.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("Param")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "Param"
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Attribute)
  assert isinstance(res.func.value.value, cst.Name)
  assert res.func.value.value.value == "torch"
  assert res.func.attr.value == "Parameter"
  assert len(res.args) == 1


def test_nnx_param_batch_stat(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` returns `2` for NNX param batch stat.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")), args=[cst.Arg(value=cst.Name("zeros"))]
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
  assert res.args[1].keyword is not None
  assert res.args[1].keyword.value == "requires_grad"
  assert isinstance(res.args[1].value, cst.Name)
  assert res.args[1].value.value == "False"


def test_nnx_param_batch_stat_already_has_requires_grad(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` handles NNX param batch stat already has requires grad as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros")), cst.Arg(keyword=cst.Name("requires_grad"), value=cst.Name("False"))],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2


def test_nnx_param_missing_api(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` handles NNX param missing API as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(func=cst.Name("Param"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value=None), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert res is node


def test_nnx_param_leaf_name_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` returns `0` for NNX param leaf name fallback.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(func=cst.Name("Unknown"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 0


def test_nnx_param_unsupported_type(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies that an unsupported func node type returns None from _get_leaf_name.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(func=cst.SimpleString("'string'"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Attribute)
  assert isinstance(res.func.value.value, cst.Name)
  assert res.func.value.value.value == "torch"
  assert res.func.attr.value == "Parameter"


def test_nnx_param_batch_stat_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` returns `1` for NNX param batch stat no args.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 1
  assert res.args[0].keyword is not None
  assert res.args[0].keyword.value == "requires_grad"


def test_nnx_param_batch_stat_explicit_comma(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` handles NNX param batch stat explicit comma as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("BatchStat")),
    args=[cst.Arg(value=cst.Name("zeros"), comma=cst.Comma())],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.current_op_id = "BatchStat"
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="torch.nn.Parameter"), raising=False)
  res: cst.CSTNode | cst.Call = transform_nnx_param(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2


def test_nnx_param_value_keyword_normalized() -> None:
  """Verify value= keyword initializer is unwrapped to positional arg."""
  node = cst.ensure_type(cst.parse_expression("nnx.Param(value=zeros(1))"), cst.Call)
  sem = MagicMock()
  sem.resolve_variant.return_value = {"api": "torch.nn.Parameter"}
  ctx = HookContext(semantics=sem, config=MagicMock())
  ctx.current_op_id = "Param"
  res = transform_nnx_param(node, ctx)
  assert len(res.args) == 1
  assert res.args[0].keyword is None
