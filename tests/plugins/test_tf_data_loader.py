"""Test suite for the Tf Data Loader module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.tf_data_loader import _extract_tensor_dataset_inputs, _get_arg_by_name, transform_tf_dataloader


def test_get_arg_by_name() -> None:
  """Gets argument by name."""
  arg: cst.Arg = cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True"))
  assert _get_arg_by_name([arg], "shuffle") is arg
  assert _get_arg_by_name([arg], "missing") is None


def test_extract_tensor_dataset_inputs() -> None:
  """Verify `Call` returns `2` for extract tensor dataset inputs."""
  node: cst.Call = cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])
  res: list[cst.BaseExpression] | None = _extract_tensor_dataset_inputs(node)
  assert res is not None
  assert len(res) == 2


def test_extract_tensor_dataset_inputs_attribute() -> None:
  """Verify `Call` handles extract tensor dataset inputs attribute as expected."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("data"), attr=cst.Name("TensorDataset")), args=[cst.Arg(cst.Name("x"))]
  )
  res: list[cst.BaseExpression] | None = _extract_tensor_dataset_inputs(node)
  assert res is not None
  assert len(res) == 1


def test_extract_tensor_dataset_inputs_other() -> None:
  """Verify `Call` handles extract tensor dataset inputs other as expected."""
  node: cst.Call = cst.Call(func=cst.Name("OtherDataset"), args=[cst.Arg(cst.Name("x"))])
  res: list[cst.BaseExpression] | None = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_extract_tensor_dataset_inputs_func_is_call() -> None:
  """Test extract_tensor_dataset_inputs when func is neither Name nor Attribute."""
  node: cst.Call = cst.Call(func=cst.Call(func=cst.Name("get_dataset"), args=[]), args=[cst.Arg(cst.Name("x"))])
  res = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_extract_tensor_dataset_inputs_not_call() -> None:
  """Verify `Name` handles extract tensor dataset inputs not call as expected."""
  node: cst.Name = cst.Name("x")
  res: list[cst.BaseExpression] | None = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_transform_tf_dataloader_empty() -> None:
  """Verify unmapped or empty DataLoader calls return original node safely."""
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[])
  unmapped_ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  assert transform_tf_dataloader(node, unmapped_ctx) is node

  sem = MagicMock()
  sem.resolve_variant.return_value = {"api": "tf.data.Dataset"}
  mapped_ctx = HookContext(semantics=sem, config=MagicMock())
  assert transform_tf_dataloader(node, mapped_ctx) is node


def test_transform_tf_dataloader() -> None:
  """Verify mapped DataLoader with TensorDataset and shuffle rewrites to pipeline."""
  node: cst.Call = cst.Call(
    func=cst.Name("DataLoader"),
    args=[
      cst.Arg(value=cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])),
      cst.Arg(keyword=cst.Name("batch_size"), value=cst.Integer("64")),
      cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True")),
    ],
  )
  sem = MagicMock()
  sem.resolve_variant.return_value = {"api": "tf.data.Dataset"}
  ctx: HookContext = HookContext(semantics=sem, config=MagicMock())
  res: cst.CSTNode = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "prefetch"


def test_transform_tf_dataloader_single() -> None:
  """Verify mapped DataLoader with positional and keyword dataset arguments."""
  sem = MagicMock()
  sem.resolve_variant.return_value = {"api": "tf.data.Dataset"}
  ctx: HookContext = HookContext(semantics=sem, config=MagicMock())

  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("dataset"))])
  res: cst.CSTNode = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "prefetch"

  kw_node = cst.Call(
    func=cst.Name("DataLoader"),
    args=[cst.Arg(keyword=cst.Name("dataset"), value=cst.Name("dataset"))],
  )
  sem_bare = MagicMock()
  sem_bare.resolve_variant.return_value = {"api": "Dataset"}
  bare_ctx = HookContext(semantics=sem_bare, config=MagicMock())
  res_kw = transform_tf_dataloader(kw_node, bare_ctx)
  assert isinstance(res_kw, cst.Call)
