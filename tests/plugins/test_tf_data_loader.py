"""Test suite for the Tf Data Loader module."""

import libcst as cst
from typing import List, Optional, Union
from unittest.mock import MagicMock
from ml_switcheroo.plugins.tf_data_loader import transform_tf_dataloader, _get_arg_by_name, _extract_tensor_dataset_inputs
from ml_switcheroo.core.hooks import HookContext


def test_get_arg_by_name() -> None:
  """Gets argument by name."""
  arg: cst.Arg = cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True"))
  assert _get_arg_by_name([arg], "shuffle") is arg
  assert _get_arg_by_name([arg], "missing") is None


def test_extract_tensor_dataset_inputs() -> None:
  """Extracts tensor dataset inputs."""
  node: cst.Call = cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])
  res: Optional[List[cst.Arg]] = _extract_tensor_dataset_inputs(node)
  assert res is not None
  assert len(res) == 2


def test_extract_tensor_dataset_inputs_attribute() -> None:
  """Extracts tensor dataset inputs when func is an attribute."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("data"), attr=cst.Name("TensorDataset")), args=[cst.Arg(cst.Name("x"))]
  )
  res: Optional[List[cst.Arg]] = _extract_tensor_dataset_inputs(node)
  assert res is not None
  assert len(res) == 1


def test_extract_tensor_dataset_inputs_other() -> None:
  """Extracts tensor dataset inputs other."""
  node: cst.Call = cst.Call(func=cst.Name("OtherDataset"), args=[cst.Arg(cst.Name("x"))])
  res: Optional[List[cst.Arg]] = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_extract_tensor_dataset_inputs_not_call() -> None:
  """Extracts tensor dataset inputs not call."""
  node: cst.Name = cst.Name("x")
  res: Optional[List[cst.Arg]] = _extract_tensor_dataset_inputs(node)  # type: ignore
  assert res is None


def test_transform_tf_dataloader_empty() -> None:
  """Transforms tf dataloader empty."""
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.Call] = transform_tf_dataloader(node, ctx)
  assert res is node


def test_transform_tf_dataloader() -> None:
  """Transforms tf dataloader."""
  node: cst.Call = cst.Call(
    func=cst.Name("DataLoader"),
    args=[
      cst.Arg(value=cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])),
      cst.Arg(keyword=cst.Name("batch_size"), value=cst.Integer("64")),
      cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True")),
    ],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.Call] = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "prefetch"


def test_transform_tf_dataloader_single() -> None:
  """Transforms tf dataloader single."""
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("dataset"))])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: Union[cst.CSTNode, cst.Call] = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "prefetch"
