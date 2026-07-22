"""Test suite for the Tf Data Loader module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.tf_data_loader import transform_tf_dataloader, _get_arg_by_name, _extract_tensor_dataset_inputs
from ml_switcheroo.core.hooks import HookContext


def test_get_arg_by_name():
  """Gets argument by name."""
  arg = cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True"))
  assert _get_arg_by_name([arg], "shuffle") is arg
  assert _get_arg_by_name([arg], "missing") is None


def test_extract_tensor_dataset_inputs():
  """Extracts tensor dataset inputs."""
  node = cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])
  res = _extract_tensor_dataset_inputs(node)
  assert len(res) == 2


def test_extract_tensor_dataset_inputs_other():
  """Extracts tensor dataset inputs other."""
  node = cst.Call(func=cst.Name("OtherDataset"), args=[cst.Arg(cst.Name("x"))])
  res = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_extract_tensor_dataset_inputs_not_call():
  """Extracts tensor dataset inputs not call."""
  node = cst.Name("x")
  res = _extract_tensor_dataset_inputs(node)
  assert res is None


def test_transform_tf_dataloader_empty():
  """Transforms tf dataloader empty."""
  node = cst.Call(func=cst.Name("DataLoader"), args=[])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = transform_tf_dataloader(node, ctx)
  assert res is node


def test_transform_tf_dataloader():
  """Transforms tf dataloader."""
  node = cst.Call(
    func=cst.Name("DataLoader"),
    args=[
      cst.Arg(value=cst.Call(func=cst.Name("TensorDataset"), args=[cst.Arg(cst.Name("x")), cst.Arg(cst.Name("y"))])),
      cst.Arg(keyword=cst.Name("batch_size"), value=cst.Integer("64")),
      cst.Arg(keyword=cst.Name("shuffle"), value=cst.Name("True")),
    ],
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert res.func.attr.value == "prefetch"


def test_transform_tf_dataloader_single():
  """Transforms tf dataloader single."""
  node = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("dataset"))])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  res = transform_tf_dataloader(node, ctx)
  assert isinstance(res, cst.Call)
  assert res.func.attr.value == "prefetch"
