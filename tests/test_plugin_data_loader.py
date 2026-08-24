"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.data_loader import transform_dataloader, get_shim_code
from ml_switcheroo.core.hooks import HookContext


def test_get_shim_code():
  """Docstring."""
  code = get_shim_code()
  assert "class GenericDataLoader:" in code


def test_transform_dataloader_injects_once():
  """Docstring."""
  node = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("ds"))])
  ctx = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result = transform_dataloader(node, ctx)
  assert ctx.inject_preamble.called
  assert ctx.metadata["dataloader_shim_injected"] is True
  assert isinstance(result, cst.Call)
  assert result.func.value == "GenericDataLoader"

  # second call
  ctx.inject_preamble.reset_mock()
  transform_dataloader(node, ctx)
  assert not ctx.inject_preamble.called


def test_transform_dataloader_args():
  """Docstring."""
  args = [
    cst.Arg(value=cst.Name("ds")),
    cst.Arg(value=cst.Name("pos_arg")),
    cst.Arg(keyword=cst.Name("batch_size"), value=cst.Integer("32")),
  ]
  node = cst.Call(func=cst.Name("DataLoader"), args=args)
  ctx = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result = transform_dataloader(node, ctx)
  assert len(result.args) == 3
  assert result.args[0].value.value == "ds"
  assert result.args[1].value.value == "pos_arg"
  assert result.args[2].keyword.value == "batch_size"


def test_transform_dataloader_no_pos_args():
  """Docstring."""
  args = [
    cst.Arg(keyword=cst.Name("dataset"), value=cst.Name("ds")),
  ]
  node = cst.Call(func=cst.Name("DataLoader"), args=args)
  ctx = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result = transform_dataloader(node, ctx)
  assert len(result.args) == 1
  assert result.args[0].keyword.value == "dataset"


def test_generic_data_loader_shim_exec():
  """Docstring."""
  code = get_shim_code()
  ns = {}
  exec(code, ns)
  GenericDataLoader = ns["GenericDataLoader"]

  # Test DataLoader iteration behavior
  dataset = list(range(10))
  loader = GenericDataLoader(dataset, batch_size=3, drop_last=False)
  assert len(loader) == 4
  batches = list(loader)
  assert len(batches) == 4

  loader_drop = GenericDataLoader(dataset, batch_size=3, drop_last=True)
  assert len(loader_drop) == 3
  batches_drop = list(loader_drop)
  assert len(batches_drop) == 3

  # Custom collate
  loader_collate = GenericDataLoader(dataset, batch_size=3, collate_fn=lambda x: sum(x))
  assert next(iter(loader_collate)) == 3  # 0+1+2
