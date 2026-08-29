"""Docstring."""

from typing import Dict, List
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.data_loader import get_shim_code, transform_dataloader


def test_get_shim_code() -> None:
  """Docstring."""
  code: str = get_shim_code()
  assert "class GenericDataLoader:" in code


def test_transform_dataloader_injects_once() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("ds"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result: cst.CSTNode = transform_dataloader(node, ctx)
  assert ctx.inject_preamble.called
  assert ctx.metadata["dataloader_shim_injected"] is True
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "value", None) == "GenericDataLoader"

  # second call
  ctx.inject_preamble.reset_mock()
  transform_dataloader(node, ctx)
  assert not ctx.inject_preamble.called


def test_transform_dataloader_args() -> None:
  """Docstring."""
  args: List[cst.Arg] = [
    cst.Arg(value=cst.Name("ds")),
    cst.Arg(value=cst.Name("pos_arg")),
    cst.Arg(keyword=cst.Name("batch_size"), value=cst.Integer("32")),
  ]
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=args)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result: cst.CSTNode = transform_dataloader(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 3
  assert getattr(result.args[0].value, "value", None) == "ds"
  assert getattr(result.args[1].value, "value", None) == "pos_arg"
  assert getattr(result.args[2].keyword, "value", None) == "batch_size"


def test_transform_dataloader_no_pos_args() -> None:
  """Docstring."""
  args: List[cst.Arg] = [
    cst.Arg(keyword=cst.Name("dataset"), value=cst.Name("ds")),
  ]
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=args)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result: cst.CSTNode = transform_dataloader(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 1
  assert getattr(result.args[0].keyword, "value", None) == "dataset"


def test_generic_data_loader_shim_exec() -> None:
  """Docstring."""
  code: str = get_shim_code()
  ns: Dict[str, type] = {}
  exec(code, ns)
  GenericDataLoader: type = ns["GenericDataLoader"]

  # Test DataLoader iteration behavior
  dataset: List[int] = list(range(10))
  loader = GenericDataLoader(dataset, batch_size=3, drop_last=False)
  assert len(loader) == 4
  batches: List[object] = list(loader)
  assert len(batches) == 4

  loader_drop = GenericDataLoader(dataset, batch_size=3, drop_last=True)
  assert len(loader_drop) == 3
  batches_drop: List[object] = list(loader_drop)
  assert len(batches_drop) == 3

  # Custom collate
  loader_collate = GenericDataLoader(dataset, batch_size=3, collate_fn=lambda x: sum(x))
  assert next(iter(loader_collate)) == 3  # 0+1+2
