"""Test suite for plugin data loader."""

from pathlib import Path
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.data_loader import get_shim_code, transform_dataloader


def test_get_shim_code() -> None:
  """Verify `get_shim_code` emits `class GenericDataLoader:` across plugin data loader inputs."""
  code: str = get_shim_code()
  assert "class GenericDataLoader:" in code


def test_transform_dataloader_injects_once() -> None:
  """Verify `Call` handles transform dataloader injects once as expected."""
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=[cst.Arg(value=cst.Name("ds"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result: cst.CSTNode = transform_dataloader(node, ctx)
  assert ctx.inject_preamble.called
  assert ctx.metadata["dataloader_shim_injected"] is True
  assert isinstance(result, cst.Call)
  assert cst.ensure_type(result.func, cst.Name).value == "GenericDataLoader"

  # second call
  ctx.inject_preamble.reset_mock()
  transform_dataloader(node, ctx)
  assert not ctx.inject_preamble.called


def test_transform_dataloader_args() -> None:
  """Verify `Call` returns `3` for transform dataloader args."""
  args: list[cst.Arg] = [
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
  assert cst.ensure_type(result.args[0].value, cst.Name).value == "ds"
  assert cst.ensure_type(result.args[1].value, cst.Name).value == "pos_arg"
  assert result.args[2].keyword is not None
  assert result.args[2].keyword.value == "batch_size"


def test_transform_dataloader_no_pos_args() -> None:
  """Verify `Call` returns `1` for transform dataloader no pos args."""
  args: list[cst.Arg] = [
    cst.Arg(keyword=cst.Name("dataset"), value=cst.Name("ds")),
  ]
  node: cst.Call = cst.Call(func=cst.Name("DataLoader"), args=args)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}

  result: cst.CSTNode = transform_dataloader(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 1
  assert result.args[0].keyword is not None
  assert result.args[0].keyword.value == "dataset"


def test_generic_data_loader_shim_exec(tmp_path: "Path") -> None:
  """Verify `get_shim_code` returns `4` for generic data loader shim exec.

  Args:
      tmp_path: Pytest temporary path fixture.
  """
  import importlib.util
  from pathlib import Path

  assert isinstance(tmp_path, Path)
  code: str = get_shim_code()
  shim_file = tmp_path / "_dataloader_shim.py"
  shim_file.write_text(code, encoding="utf-8")
  spec = importlib.util.spec_from_file_location("_dataloader_shim", shim_file)
  assert spec is not None
  assert spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  GenericDataLoader: type = mod.GenericDataLoader

  # Test DataLoader iteration behavior
  dataset: list[int] = list(range(10))
  loader = GenericDataLoader(dataset, batch_size=3, drop_last=False)
  assert len(loader) == 4
  batches: list[list[int]] = list(loader)
  assert len(batches) == 4

  loader_drop = GenericDataLoader(dataset, batch_size=3, drop_last=True)
  assert len(loader_drop) == 3
  batches_drop: list[list[int]] = list(loader_drop)
  assert len(batches_drop) == 3

  # Custom collate
  loader_collate = GenericDataLoader(dataset, batch_size=3, collate_fn=lambda x: sum(x))
  assert next(iter(loader_collate)) == 3  # 0+1+2
