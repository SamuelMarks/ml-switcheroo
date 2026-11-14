"""Test suite for the Data Loader module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.base import register_framework
from ml_switcheroo.plugins.data_loader import transform_dataloader
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Create a configured `rewriter_factory` fixture for data loader tests.

  Returns:
      Configured `rewriter_factory` fixture value (typing.Callable[[str], PivotRewriter]).
  """
  hooks.register_hook("convert_dataloader")(transform_dataloader)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  dl_def = {
    "variants": {
      "torch": {"api": "TorchShim", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
      "custom": {"api": "CustomShim", "requires_plugin": "convert_dataloader"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  @register_framework("custom")
  class CustomAdapter:
    """Sentinel `CustomAdapter` fixture class used in data loader tests."""

  def create(target: str) -> PivotRewriter:
    """Execute `RuntimeConfig` to support create in data loader tests.

    Args:
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `PivotRewriter` from create.
    """
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_blind_execution(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `GenericDataLoader` for blind execution.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw: PivotRewriter = rewriter_factory("torch")
  code: str = "\ndef load_data():\n    loader = torch.utils.data.DataLoader(dataset)\n"
  res: str = rewrite_code(rw, code)
  assert "GenericDataLoader" in res
  assert "class GenericDataLoader" in res


def test_jax_shim_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `class GenericDataLoader` for JAX shim injection.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef load_data():\n    loader = torch.utils.data.DataLoader(dataset, batch_size=32)\n"
  res: str = rewrite_code(rw, code)
  assert "class GenericDataLoader" in res
  assert "def __iter__(self):" in res
  clean_res: str = res.replace(" = ", "=")
  assert "GenericDataLoader(dataset" in clean_res
  assert "batch_size=32" in clean_res


def test_dataloader_arg_extraction(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `GenericDataLoader(my_ds` for dataloader arg extraction.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef train():\n    dl = DataLoader(my_ds, batch_size=64, shuffle=True)\n"
  res: str = rewrite_code(rw, code)
  clean: str = res.replace(" ", "")
  assert "GenericDataLoader(my_ds" in clean
  assert "batch_size=64" in clean
  assert "shuffle=True" in clean


def test_dataloader_idempotent_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` returns `1` for dataloader idempotent injection.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "\ndef run():\n    dl1 = DataLoader(d1)\n    dl2 = DataLoader(d2)\n"
  res: str = rewrite_code(rw, code)
  assert res.count("class GenericDataLoader") == 1
  assert res.count("GenericDataLoader(") == 2


def test_custom_target_execution(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `GenericDataLoader` for custom target execution.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw: PivotRewriter = rewriter_factory("custom")
  code: str = "dl = DataLoader(ds)"
  res: str = rewrite_code(rw, code)
  assert "GenericDataLoader" in res


def test_dataloader_workers_pin_memory_collate_and_positional_args(
  rewriter_factory: typing.Callable[[str], PivotRewriter],
) -> None:
  """Verify `GenericDataLoader` preserves `num_workers`, `pin_memory`, `collate_fn`, and positional args.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rw = rewriter_factory("jax")
  res_kw = rewrite_code(rw, "dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True, collate_fn=my_col)")
  clean_kw = res_kw.replace(" ", "")
  assert "GenericDataLoader(ds" in clean_kw
  assert "num_workers=4" in clean_kw
  assert "pin_memory=True" in clean_kw
  assert "collate_fn=my_col" in clean_kw
  assert "self.num_workers=num_workers" in clean_kw
  assert "self.collate_fn=collate_fn" in clean_kw

  res_pos = rewrite_code(rw, "dl = DataLoader(ds, 64, shuffle=True)")
  clean_pos = res_pos.replace(" ", "")
  assert "GenericDataLoader(ds,64,shuffle=True)" in clean_pos
