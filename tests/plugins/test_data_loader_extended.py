"""Test suite for the Data Loader Extended module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.data_loader import transform_dataloader
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  return typing.cast(str, rewriter.convert(tree).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["convert_dataloader"] = transform_dataloader
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  dl_def: dict[str, typing.Any] = {
    "variants": {
      "torch": {"api": "torch.utils.data.DataLoader", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: typing.cast(
    typing.Optional[dict[str, typing.Any]], dl_def["variants"].get(fw)
  )
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_shim_arguments_passing(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of shim arguments passing."""
  code: str = "dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)"
  res: str = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds" in res
  clean: str = res.replace(" ", "")
  assert "batch_size=32" in clean
  assert "num_workers=4" in clean
  assert "pin_memory=True" in clean


def test_collate_fn_passing(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of collate fn passing."""
  code: str = "dl = DataLoader(ds, collate_fn=my_collate)"
  res: str = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds" in res
  clean: str = res.replace(" ", "")
  assert "collate_fn=my_collate" in clean


def test_positional_preservation(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of positional preservation."""
  code: str = "dl = DataLoader(ds, 64, shuffle=True)"
  res: str = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds," in res.replace(" ", "")
  assert ", 64," in res or ",64," in res.replace(" ", "")
  assert "shuffle=True" in res


def test_shim_code_injection_check(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of shim code injection check."""
  code: str = "def main(): dl = DataLoader(ds)"
  res: str = rewrite_code(rewriter, code)
  assert "class GenericDataLoader" in res
  assert "num_workers=0" in res
  assert "pin_memory=False" in res
  assert "collate_fn=None" in res
  assert "self.num_workers = num_workers" in res
  assert "self.collate_fn = collate_fn" in res
