"""Test suite for the Data Loader Extended module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.data_loader import transform_dataloader


def rewrite_code(rewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  return rewriter.convert(tree).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["convert_dataloader"] = transform_dataloader
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  dl_def = {
    "variants": {
      "torch": {"api": "torch.utils.data.DataLoader", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_shim_arguments_passing(rewriter):
  """Verifies the behavior of shim arguments passing."""
  code = "dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)"
  res = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds" in res
  clean = res.replace(" ", "")
  assert "batch_size=32" in clean
  assert "num_workers=4" in clean
  assert "pin_memory=True" in clean


def test_collate_fn_passing(rewriter):
  """Verifies the behavior of collate fn passing."""
  code = "dl = DataLoader(ds, collate_fn=my_collate)"
  res = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds" in res
  clean = res.replace(" ", "")
  assert "collate_fn=my_collate" in clean


def test_positional_preservation(rewriter):
  """Verifies the behavior of positional preservation."""
  code = "dl = DataLoader(ds, 64, shuffle=True)"
  res = rewrite_code(rewriter, code)
  assert "GenericDataLoader(ds," in res.replace(" ", "")
  assert ", 64," in res or ",64," in res.replace(" ", "")
  assert "shuffle=True" in res


def test_shim_code_injection_check(rewriter):
  """Verifies the behavior of shim code injection check."""
  code = "def main(): dl = DataLoader(ds)"
  res = rewrite_code(rewriter, code)
  assert "class GenericDataLoader" in res
  assert "num_workers=0" in res
  assert "pin_memory=False" in res
  assert "collate_fn=None" in res
  assert "self.num_workers = num_workers" in res
  assert "self.collate_fn = collate_fn" in res
