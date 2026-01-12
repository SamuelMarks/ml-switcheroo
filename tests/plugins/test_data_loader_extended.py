"""
Tests for Extended Data Loader Support (Shim Hardening).

Verifies:
1. `num_workers` passing.
2. `pin_memory` passing.
3. `collate_fn` passing.
4. Correct Shim injection.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.data_loader import transform_dataloader


def rewrite_code(rewriter, code: str) -> str:
  """Executes the rewriter pipeline."""
  tree = cst.parse_module(code)
  # Use convert() via shim
  return rewriter.convert(tree).code


@pytest.fixture
def rewriter():
  # Register hook manually
  hooks._HOOKS["convert_dataloader"] = transform_dataloader
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()
  # Mock definition
  dl_def = {
    "variants": {
      "torch": {
        "api": "torch.utils.data.DataLoader",
        "requires_plugin": "convert_dataloader",
      },
      "jax": {
        "api": "GenericDataLoader",
        "requires_plugin": "convert_dataloader",
      },
    }
  }

  # Setup Lookup
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_shim_arguments_passing(rewriter):
  """
  Scenario: Torch initialization with perf args.
  Input: DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
  Output: GenericDataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)
  """
  code = "dl = DataLoader(ds, batch_size=32, num_workers=4, pin_memory=True)"
  res = rewrite_code(rewriter, code)

  # Check Class Swap
  assert "GenericDataLoader(ds" in res

  # Check Argument Preservation
  clean = res.replace(" ", "")
  assert "batch_size=32" in clean
  assert "num_workers=4" in clean
  assert "pin_memory=True" in clean


def test_collate_fn_passing(rewriter):
  """
  Scenario: Torch custom collate function.
  Input: DataLoader(ds, collate_fn=my_collate)
  Output: GenericDataLoader(ds, collate_fn=my_collate)
  """
  code = "dl = DataLoader(ds, collate_fn=my_collate)"
  res = rewrite_code(rewriter, code)

  assert "GenericDataLoader(ds" in res
  clean = res.replace(" ", "")
  assert "collate_fn=my_collate" in clean


def test_positional_preservation(rewriter):
  """
  Scenario: Mixed positional/keyword.
  Input: DataLoader(ds, 64, shuffle=True)
  """
  code = "dl = DataLoader(ds, 64, shuffle=True)"
  res = rewrite_code(rewriter, code)

  assert "GenericDataLoader(ds," in res.replace(" ", "")
  # "64" should be preserved as positional arg
  assert ", 64," in res or ",64," in res.replace(" ", "")
  assert "shuffle=True" in res


def test_shim_code_injection_check(rewriter):
  """
  Verify the updated Shim code contains the new arguments in __init__.
  """
  # Wrap in function to trigger preamble injection
  code = "def main(): dl = DataLoader(ds)"
  res = rewrite_code(rewriter, code)

  # 1. Check for the shim class definition
  assert "class GenericDataLoader" in res

  # 2. Check that the definition accepts the parameters.
  # Note: `__init__` might be split across multiple lines, so we check
  # for specific substrings in the generated code that correspond to the signature defaults.
  # The Shim defines: num_workers=0, pin_memory=False, collate_fn=None

  assert "num_workers=0" in res
  assert "pin_memory=False" in res
  assert "collate_fn=None" in res

  # 3. Check that these variables are assigned to self (stored)
  assert "self.num_workers = num_workers" in res
  assert "self.collate_fn = collate_fn" in res
