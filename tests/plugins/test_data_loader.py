"""
Tests for Data Loader Plugin.

Verifies:
1. Torch -> Torch passthrough.
2. Torch -> JAX rewrite (Injection of Shim class + Call rewrite).
3. Argument mapping (positional/keyword preservation).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.data_loader import transform_dataloader


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  # Register hook manually
  hooks._HOOKS["convert_dataloader"] = transform_dataloader
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()
  # Mock definition
  dl_def = {
    "variants": {
      "torch": {"api": "torch.utils.data.DataLoader", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
    }
  }

  # Setup Lookup
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True

  def create(target):
    # We assume source is always torch for these tests
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_torch_passthrough(rewriter_factory):
  """Verify Torch -> Torch preserves the original API."""
  rw = rewriter_factory("torch")
  code = "loader = torch.utils.data.DataLoader(dataset, batch_size=32)"
  res = rewrite_code(rw, code)

  assert "torch.utils.data.DataLoader" in res
  assert "GenericDataLoader" not in res
  assert "class GenericDataLoader" not in res  # No preamble injection


def test_jax_shim_injection(rewriter_factory):
  """Verify Torch -> JAX injects the class and rewrites the call inside a function scope."""
  rw = rewriter_factory("jax")
  # Wrap in function to ensure preamble injection works (FuncStructureMixin logic)
  code = """
def load_data():
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
"""
  res = rewrite_code(rw, code)

  # 1. Preamble Check
  assert "class GenericDataLoader" in res
  assert "def __iter__(self):" in res

  # 2. Call Check
  # Normalize spaces for robust check
  clean_res = res.replace(" = ", "=")
  # Check for the correct call structure
  assert "GenericDataLoader(dataset" in clean_res
  assert "batch_size=32" in clean_res
  assert "loader=" in clean_res  # Check assignment happened


def test_dataloader_arg_extraction(rewriter_factory):
  """Verify positional args are correctly mapped."""
  rw = rewriter_factory("jax")
  # Wrap in function
  code = """
def train():
    dl = DataLoader(my_ds, batch_size=64, shuffle=True)
"""
  res = rewrite_code(rw, code)
  clean = res.replace(" ", "")

  assert "GenericDataLoader(my_ds" in clean
  assert "batch_size=64" in clean
  assert "shuffle=True" in clean


def test_dataloader_only_injects_once(rewriter_factory):
  """Verify shim class isn't duplicated if multiple calls exist."""
  rw = rewriter_factory("jax")
  code = """
def run():
    dl1 = DataLoader(d1)
    dl2 = DataLoader(d2)
"""
  res = rewrite_code(rw, code)

  # Should contain class definition only once
  assert res.count("class GenericDataLoader") == 1
  assert res.count("GenericDataLoader(") == 2
