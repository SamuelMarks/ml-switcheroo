"""
Tests for Data Loader Plugin (Decoupled Logic).

Verifies:
1. Plugin execution is driven by Metadata, not Target string hardcoding.
2. Correct Shim injection.
3. Argument preservation logic.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.data_loader import transform_dataloader
from ml_switcheroo.frameworks.base import register_framework


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

  # Mock Definition: multiple frameworks request the plugin.
  # This setup proves that the plugin runs blindly if requested,
  # breaking the coupling where "torch" was formerly hardcoded to skip.
  dl_def = {
    "variants": {
      "torch": {"api": "TorchShim", "requires_plugin": "convert_dataloader"},
      "jax": {"api": "GenericDataLoader", "requires_plugin": "convert_dataloader"},
      "custom": {"api": "CustomShim", "requires_plugin": "convert_dataloader"},
    }
  }

  # Setup Lookup
  mgr.get_definition.side_effect = lambda n: ("DataLoader", dl_def) if "DataLoader" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: dl_def["variants"].get(fw)
  mgr.is_verified.return_value = True
  # Fix: Ensure fallback defaults for trait lookups
  mgr.get_framework_config.return_value = {}

  # --- FIX: Register dummy framework 'custom' to allow RuntimeConfig validation to pass ---
  # We define it locally so it affects only tests consuming this fixture.
  @register_framework("custom")
  class CustomAdapter:
    pass

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_blind_execution(rewriter_factory):
  """
  Verify that if 'torch' target explicitly requests the plugin (via mock semantics),
  the plugin EXECUTES and injects the shim.
  (Previously, this would return original node because of hardcoded 'if target==torch: return')
  """
  rw = rewriter_factory("torch")
  # Wrap in function to ensure preamble injection works (FuncStructureMixin logic)
  code = """ 
def load_data(): 
    loader = torch.utils.data.DataLoader(dataset) 
"""
  res = rewrite_code(rw, code)

  # 1. Verify Transformation happened
  assert "GenericDataLoader" in res
  # 2. Verify Preamble Injection
  assert "class GenericDataLoader" in res


def test_jax_shim_injection(rewriter_factory):
  """Verify Standard JAX replacement."""
  rw = rewriter_factory("jax")
  code = """ 
def load_data(): 
    loader = torch.utils.data.DataLoader(dataset, batch_size=32) 
"""
  res = rewrite_code(rw, code)

  # 1. Preamble Check
  assert "class GenericDataLoader" in res
  assert "def __iter__(self):" in res

  # 2. Call Check
  clean_res = res.replace(" = ", "=")
  assert "GenericDataLoader(dataset" in clean_res
  assert "batch_size=32" in clean_res


def test_dataloader_arg_extraction(rewriter_factory):
  """Verify positional args are correctly mapped."""
  rw = rewriter_factory("jax")
  code = """ 
def train(): 
    dl = DataLoader(my_ds, batch_size=64, shuffle=True) 
"""
  res = rewrite_code(rw, code)
  clean = res.replace(" ", "")

  assert "GenericDataLoader(my_ds" in clean
  assert "batch_size=64" in clean
  assert "shuffle=True" in clean


def test_dataloader_idempotent_injection(rewriter_factory):
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


def test_custom_target_execution(rewriter_factory):
  """Verify unknown 'custom' target works if wired."""
  rw = rewriter_factory("custom")
  code = "dl = DataLoader(ds)"
  res = rewrite_code(rw, code)
  assert "GenericDataLoader" in res
