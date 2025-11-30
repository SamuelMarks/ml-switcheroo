"""
Tests for Tier C (Extras) Semantics Loading.

Verifies that the `k_framework_extras.json` file is structurally valid,
that special sections like `__imports__` and `__frameworks__` are parsed,
and that the SemanticsManager correctly exposes 'null' variants to trigger
the Escape Hatch in the Rewriter.
"""

import json
import pytest
from unittest.mock import patch
import libcst as cst

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig

# --- 1. Integration Tests (Real Data from Disk) ---


@pytest.fixture
def real_manager():
  """Loads the real JSON files from disk (standard package capability)."""
  return SemanticsManager()


def test_extras_are_loaded(real_manager):
  """
  Verify keys from k_framework_extras.json are present in the knowledge graph.
  """
  # Key check: 'DataLoader' is a canonical example of an Extra (Framework Utility)
  details = real_manager.get_definition("torch.utils.data.DataLoader")

  if not details:
    pytest.skip("Standard definition for DataLoader missing from semantics.")

  op_name, data = details
  assert op_name == "DataLoader"
  assert data["variants"]["torch"]["api"] == "torch.utils.data.DataLoader"


def test_null_variant_trigger(real_manager):
  """
  Verify that JAX variant explicitly set to null is preserved as None.
  This dictates Escape Hatch behavior in the Rewriter.
  """
  details = real_manager.get_definition("torch.utils.data.DataLoader")
  if not details:
    pytest.skip("Standard definition for DataLoader missing.")

  _, data = details
  jax_variant = data["variants"].get("jax")

  # JSON 'null' becomes Python 'None'
  assert jax_variant is None


def test_plugin_flag_parsing(real_manager):
  """
  Verify 'no_grad' correctly flags a required plugin.
  """
  details = real_manager.get_definition("torch.no_grad")
  if not details:
    pytest.skip("Standard definition for no_grad missing.")

  _, data = details
  jax_variant = data["variants"]["jax"]
  assert jax_variant is not None
  assert jax_variant["requires_plugin"] == "context_to_function_wrap"


def test_manual_seed_mapping(real_manager):
  """
  Verify 'manual_seed' maps to 'numpy.random.seed'.
  """
  details = real_manager.get_definition("torch.manual_seed")
  if not details:
    pytest.skip("Standard definition for manual_seed missing.")

  _, data = details
  jax_variant = data["variants"]["jax"]
  assert jax_variant["api"] == "numpy.random.seed"


# --- 2. Isolated Logic Tests (Mock Data) ---


@pytest.fixture
def mock_extras_content():
  """Defines a comprehensive Tier C JSON structure."""
  return {
    "__imports__": {"torch.optim": {"variants": {"jax": {"root": "optax", "sub": None, "alias": None}}}},
    "__frameworks__": {"jax": {"stateful_call": {"method": "apply"}}},
    "CustomLoader": {
      "std_args": ["dataset"],
      "variants": {
        "torch": {"api": "torch.utils.data.DataLoader"},
        # Explicit Null maps to Python None
        "jax": None,
      },
    },
    "MagicContext": {
      "std_args": [],
      "variants": {
        "torch": {"api": "torch.magic"},
        # Plugin only (no API path)
        "jax": {"requires_plugin": "magic_shim"},
      },
    },
  }


@pytest.fixture
def isolated_manager(tmp_path, mock_extras_content):
  """
  Creates a SemanticsManager pointing to a temp dir with a crafted
  k_framework_extras.json.
  """
  # 1. Create the file structure
  extras_path = tmp_path / "k_framework_extras.json"
  extras_path.write_text(json.dumps(mock_extras_content), encoding="utf-8")

  # 2. Patch resolution to force loading ONLY this file (effectively)
  # (Other files won't exist in tmp_path)
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    mgr = SemanticsManager()
    return mgr


def test_load_structure_from_extras(isolated_manager):
  """Verify operations are loaded from Tier C."""
  # Check CustomLoader
  details = isolated_manager.get_definition("torch.utils.data.DataLoader")
  assert details is not None
  op_name, data = details
  assert op_name == "CustomLoader"
  # Check null variant consistency
  assert data["variants"]["jax"] is None


def test_load_imports_from_extras(isolated_manager):
  """Verify __imports__ section parses correctly."""
  assert "torch.optim" in isolated_manager.import_data
  variant = isolated_manager.import_data["torch.optim"]["variants"]["jax"]
  assert variant["root"] == "optax"


def test_load_framework_config_from_extras(isolated_manager):
  """Verify __frameworks__ section parses correctly."""
  assert "jax" in isolated_manager.framework_configs
  assert isolated_manager.framework_configs["jax"]["stateful_call"]["method"] == "apply"


def test_rewriter_integration_null_variant(isolated_manager):
  """
  Verify PivotRewriter marks failure for explicit None variants loaded from Tier C.
  This confirms the "Escape Hatch" is triggered by the loaded JSON 'null'.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  rewriter = PivotRewriter(isolated_manager, config)

  # Code: y = torch.utils.data.DataLoader(x)
  # Should fail because jax variant is None in isolated_manager
  code = "y = torch.utils.data.DataLoader(x)"

  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  result_code = new_tree.code

  # Check that the logic bubbled an error via EscapeHatch mechanism
  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in result_code
  assert "No mapping defined" in result_code
  # Assert verbatim code preservation
  assert "torch.utils.data.DataLoader(x)" in result_code


def test_rewriter_integration_plugin_only(isolated_manager):
  """
  Verify PivotRewriter handles plugin-only variants (no API mapping).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  rewriter = PivotRewriter(isolated_manager, config)

  # We call 'torch.magic' which maps to 'MagicContext'.
  # Target 'jax' has "requires_plugin": "magic_shim".
  # Since we haven't registered the hook, it should report missing plugin.

  code = "res = torch.magic()"
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  result_code = new_tree.code

  # Logic flow check:
  # 1. Matches "MagicContext" -> variants[jax] has plugin req.
  # 2. rewriter calls get_hook("magic_shim") -> returns None (not registered).
  # 3. rewriter reports failure "Missing required plugin".

  assert "# <SWITCHEROO_FAILED_TO_TRANS>" in result_code
  assert "Missing required plugin" in result_code
  assert "'magic_shim'" in result_code
