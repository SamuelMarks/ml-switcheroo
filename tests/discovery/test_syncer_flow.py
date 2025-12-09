"""
Tests for FrameworkSyncer Import and Linking Flow.

Verifies that:
1. The Syncer resolves modules dynamicalls via `get_adapter`.
2. Modules are imported via importlib.
3. Matching functions are linked into the semantics dictionary.
4. Fallback behavior (if no adapter exists) works safely.
"""

import types
from unittest.mock import patch, MagicMock
import pytest

from ml_switcheroo.discovery.syncer import FrameworkSyncer


def mock_module(name: str, functions: dict) -> types.ModuleType:
  """Helper to create a dummy module with specific callable attributes."""
  mod = types.ModuleType(name)
  for func_name, func_obj in functions.items():
    setattr(mod, func_name, func_obj)
  return mod


@pytest.fixture
def syncer():
  """Returns a fresh instance of FrameworkSyncer."""
  return FrameworkSyncer()


def test_linking_via_adapter_registry(syncer):
  """
  Scenario: User syncs 'custom_fw'.
  Action:
      1. Queries registry for 'custom_fw'.
      2. Configures search paths: ['custom', 'custom.math'].
  Result:
      - `custom.math.abs` matches `abs`.
      - `custom.add` is ignored if not in search paths (but here custom is root).
  """
  # 1. Setup Data
  semantics = {
    "abs": {"std_args": ["x"], "variants": {}},
    "unknown_op": {"std_args": ["x"], "variants": {}},
  }

  # 2. Mock Adapter
  mock_adapter = MagicMock()
  mock_adapter.search_modules = ["custom_fw.math", "custom_fw"]

  # 3. Mock Modules
  def c_abs(_x):
    pass

  mock_mod_math = mock_module("custom_fw.math", {"abs": c_abs})
  mock_mod_root = mock_module("custom_fw", {})

  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=mock_adapter) as mock_get:
    with patch("importlib.import_module") as mock_import:
      # Setup import logic
      def import_side_effect(name):
        if name == "custom_fw.math":
          return mock_mod_math
        if name == "custom_fw":
          return mock_mod_root
        raise ImportError(name)

      mock_import.side_effect = import_side_effect

      # 4. Run Sync
      syncer.sync(semantics, "custom_fw")

      # Verify adapter was queried
      mock_get.assert_called_with("custom_fw")

  # 5. Verify Results
  assert "custom_fw" in semantics["abs"]["variants"]
  # Path should come from .math because it was first in search_modules
  assert semantics["abs"]["variants"]["custom_fw"]["api"] == "custom_fw.math.abs"


def test_fallback_no_adapter(syncer):
  """
  Scenario: User syncs 'legacy_lib' which has no adapter defined.
  Action: Should fallback to searching default module 'legacy_lib'.
  """
  semantics = {"relu": {"std_args": ["x"], "variants": {}}}

  def l_relu(_x):
    pass

  mock_lib = mock_module("legacy_lib", {"relu": l_relu})

  # Return None for adapter
  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=None):
    with patch("importlib.import_module", return_value=mock_lib) as mock_import:
      syncer.sync(semantics, "legacy_lib")

      # Assert tried to import the framework name itself
      mock_import.assert_called_with("legacy_lib")

  assert semantics["relu"]["variants"]["legacy_lib"]["api"] == "legacy_lib.relu"


def test_adapter_missing_search_modules_property(syncer):
  """
  Scenario: Adapter exists but relies on older interface (no search_modules).
  Action: Handle gracefully (fallback to default name).
  """
  semantics = {"op": {"std_args": ["x"], "variants": {}}}

  # Adapter object exists but is empty/legacy
  mock_adapter = MagicMock(spec=[])

  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=mock_adapter):
    with patch("importlib.import_module", side_effect=ImportError):
      # Just ensure it doesn't crash on attribute access
      syncer.sync(semantics, "partial_fw")


def test_sync_skips_incompatible_signatures(syncer):
  """
  Scenario: Framework has a function with match name but wrong signature.
  Result: It is NOT linked.
  """
  semantics = {"matmul": {"std_args": ["x", "y"], "variants": {}}}

  # Incompatible: matmul taking only 1 arg vs standard 2
  def bad_matmul(_x):
    pass

  mock_mod = mock_module("my_lib", {"matmul": bad_matmul})

  # Mock adapter to force search path
  adapter = MagicMock()
  adapter.search_modules = ["my_lib"]

  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mock_mod):
      syncer.sync(semantics, "my_lib")

  # Should NOT have linked
  assert "my_lib" not in semantics["matmul"]["variants"]


def test_fails_gracefully_on_import_error(syncer):
  """
  Scenario: User asks to sync a framework that is not installed.
  Result: Error message logged, no crash.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {}}}

  # Adapter says search "ghost", importlib raises Error
  adapter = MagicMock()
  adapter.search_modules = ["ghost"]

  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=adapter):
    with patch("importlib.import_module", side_effect=ImportError("Not installed")):
      syncer.sync(semantics, "ghost")

  assert "ghost" not in semantics["abs"]["variants"]


def test_sync_preserves_existing_mappings(syncer):
  """
  Scenario: 'variants' already has an entry for this framework (manual override).
  Result: The existing entry is preserved, scan is skipped for that op.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "manual.override.abs"}}}}

  # Even if we provide a "real" scan result
  def real_abs(_x):
    pass

  mock_torch = mock_module("torch", {"abs": real_abs})

  adapter = MagicMock()
  adapter.search_modules = ["torch"]

  with patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=adapter):
    with patch("importlib.import_module", return_value=mock_torch):
      syncer.sync(semantics, "torch")

  # Should remain the manual override
  assert semantics["abs"]["variants"]["torch"]["api"] == "manual.override.abs"
