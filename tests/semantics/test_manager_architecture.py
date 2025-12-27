"""
Tests for SemanticsManager Architecture Compliance.
Updated to verify extraction of definitions and imports from Adapters.

Fixes applied:
- Patched `ml_switcheroo.semantics.file_loader.resolve_semantics_dir` instead of `paths`
  to ensure the file loader (which consumes it) receives the mock.
- Patched individual split operation dictionaries (MATH_OPS, NEURAL_OPS, EXTRAS_OPS)
  in SemanticsManager to reflect the refactored initialization logic.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.frameworks.base import StandardMap


@pytest.fixture
def empty_directories(tmp_path):
  """Creates temporary directories with no JSON files."""
  (tmp_path / "semantics").mkdir()
  (tmp_path / "snapshots").mkdir()
  return tmp_path


def test_definition_hydration_from_adapter(empty_directories):
  """
  Scenario: A new custom adapter is registered with 'definitions'.
  Expectation: Manager merges these definitions into self.data.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  # 1. Register a dynamic adapter
  @register_framework("code_def_fw")
  class DefinesAdapter:
    import_alias = ("code_def_fw", "cd")

    @property
    def definitions(self):
      return {"DynamicOp": StandardMap(api="cd.dynamic.op", args={"x": "val"})}

    # Required stubs
    def convert(self, x):
      return x

  # 2. Initialize Manager
  # Patch file_loader specifically as it imports the resolve function
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
      mgr = SemanticsManager()

  # 3. Verify Definition merged
  assert "DynamicOp" in mgr.data
  variants = mgr.data["DynamicOp"]["variants"]
  assert "code_def_fw" in variants
  assert variants["code_def_fw"]["api"] == "cd.dynamic.op"
  assert variants["code_def_fw"]["args"] == {"x": "val"}


def test_import_namespace_hydration(empty_directories):
  """
  Scenario: Adapter provides 'import_namespaces'.
  Expectation: Manager merges these into internal store.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  @register_framework("import_fw")
  class ImportAdapter:
    @property
    def import_namespaces(self):
      # Provide legacy dict format
      return {"standard.lib": {"root": "import_fw", "sub": "lib", "alias": "il"}}

    def convert(self, x):
      return x

  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
      mgr = SemanticsManager()

  # Verify Import Data via new Public access or internal check
  # Check if 'standard.lib' is registered as a source
  assert "standard.lib" in mgr._source_registry

  # Check if 'import_fw' is registered as a provider for that tier
  # Legacy dicts default to EXTRAS in the new logic
  from ml_switcheroo.enums import SemanticTier

  tier = SemanticTier.EXTRAS

  assert "import_fw" in mgr._providers
  config = mgr._providers["import_fw"].get(tier)
  assert config is not None
  assert config["alias"] == "il"


def test_clean_slate_if_registry_empty(empty_directories):
  """
  Scenario: No JSONs and No Adapters.
  Expectation: Manager is populated ONLY with defaults injected via patch.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  # Patch file_loader consumers
  with patch("ml_switcheroo.semantics.file_loader.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.file_loader.resolve_snapshots_dir", return_value=snap):
      # Mock the registry loader to return nothing
      with patch("ml_switcheroo.semantics.registry_loader.available_frameworks", return_value=[]):
        # Patch the split internal ops dictionaries in manager module.
        with (
          patch("ml_switcheroo.semantics.manager.MATH_OPS", {"BuiltinOp": {}}),
          patch("ml_switcheroo.semantics.manager.NEURAL_OPS", {}),
          patch("ml_switcheroo.semantics.manager.EXTRAS_OPS", {}),
        ):
          mgr = SemanticsManager()
          mgr._reverse_index = {}
          # Should contain exactly the mocked internal ops
          assert len(mgr.data) == 1
          # Check key existence, don't re-assert len
          assert "BuiltinOp" in mgr.data
