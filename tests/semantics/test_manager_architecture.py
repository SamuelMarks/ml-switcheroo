"""
Tests for SemanticsManager Architecture Compliance.
Updated to verify extraction of definitions and imports from Adapters.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.manager import SemanticsManager, resolve_snapshots_dir
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.schema import StructuralTraits
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
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
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
  Expectation: Manager merges these into self.import_data.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  @register_framework("import_fw")
  class ImportAdapter:
    @property
    def import_namespaces(self):
      return {"standard.lib": {"root": "import_fw", "sub": "lib", "alias": "il"}}

    def convert(self, x):
      return x

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
      mgr = SemanticsManager()

  # Verify Import Data
  assert "standard.lib" in mgr.import_data
  variants = mgr.import_data["standard.lib"]["variants"]
  assert "import_fw" in variants
  assert variants["import_fw"]["alias"] == "il"


def test_clean_slate_if_registry_empty(empty_directories):
  """
  Scenario: No JSONs and No Adapters.
  Expectation: Manager is truly empty.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
      # Mock available_frameworks to return nothing
      with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
        mgr = SemanticsManager()
        mgr._reverse_index = {}
        assert len(mgr.data) == 0
