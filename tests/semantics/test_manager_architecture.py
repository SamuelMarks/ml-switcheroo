"""
Tests for SemanticsManager Architecture Compliance.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.manager import SemanticsManager, resolve_snapshots_dir
from ml_switcheroo.testing.adapters import register_adapter
from ml_switcheroo.semantics.schema import StructuralTraits


@pytest.fixture
def empty_directories(tmp_path):
  """Creates temporary directories with no JSON files."""
  (tmp_path / "semantics").mkdir()
  (tmp_path / "snapshots").mkdir()
  return tmp_path


def test_initialization_traits_from_registry(empty_directories):
  """
  Scenario: Environment with no JSON files, but Adapters are registered.
  Expectation: Manager populates Structural Traits (Aliases, Base Classes) from Code.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
      mgr = SemanticsManager()
      mgr._reverse_index = {}

      # Should have aliases from registry (e.g. torch -> (torch, torch))
      aliases = mgr.get_framework_aliases()
      assert "torch" in aliases
      assert aliases["torch"] == ("torch", "torch")

      # Should have traits
      conf = mgr.get_framework_config("torch")
      assert "traits" in conf


def test_traits_hydration_from_adapter(empty_directories):
  """
  Scenario: A new custom adapter is registered with 'structural_traits'.
  Expectation: SemanticsManager populates framework_configs with these traits.
  """
  sem = empty_directories / "semantics"
  snap = empty_directories / "snapshots"

  # 1. Register a dynamic adapter with Structural Traits
  @register_adapter("fastnet")
  class FastNetAdapter:
    import_alias = ("fastnet", "fn")

    @property
    def structural_traits(self) -> StructuralTraits:
      return StructuralTraits(module_base="fastnet.Module", forward_method="run", requires_super_init=True)

    # Required stubs
    def convert(self, x):
      return x

  # 2. Initialize Manager (this triggers _hydrate_defaults_from_registry)
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=sem):
    with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir", return_value=snap):
      mgr = SemanticsManager()
      mgr._reverse_index = {}

  # 3. Verify Traits were loaded into config
  config = mgr.get_framework_config("fastnet")
  assert "traits" in config
  traits = config["traits"]

  assert traits["module_base"] == "fastnet.Module"
  assert traits["forward_method"] == "run"
  assert traits["requires_super_init"] is True


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

        assert len(mgr.test_templates) == 0
        assert len(mgr.framework_configs) == 0

        # Assert Data is empty.
        assert len(mgr.data) == 0


def test_test_alpha_add_removed():
  """
  Safeguard against regressions of hardcoded operations.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/non/existent/path")
    with patch.object(Path, "exists", return_value=False):
      # Also mock snapshots dir to avoid IO error
      with patch("ml_switcheroo.semantics.manager.resolve_snapshots_dir") as mock_snap:
        mock_snap.return_value = Path("/non/existent/snap")
        with patch.object(Path, "exists", return_value=False):
          # Also mock empty registry to ensure total isolation
          with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
            mgr = SemanticsManager()
            mgr._reverse_index = {}

            assert "test_alpha_add" not in mgr.data
