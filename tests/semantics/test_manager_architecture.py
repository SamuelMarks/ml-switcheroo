"""
Tests for SemanticsManager Architecture Compliance.

Verifies that:
1. The Manager populates defaults from Registry (Adapters).
2. It correctly consumes Structural Traits from the Adapter Protocol.
3. It stays clean of JSON data when files are missing.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.testing.adapters import register_adapter
from ml_switcheroo.semantics.schema import StructuralTraits


@pytest.fixture
def empty_semantics_dir(tmp_path):
  """Creates a temporary directory with no JSON files."""
  return tmp_path


def test_initialization_populates_from_registry(empty_semantics_dir):
  """
  Scenario: Environment with no JSON files, but Adapters are registered.
  Expectation: Manager has templates/aliases for registered frameworks (e.g. torch).
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=empty_semantics_dir):
    mgr = SemanticsManager()

    # Should have data from TorchAdapter (registered by default)
    assert "torch" in mgr.test_templates
    assert "import torch" in mgr.test_templates["torch"]["import"]

    # Should have aliases
    aliases = mgr.get_framework_aliases()
    assert "torch" in aliases
    assert aliases["torch"] == ("torch", "torch")


def test_traits_hydration_from_adapter(empty_semantics_dir):
  """
  Scenario: A new custom adapter is registered with 'structural_traits'.
  Expectation: SemanticsManager populates framework_configs with these traits.
  """

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
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=empty_semantics_dir):
    mgr = SemanticsManager()

  # 3. Verify Traits were loaded into config
  config = mgr.get_framework_config("fastnet")
  assert "traits" in config
  traits = config["traits"]

  assert traits["module_base"] == "fastnet.Module"
  assert traits["forward_method"] == "run"
  assert traits["requires_super_init"] is True


def test_clean_slate_if_registry_empty(empty_semantics_dir):
  """
  Scenario: No JSONs and No Adapters.
  Expectation: Manager is truly empty.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=empty_semantics_dir):
    # Mock available_frameworks to return nothing
    with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
      mgr = SemanticsManager()

      assert len(mgr.test_templates) == 0
      assert len(mgr.framework_configs) == 0
      assert len(mgr.data) == 0


def test_test_alpha_add_removed():
  """
  Safeguard against regressions of hardcoded operations.
  """
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir") as mock_resolve:
    mock_resolve.return_value = Path("/non/existent/path")
    with patch.object(Path, "exists", return_value=False):
      # Also mock empty registry to ensure total isolation
      with patch("ml_switcheroo.semantics.manager.available_frameworks", return_value=[]):
        mgr = SemanticsManager()
        assert "test_alpha_add" not in mgr.data
