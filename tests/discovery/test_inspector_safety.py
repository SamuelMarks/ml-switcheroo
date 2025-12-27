"""
Tests for Inspector Safety Logic (Blacklist Decoupling).

Verifies that:
1. Inspector accepts `unsafe_modules` argument.
2. Inspector respects the blacklist by skipping traversal.
3. Scaffolder passes adapter blacklist to Inspector.
"""

import pytest
from unittest.mock import MagicMock, patch

from ml_switcheroo.discovery.inspector import ApiInspector
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.semantics.manager import SemanticsManager


def test_inspector_skips_blacklisted_module():
  """
  Scenario: Object has a submodule 'dangerous' that triggers infinite loop
  or crash mock if accessed. Blacklist prevents access.
  """
  # Setup a trap object that fails if inspected
  trap = MagicMock()

  mod = MagicMock()
  mod.dangerous = trap
  mod.safe_func = MagicMock()

  inspector = ApiInspector()

  # Configure scanning
  with patch("inspect.getmembers") as mock_members:
    # Return members dict
    def get_members(obj):
      if obj == mod:
        return [("dangerous", trap), ("safe_func", mod.safe_func)]
      return []

    mock_members.side_effect = get_members

    # We also need to mock importlib to return our mod
    with patch("importlib.import_module", return_value=mod):
      # Execute with blacklist
      catalog = inspector.inspect("pkg", unsafe_modules={"dangerous"})

  # If skipped, it won't be in the catalog
  assert "pkg.dangerous" not in catalog


def test_scaffolder_integration(tmp_path):
  """
  Verify Scaffolder reads `unsafe_submodules` from adapter and passes to inspector.
  """
  # Mock Adapter
  mock_adapter = MagicMock()
  mock_adapter.unsafe_submodules = {"_C", "internal"}
  mock_adapter.search_modules = ["pkg"]

  # Mock Inspector
  mock_inspector = MagicMock()

  # Setup scaffolder with mocks
  scaffolder = Scaffolder(semantics=SemanticsManager())
  scaffolder.inspector = mock_inspector

  # Mock dependencies
  with patch("ml_switcheroo.discovery.scaffolder.get_adapter", return_value=mock_adapter):
    with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["pkg"]):
      # Run
      scaffolder.scaffold(["pkg"], root_dir=tmp_path)

      # Assert
      mock_inspector.inspect.assert_called_with("pkg", unsafe_modules={"_C", "internal"})
