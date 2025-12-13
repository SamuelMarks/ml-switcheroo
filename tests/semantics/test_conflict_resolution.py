"""
Tests for Semantics Conflict Resolution.

Verifies that:
1. Loading defines origins.
2. Extras override Array/Neural without standard warnings (or handled gracefully).
3. Array vs Neural conflicts trigger warnings.
4. Data is actually updated (last write wins).
"""

import pytest
import warnings
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class TestSemanticsManager(SemanticsManager):
  """Subclass to intercept loading logic for testing."""

  def __init__(self):
    # We manually init structures to skip real file loading
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    # FIX: Initialize required attributes for merge_tier_data
    self.import_data = {}
    self.framework_configs = {}


def test_merge_clean_insert():
  """Verify standard insertion works."""
  mgr = TestSemanticsManager()

  data = {"abs": {"doc": "Math"}}
  mgr._merge_tier(data, SemanticTier.ARRAY_API)

  assert "abs" in mgr.data
  assert mgr._key_origins["abs"] == SemanticTier.ARRAY_API.value


def test_conflict_array_vs_neural_warning():
  """
  Scenario: 'sigmoid' defined in Array API, then redefined in Neural.
  Expectation: Warning issued, data updated to Neural version.
  """
  mgr = TestSemanticsManager()

  # 1. Load Array
  mgr._merge_tier({"sigmoid": {"type": "math"}}, SemanticTier.ARRAY_API)

  # 2. Load Neural (Collision)
  with pytest.warns(UserWarning, match="Conflict detected"):
    mgr._merge_tier({"sigmoid": {"type": "layer"}}, SemanticTier.NEURAL)

  # 3. Verify Overwrite occurred
  assert mgr.data["sigmoid"]["type"] == "layer"
  assert mgr._key_origins["sigmoid"] == SemanticTier.NEURAL.value


def test_extras_override_silence():
  """
  Scenario: 'DataLoader' defined in Neural (hypothetically), overridden in Extras.
  Expectation: No Warning (or distinct log), data updated.
  """
  mgr = TestSemanticsManager()

  # 1. Load Base
  mgr._merge_tier({"DataLoader": {"ver": 1}}, SemanticTier.NEURAL)

  # 2. Load Extras (Override)
  # catch_warnings(record=True) checks that NO warning is issued
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")  # Cause all warnings to always be triggered.
    mgr._merge_tier({"DataLoader": {"ver": 2}}, SemanticTier.EXTRAS)

    # Filter for our specific warning type to be sure
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0, "Extras override should not trigger conflict warning"

  # 3. Verify Overwrite
  assert mgr.data["DataLoader"]["ver"] == 2
  assert mgr._key_origins["DataLoader"] == SemanticTier.EXTRAS.value


def test_duplicate_same_tier_warning():
  """
  Scenario: Same key appears twice in processes labeled as same Tier.
  Expectation: Warning.
  """
  mgr = TestSemanticsManager()
  mgr._merge_tier({"add": {}}, SemanticTier.ARRAY_API)

  with pytest.warns(UserWarning, match="overwritten by"):
    mgr._merge_tier({"add": {}}, SemanticTier.ARRAY_API)


def test_build_index_refresh():
  """
  Verify _build_index is capable of updating mappings after a merge.
  """
  mgr = TestSemanticsManager()

  # Init Data: abs -> torch.abs
  data_a = {"abs": {"variants": {"torch": {"api": "torch.abs"}}}}

  # Override Data: abs -> torch.absolute (hypothetical fix in Extras)
  data_c = {"abs": {"variants": {"torch": {"api": "torch.absolute"}}}}

  # 1. Process A
  mgr._merge_tier(data_a, SemanticTier.ARRAY_API)
  mgr._build_index()
  assert mgr.get_definition("torch.abs")[0] == "abs"

  # 2. Process C
  mgr._merge_tier(data_c, SemanticTier.EXTRAS)
  mgr._build_index()

  # Old mapping should be arguably gone if the data object was replaced?
  # Yes, _build_index clears _reverse_index and rebuilds from self.data.
  # self.data["abs"] is now the object from data_c.

  # New mapping exists
  assert mgr.get_definition("torch.absolute")[0] == "abs"

  # Old mapping check:
  # Since 'torch.abs' is NOT in data_c's variant list, it should point to nothing.
  assert mgr.get_definition("torch.abs") is None
