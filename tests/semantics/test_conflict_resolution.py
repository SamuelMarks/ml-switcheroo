"""
Tests for Semantics Conflict Resolution.

Verifies that:
1. Loading defines origins.
2. Extras override Array/Neural without standard warnings, BUT preserve high-value tiers.
3. Array vs Neural conflicts handle upgrades silently.
4. Genuine ambiguous signature mismatches trigger warnings.
5. Subset/Superset signature updates are silent (Merge Logic).
"""

import pytest
import warnings
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_tier_data
from ml_switcheroo.enums import SemanticTier


class MockConflictSemantics(SemanticsManager):
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
  mgr = MockConflictSemantics()

  data = {"abs": {"doc": "Math"}}

  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=data,
    tier=SemanticTier.ARRAY_API,
  )

  assert "abs" in mgr.data
  assert mgr._key_origins["abs"] == SemanticTier.ARRAY_API.value


def test_array_vs_neural_silent_upgrade():
  """
  Scenario: 'sigmoid' defined in Array API, then upgraded in Neural.
  Expectation: Content updated, Tier Origin updated to Neural, NO warning emitted (Refinement upgrade).
  """
  mgr = MockConflictSemantics()

  # 1. Load Array
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content={"sigmoid": {"type": "math"}},
    tier=SemanticTier.ARRAY_API,
  )

  # 2. Load Neural (Upgrade)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content={"sigmoid": {"type": "layer"}},
      tier=SemanticTier.NEURAL,
    )

    # Assert no "Conflict detected" warning for legitimate upgrade
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0

  # 3. Verify Overwrite
  assert mgr.data["sigmoid"]["type"] == "layer"
  assert mgr._key_origins["sigmoid"] == SemanticTier.NEURAL.value


def test_extras_override_silence():
  """
  Scenario: 'DataLoader' defined in Neural (high precedence), overridden in Extras (low precedence).

  Expectation:
  1. Data matches Extras (Patching allowed).
  2. Tier remains Neural (Downgrade protection).
  3. No Warning issued (Extras are silent patchers).
  """
  mgr = MockConflictSemantics()

  # 1. Load Base (Precedence 3)
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content={"DataLoader": {"ver": 1}},
    tier=SemanticTier.NEURAL,
  )

  # 2. Load Extras (Override, Precedence 1)
  # catch_warnings(record=True) checks that NO warning is issued
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")  # Cause all warnings to always be triggered.
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content={"DataLoader": {"ver": 2}},
      tier=SemanticTier.EXTRAS,
    )

    # Filter for our specific warning type to be sure
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0, "Extras override should not trigger conflict warning"

  # 3. Verify Data Overwrite (Content patched)
  assert mgr.data["DataLoader"]["ver"] == 2

  # 4. Verify Tier Preservation (Critical for state injection logic)
  # Must remain NEURAL
  assert mgr._key_origins["DataLoader"] == SemanticTier.NEURAL.value


def test_duplicate_same_tier_arg_count_upgrade_silent():
  """
  Scenario: Same key, same tier. New def has MORE args (Superset).
  Expectation: Silent Upgrade.
  """
  mgr = MockConflictSemantics()

  content_a = {"add": {"std_args": ["a"]}}
  content_b = {"add": {"std_args": ["x", "y"]}}

  # 1. Load First (1 arg)
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_a,
    tier=SemanticTier.ARRAY_API,
  )

  # 2. Load Second (2 args, upgrade)
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content=content_b,
      tier=SemanticTier.ARRAY_API,
    )
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0

  # Verify Upgrade
  assert mgr.data["add"]["std_args"] == ["x", "y"]


def test_duplicate_same_tier_arg_count_downgrade_protects_old():
  """
  Scenario: Same key, same tier. New def has FEWER args (Subset).
  Expectation: Silent Persistence of Old Data.
  """
  mgr = MockConflictSemantics()

  content_rich = {"add": {"std_args": ["x", "y"], "description": "Rich"}}
  content_poor = {"add": {"std_args": ["x"], "description": "Poor"}}

  # 1. Load Rich
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_rich,
    tier=SemanticTier.ARRAY_API,
  )

  # 2. Load Poor
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_poor,
    tier=SemanticTier.ARRAY_API,
  )

  # Verify Preservation
  assert mgr.data["add"]["std_args"] == ["x", "y"]
  assert mgr.data["add"]["description"] == "Rich"


def test_duplicate_same_tier_ambiguous_warning():
  """
  Scenario: Same key, same tier, SAME LENGTH, different names.
  Expectation: Warning issued.
  """
  mgr = MockConflictSemantics()

  content_a = {"add": {"std_args": ["x", "y"]}}
  content_b = {"add": {"std_args": ["a", "b"]}}

  # 1. Load First
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_a,
    tier=SemanticTier.ARRAY_API,
  )

  # 2. Load Second (Collision)
  with pytest.warns(UserWarning, match="Signature mismatch"):
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content=content_b,
      tier=SemanticTier.ARRAY_API,
    )


def test_duplicate_same_tier_identical_is_silent():
  """
  Scenario: Content is reloaded (identical or minor metadata change only).
  Expectation: No warning.
  """
  mgr = MockConflictSemantics()

  # Same signature, different descriptions (minor update)
  content_a = {"add": {"std_args": ["x"], "description": "v1"}}
  content_b = {"add": {"std_args": ["x"], "description": "v2"}}  # Same args

  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_a,
    tier=SemanticTier.ARRAY_API,
  )

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content=content_b,
      tier=SemanticTier.ARRAY_API,
    )
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0


def test_build_index_refresh():
  """
  Verify _build_index is capable of updating mappings after a merge.
  """
  mgr = MockConflictSemantics()

  # Init Data: abs -> torch.abs
  data_a = {"abs": {"variants": {"torch": {"api": "torch.abs"}}}}

  # Override Data: abs -> torch.absolute (hypothetical fix in Extras)
  data_c = {"abs": {"variants": {"torch": {"api": "torch.absolute"}}}}

  # 1. Process A
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=data_a,
    tier=SemanticTier.ARRAY_API,
  )
  mgr._build_index()
  assert mgr.get_definition("torch.abs")[0] == "abs"

  # 2. Process C
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=data_c,
    tier=SemanticTier.EXTRAS,
  )
  mgr._build_index()

  # New mapping exists
  assert mgr.get_definition("torch.absolute")[0] == "abs"

  # Old mapping check
  assert mgr.get_definition("torch.abs") is None
