"""Test suite for the Conflict Resolution module."""

import typing

import warnings

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_tier_data


class MockConflictSemantics(SemanticsManager):
  """Test helper class MockConflictSemantics."""

  def __init__(self) -> None:
    """Initializes the MockConflictSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self.framework_configs = {}


def test_merge_clean_insert() -> None:
  """Merges clean insert."""
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


def test_array_vs_neural_silent_upgrade() -> None:
  """Verifies the behavior of array vs neural silent upgrade."""
  mgr = MockConflictSemantics()
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content={"sigmoid": {"type": "math"}},
    tier=SemanticTier.ARRAY_API,
  )
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content={"sigmoid": {"type": "layer"}},
      tier=SemanticTier.NEURAL,
    )
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0
  assert mgr.data["sigmoid"]["type"] == "layer"
  assert mgr._key_origins["sigmoid"] == SemanticTier.NEURAL.value


def test_extras_override_silence() -> None:
  """Verify extras override silence behavior."""
  mgr = MockConflictSemantics()
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content={"DataLoader": {"ver": 1}},
    tier=SemanticTier.NEURAL,
  )
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content={"DataLoader": {"ver": 2}},
      tier=SemanticTier.EXTRAS,
    )
    relevant = [x for x in w if "Conflict detected" in str(x.message)]
    assert len(relevant) == 0, "Extras override should not trigger conflict warning"
  assert mgr.data["DataLoader"]["ver"] == 2
  assert mgr._key_origins["DataLoader"] == SemanticTier.NEURAL.value


def test_duplicate_same_tier_arg_count_upgrade_silent() -> None:
  """Verifies the behavior of duplicate same tier argument count upgrade silent."""
  mgr = MockConflictSemantics()
  content_a = {"add": {"std_args": ["a"]}}
  content_b = {"add": {"std_args": ["x", "y"]}}
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
  assert mgr.data["add"]["std_args"] == ["x", "y"]


def test_duplicate_same_tier_arg_count_downgrade_protects_old() -> None:
  """Verifies the behavior of duplicate same tier argument count downgrade protects old."""
  mgr = MockConflictSemantics()
  content_rich = {"add": {"std_args": ["x", "y"], "description": "Rich"}}
  content_poor = {"add": {"std_args": ["x"], "description": "Poor"}}
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_rich,
    tier=SemanticTier.ARRAY_API,
  )
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_poor,
    tier=SemanticTier.ARRAY_API,
  )
  assert mgr.data["add"]["std_args"] == ["x", "y"]
  assert mgr.data["add"]["description"] == "Rich"


def test_duplicate_same_tier_ambiguous_warning() -> None:
  """Verifies the behavior of duplicate same tier ambiguous warning."""
  mgr = MockConflictSemantics()
  content_a = {"add": {"std_args": ["x", "y"]}}
  content_b = {"add": {"std_args": ["a", "b"]}}
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=content_a,
    tier=SemanticTier.ARRAY_API,
  )
  with pytest.warns(UserWarning, match="Signature mismatch"):
    merge_tier_data(
      data=mgr.data,
      key_origins=mgr._key_origins,
      framework_configs=mgr.framework_configs,
      new_content=content_b,
      tier=SemanticTier.ARRAY_API,
    )


def test_duplicate_same_tier_identical_is_silent() -> None:
  """Verifies the behavior of duplicate same tier identical is silent."""
  mgr = MockConflictSemantics()
  content_a = {"add": {"std_args": ["x"], "description": "v1"}}
  content_b = {"add": {"std_args": ["x"], "description": "v2"}}
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


def test_build_index_refresh() -> None:
  """Builds index refresh."""
  mgr = MockConflictSemantics()
  data_a = {"abs": {"variants": {"torch": {"api": "torch.abs"}}}}
  data_c = {"abs": {"variants": {"torch": {"api": "torch.absolute"}}}}
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=data_a,
    tier=SemanticTier.ARRAY_API,
  )
  mgr._build_index()
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": mgr}, "__getitem__")("v").get_definition}, "__getitem__")("v")(
            getattr({"v": "torch.abs"}, "__getitem__")("v")
          )
        },
        "__getitem__",
      )("v")[0]
    },
    "__getitem__",
  )("v") == getattr({"v": "abs"}, "__getitem__")("v")
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=data_c,
    tier=SemanticTier.EXTRAS,
  )
  mgr._build_index()
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": mgr}, "__getitem__")("v").get_definition}, "__getitem__")("v")(
            getattr({"v": "torch.absolute"}, "__getitem__")("v")
          )
        },
        "__getitem__",
      )("v")[0]
    },
    "__getitem__",
  )("v") == getattr({"v": "abs"}, "__getitem__")("v")
  assert mgr.get_definition("torch.abs") is None
