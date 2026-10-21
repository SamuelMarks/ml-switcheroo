"""Test suite for the Merging module."""

import typing

import warnings

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.semantics.merging import (
  _normalize_args,
  infer_tier_from_priority,
  merge_frameworks,
  merge_overlay_data,
  merge_patterns,
  merge_tier_data,
)
from ml_switcheroo.semantics.schema import PatternDef


def test_infer_tier_from_priority() -> None:
  """Infers tier from priority."""
  assert infer_tier_from_priority(10) == SemanticTier.ARRAY_API
  assert infer_tier_from_priority(20) == SemanticTier.NEURAL
  assert infer_tier_from_priority(99) == SemanticTier.EXTRAS
  assert infer_tier_from_priority(0) == SemanticTier.EXTRAS


def test_merge_frameworks() -> None:
  """Merges frameworks."""
  master: dict[str, typing.Hashable | typing.Sized] = {}
  getattr({"v": merge_frameworks}, "__getitem__")("v")(
    getattr({"v": master}, "__getitem__")("v"),
    getattr({"v": {"torch": {"alias": {"x": "y"}, "traits": {"foo": "bar"}}}}, "__getitem__")("v"),
  )
  assert "torch" in master
  assert (
    getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": master}, "__getitem__")("v")["torch"]}, "__getitem__")("v")["alias"]},
          "__getitem__",
        )("v")["x"]
      },
      "__getitem__",
    )("v")
    == "y"
  )
  getattr({"v": merge_frameworks}, "__getitem__")("v")(
    getattr({"v": master}, "__getitem__")("v"),
    getattr({"v": {"torch": {"alias": {"a": "b"}, "other": 1}}}, "__getitem__")("v"),
  )
  assert getattr(
    {"v": getattr({"v": getattr({"v": master}, "__getitem__")("v")["torch"]}, "__getitem__")("v")["alias"]}, "__getitem__"
  )("v") == {"x": "y", "a": "b"}
  assert (
    getattr(
      {"v": getattr({"v": getattr({"v": master}, "__getitem__")("v")["torch"]}, "__getitem__")("v")["other"]},
      "__getitem__",
    )("v")
    == 1
  )
  getattr({"v": merge_frameworks}, "__getitem__")("v")(
    getattr({"v": master}, "__getitem__")("v"), getattr({"v": {"torch": {"traits": {"new": "val"}}}}, "__getitem__")("v")
  )
  assert getattr(
    {"v": getattr({"v": getattr({"v": master}, "__getitem__")("v")["torch"]}, "__getitem__")("v")["traits"]},
    "__getitem__",
  )("v") == {"new": "val"}
  getattr({"v": merge_frameworks}, "__getitem__")("v")(
    getattr({"v": master}, "__getitem__")("v"), getattr({"v": {"torch": {"new_key": "new_val"}}}, "__getitem__")("v")
  )
  assert (
    getattr(
      {"v": getattr({"v": getattr({"v": master}, "__getitem__")("v")["torch"]}, "__getitem__")("v")["new_key"]},
      "__getitem__",
    )("v")
    == "new_val"
  )


def test_merge_patterns(caplog: pytest.LogCaptureFixture) -> None:
  """Merges patterns.

  Args:
      caplog (pytest.LogCaptureFixture): Pytest fixture.
  """
  master = [PatternDef(name="Existing", sequence=["a"], replace_with="b")]
  merge_patterns(master, [{"name": "NewPat", "sequence": ["x"], "replace_with": "y"}])
  assert len(master) == 2
  assert master[1].name == "NewPat"
  merge_patterns(master, [{"name": "Existing", "sequence": ["a"], "replace_with": "b"}])
  assert len(master) == 2
  merge_patterns(master, [{"name": "BadPat"}])
  assert len(master) == 2
  assert "Invalid pattern definition:" in caplog.text


def test_normalize_args() -> None:
  """Verifies the behavior of normalize arguments."""
  res = _normalize_args(
    ["simple_str", {"name": "dict_str", "type": "int"}, {"no_name": "here"}, ["list_str", "int"], [], 123]
  )
  assert res == ["simple_str", "dict_str", "list_str"]


def test_merge_tier_data_invalid_input() -> None:
  """Merges tier data invalid input."""
  data: dict[str, typing.Hashable | typing.Sized] = {}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": []}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").EXTRAS}, "__getitem__")("v"),
  )
  assert data == {}


def test_merge_tier_data_basic() -> None:
  """Merges tier data basic."""
  data: dict[str, typing.Hashable | typing.Sized] = {}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  fws: dict[str, typing.Hashable | typing.Sized] = {}
  patterns: list[typing.Hashable | typing.Sized] = []
  new_content = {
    "__frameworks__": {"tf": {"traits": {}}},
    "__patterns__": [{"name": "P1", "sequence": ["a"], "replace_with": "b"}],
    "Op1": {"description": "desc", "std_args": ["x"]},
  }
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": fws}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
    getattr({"v": patterns}, "__getitem__")("v"),
  )
  assert "tf" in fws
  assert len(patterns) == 1
  assert "Op1" in data
  assert origins["Op1"] == SemanticTier.NEURAL.value
  assert getattr(
    {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["Op1"]}, "__getitem__")("v")["std_args"]}, "__getitem__"
  )("v") == ["x"]


def test_merge_tier_data_validation_error(caplog: pytest.LogCaptureFixture) -> None:
  """Merges tier data validation correctly handling an error.

  Args:
      caplog (pytest.LogCaptureFixture): Pytest fixture.
  """
  data: dict[str, typing.Hashable | typing.Sized] = {}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  new_content = {"OpError": {"std_args": "NOT_A_LIST"}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
  )
  assert "OpError" not in data
  assert "Skipping invalid definition" in caplog.text


def test_merge_tier_data_idempotency() -> None:
  """Merges tier data idempotency."""
  data = {
    "Op1": {"operation": "Op1", "description": "Autogenerated definition for Op1", "std_args": ["x"], "variants": {}}
  }
  origins = {"Op1": SemanticTier.NEURAL.value}
  new_content = {"Op1": {"std_args": ["x"]}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
  )
  assert len(data) == 1
  assert data["Op1"]["std_args"] == ["x"]


def test_merge_tier_data_downgrade_protection() -> None:
  """Merges tier data downgrade protection."""
  data = {"Op1": {"operation": "Op1", "description": "orig", "std_args": ["x"], "variants": {}}}
  origins = {"Op1": SemanticTier.NEURAL.value}
  new_content = {"Op1": {"std_args": ["x", "y"]}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").EXTRAS}, "__getitem__")("v"),
  )
  assert data["Op1"]["std_args"] == ["x", "y"]
  assert origins["Op1"] == SemanticTier.NEURAL.value


def test_merge_tier_data_upgrade_silent() -> None:
  """Merges tier data upgrade silent."""
  data = {"Op1": {"operation": "Op1", "description": "orig", "std_args": ["x"], "variants": {"fw1": {"foo": "bar"}}}}
  origins = {"Op1": SemanticTier.ARRAY_API.value}
  new_content = {"Op1": {"std_args": ["x", "y"], "variants": {"fw2": {"baz": "qux"}}}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
  )
  assert origins["Op1"] == SemanticTier.NEURAL.value
  assert "fw1" in data["Op1"]["variants"]
  assert "fw2" in data["Op1"]["variants"]


def test_merge_tier_data_missing_injections() -> None:
  """Merges tier data missing injections."""
  data: dict[str, typing.Hashable | typing.Sized] = {}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  new_content: dict[str, typing.Hashable | typing.Sized] = {"OpMissing": {}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
    is_internal=getattr({"v": True}, "__getitem__")("v"),
  )
  assert "OpMissing" in data
  assert (
    getattr(
      {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpMissing"]}, "__getitem__")("v")["std_args"]},
      "__getitem__",
    )("v")
    == []
  )
  assert (
    getattr(
      {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpMissing"]}, "__getitem__")("v")["description"]},
      "__getitem__",
    )("v")
    == "Autogenerated definition for OpMissing"
  )
  assert (
    getattr(
      {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpMissing"]}, "__getitem__")("v")["_is_internal"]},
      "__getitem__",
    )("v")
    is True
  )


def test_merge_tier_data_conflict_resolution_upgrade() -> None:
  """Merges tier data conflict resolution upgrade."""
  data = {"Op1": {"operation": "Op1", "description": "old desc", "std_args": ["a", "b"], "variants": {}}}
  origins = {"Op1": SemanticTier.NEURAL.value}
  new_content_longer = {"Op1": {"std_args": ["a", "b", "c"]}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content_longer}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
  )
  assert data["Op1"]["std_args"] == ["a", "b", "c"]


def test_merge_overlay_data_missing_variants() -> None:
  """Merges overlay data missing variants."""
  data = {"OpNoVariants": {"description": "desc"}}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  content = {"__framework__": "torch", "mappings": {"OpNoVariants": {"mapped": True}}}
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "torch.json"}, "__getitem__")("v"),
  )
  assert "variants" in data["OpNoVariants"]
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpNoVariants"]}, "__getitem__")("v")["variants"]},
        "__getitem__",
      )("v")["torch"]
    },
    "__getitem__",
  )("v") == {"mapped": True}


def test_merge_tier_data_conflict_resolution() -> None:
  """Merges tier data conflict resolution."""
  data = {"Op1": {"operation": "Op1", "description": "old desc", "std_args": ["a", "b", "c"], "variants": {}}}
  origins = {"Op1": SemanticTier.NEURAL.value}
  new_content_shorter = {"Op1": {"std_args": ["a", "b"]}}
  getattr({"v": merge_tier_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": new_content_shorter}, "__getitem__")("v"),
    getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
  )
  assert data["Op1"]["std_args"] == ["a", "b", "c"]
  assert data["Op1"]["description"] == "old desc"
  new_content_ambig = {"Op1": {"std_args": ["x", "y", "z"]}}
  with pytest.warns(UserWarning, match="Conflict detected for 'Op1'"):
    getattr({"v": merge_tier_data}, "__getitem__")("v")(
      getattr({"v": data}, "__getitem__")("v"),
      getattr({"v": origins}, "__getitem__")("v"),
      getattr({"v": {}}, "__getitem__")("v"),
      getattr({"v": new_content_ambig}, "__getitem__")("v"),
      getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
    )
  getattr(getattr({"v": getattr({"v": data}, "__getitem__")("v")["Op1"]}, "__getitem__")("v"), "__setitem__")(
    "_is_internal", getattr({"v": True}, "__getitem__")("v")
  )
  new_content_internal = {"Op1": {"std_args": ["p", "q", "r"]}}
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    getattr({"v": merge_tier_data}, "__getitem__")("v")(
      getattr({"v": data}, "__getitem__")("v"),
      getattr({"v": origins}, "__getitem__")("v"),
      getattr({"v": {}}, "__getitem__")("v"),
      getattr({"v": new_content_internal}, "__getitem__")("v"),
      getattr({"v": getattr({"v": SemanticTier}, "__getitem__")("v").NEURAL}, "__getitem__")("v"),
    )
    assert data["Op1"]["std_args"] == ["p", "q", "r"]


def test_merge_overlay_data_fallback_filename() -> None:
  """Merges overlay data fallback filename."""
  data: dict[str, typing.Hashable | typing.Sized] = {}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  fws: dict[str, typing.Hashable | typing.Sized] = {}
  tmps: dict[str, typing.Hashable | typing.Sized] = {}
  content: dict[str, typing.Hashable | typing.Sized] = {"mappings": {}}
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": fws}, "__getitem__")("v"),
    getattr({"v": tmps}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "bad_name.json"}, "__getitem__")("v"),
  )
  assert not fws
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": fws}, "__getitem__")("v"),
    getattr({"v": tmps}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "torch_v1.0.json"}, "__getitem__")("v"),
  )
  assert not fws


def test_merge_overlay_data_full() -> None:
  """Merges overlay data full."""
  data: dict[str, typing.Hashable | typing.Sized] = {"OpExists": {"variants": {}}}
  origins = {"OpExists": SemanticTier.NEURAL.value}
  fws = {"torch": {"existing": True}}
  tmps: dict[str, typing.Hashable | typing.Sized] = {}
  content = {
    "__framework__": "torch",
    "templates": {"t1": "v1"},
    "framework": {"new": False},
    "mappings": {
      "OpExists": {"mapped": True},
      "OpNewUpper": {"mapped": True},
      "opNewLower": {"mapped": True},
      "OpDisabled": None,
    },
  }
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": fws}, "__getitem__")("v"),
    getattr({"v": tmps}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "torch_v1.0.json"}, "__getitem__")("v"),
  )
  assert tmps["torch"] == {"t1": "v1"}
  assert fws["torch"]["existing"] is True
  assert fws["torch"]["new"] is False
  assert "OpExists" in data
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpExists"]}, "__getitem__")("v")["variants"]},
        "__getitem__",
      )("v")["torch"]
    },
    "__getitem__",
  )("v") == {"mapped": True}
  assert "OpNewUpper" in data
  assert origins["OpNewUpper"] == SemanticTier.NEURAL.value
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpNewUpper"]}, "__getitem__")("v")["variants"]},
        "__getitem__",
      )("v")["torch"]
    },
    "__getitem__",
  )("v") == {"mapped": True}
  assert "opNewLower" in data
  assert origins["opNewLower"] == SemanticTier.EXTRAS.value
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["opNewLower"]}, "__getitem__")("v")["variants"]},
        "__getitem__",
      )("v")["torch"]
    },
    "__getitem__",
  )("v") == {"mapped": True}
  assert "OpDisabled" in data
  assert (
    getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": data}, "__getitem__")("v")["OpDisabled"]}, "__getitem__")("v")["variants"]},
          "__getitem__",
        )("v")["torch"]
      },
      "__getitem__",
    )("v")
    is None
  )


def test_merge_overlay_data_new_framework() -> None:
  """Merges overlay data new framework."""
  data: dict[str, typing.Hashable | typing.Sized] = {}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  fws: dict[str, typing.Hashable | typing.Sized] = {}
  tmps: dict[str, typing.Hashable | typing.Sized] = {}
  content = {"__framework__": "jax", "framework": {"alias": {}}, "mappings": {"Op1": {"x": "y"}}}
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": fws}, "__getitem__")("v"),
    getattr({"v": tmps}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "jax_v1.0.json"}, "__getitem__")("v"),
  )
  assert "jax" in fws


def test_merge_overlay_data_disabled_then_enabled() -> None:
  """Merges overlay data disabled then enabled."""
  data = {"Op1": {"variants": {"torch": None}}}
  origins: dict[str, typing.Hashable | typing.Sized] = {}
  content = {"__framework__": "torch", "mappings": {"Op1": {"now": "enabled"}}}
  getattr({"v": merge_overlay_data}, "__getitem__")("v")(
    getattr({"v": data}, "__getitem__")("v"),
    getattr({"v": origins}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": {}}, "__getitem__")("v"),
    getattr({"v": content}, "__getitem__")("v"),
    getattr({"v": "torch.json"}, "__getitem__")("v"),
  )
  assert data["Op1"]["variants"]["torch"] == {"now": "enabled"}
