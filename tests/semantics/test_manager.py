"""Test suite for manager."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_init() -> None:
  """Verify `SemanticsManager` handles manager init as expected."""
  sm = SemanticsManager()
  assert isinstance(sm.data, dict)
  assert len(sm.data) >= 1
  assert len(sm._reverse_index) >= 1


def test_manager_coverage() -> None:
  """Verify `SemanticsManager` handles manager coverage as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  assert sm.get_test_template("not_real") is None

  assert sm.get_framework_aliases() is not None

  assert sm.get_all_rng_methods() is not None

  assert sm.get_patterns() is not None


def test_manager_load_validation() -> None:
  """Verify `SemanticsManager` handles manager load validation as expected."""
  from pathlib import Path
  from unittest.mock import mock_open, patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Path doesn't exist
  with patch("pathlib.Path.exists", return_value=False):
    sm.load_validation_report(Path("dummy.json"))

  # Read err
  with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", side_effect=OSError("Read err")):
    sm.load_validation_report(Path("dummy.json"))

  # Success
  m_open = mock_open(read_data='{"Abs": true}')
  with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", m_open):
    sm.load_validation_report(Path("dummy.json"))
    assert sm._validation_status.get("Abs") is True

  # Non-dict report
  m_open_list = mock_open(read_data="[1, 2, 3]")
  with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", m_open_list):
    sm.load_validation_report(Path("dummy.json"))


def test_manager_build_index_aliases_json(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test that _build_index loads aliases.json if it exists.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from unittest.mock import MagicMock, mock_open, patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  mock_path = MagicMock()
  mock_path.is_file.return_value = True
  mock_path.open = mock_open(read_data='{"my_alias": "my_module"}')

  class MockFiles:
    """Mock importlib resources files."""

    def joinpath(self, path: str) -> MagicMock:
      """Mock joinpath for file resolution.

      Args:
          path: Filesystem path under test (`str`).

      Returns:
          Resulting `MagicMock` from joinpath.
      """
      if path == "aliases.json":
        return mock_path
      m = MagicMock()
      m.is_file.return_value = False
      return m

  with patch("importlib.resources.files", return_value=MockFiles()):
    sm = SemanticsManager()

    # Give it data so it triggers the alias registration
    sm.data = {"Op": {"variants": {"fw": {"api": "my_alias.sub"}}}}
    monkeypatch.setattr(
      sm,
      "_key_origins",
      {"Op": 1},
      raising=False,
    )
    sm._build_index()

    # "my_alias" in alias map resolves to "my_module"
    # parts[0] == "my_alias"
    # So "my_alias.sub" -> "my_module.sub"
    assert "my_module.sub" in sm._reverse_index


def test_manager_build_index_aliases_exception() -> None:
  """Test that _build_index handles exception when loading aliases."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  with patch("importlib.resources.files", side_effect=OSError("Failed")):
    sm = SemanticsManager()
    assert isinstance(sm, SemanticsManager)
    assert len(sm.data) >= 1
    assert len(sm._reverse_index) >= 1


def test_manager_empty_impl_and_missing_variant_cache() -> None:
  """Test empty variant implementation and missing _variant_cache attribute."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  # Line 158: if not impl: continue
  sm.data = {"OpWithNone": {"variants": {"torch": None, "jax": {}}}}
  sm._build_index()

  # Line 281: if not hasattr(self, "_variant_cache")
  del sm._variant_cache
  res = sm.resolve_variant("OpWithNone", "torch")
  assert res is None


def test_manager_build_index_no_priority_scores() -> None:
  """Test _build_index when priority_scores.json does not exist."""
  import os
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  orig_exists = os.path.exists
  with patch("os.path.exists", side_effect=lambda p: False if "priority_scores.json" in str(p) else orig_exists(p)):
    sm._build_index()
  assert isinstance(sm._reverse_index, dict)
  assert len(sm._reverse_index) >= 1
  assert len(sm.data) >= 1


def test_manager_update_definition() -> None:
  """Verify `SemanticsManager` emits `NewOp` for manager update definition."""
  from unittest.mock import mock_open, patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Validation error
  sm.update_definition("Abs", {"operation": "Abs", "std_args": "NOT_A_LIST"})  # Will trigger Pydantic error

  # Valid
  m_open = mock_open()
  with patch("builtins.open", m_open), patch("pathlib.Path.mkdir"):
    sm.update_definition("NewOp", {"variants": {"torch": {"api": "torch.new_op"}, "jax": {}}})
    assert "NewOp" in sm.data
    assert "torch.new_op" in sm._reverse_index

  # Write error
  with patch("builtins.open", side_effect=OSError("Write err")), patch("pathlib.Path.mkdir"):
    sm.update_definition("WriteFail", {"variants": {"torch": {"api": "torch.write_fail"}}})
    assert "WriteFail" in sm.data


def test_manager_get_import_map(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsManager` emits `torch` for manager get import map.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Mocking internal states to hit coverage
  sm._providers = {
    "jax": {
      "core": {"root": "jax.numpy", "alias": "jnp"},
      "no_root": {"root": None, "sub": None, "alias": None},
    },
    "flax": {"nn": {"root": "flax.linen", "alias": "nn"}},
  }
  sm._source_registry = {
    "torch": ("torch", "core"),
    "torch.nn": ("torch.nn", "nn"),
    "torch.no_root": ("torch.no_root", "no_root"),
    "torch.optim": ("torch.optim", "optim"),  # unmatched
  }

  # Try direct mapping
  res = sm.get_import_map("jax")
  assert "torch" in res
  assert res["torch"][0] == "jax.numpy"

  # Try inheritance mapping
  # Mock _resolve_inheritance to return "flax" for some target
  monkeypatch.setattr(
    sm,
    "_framework_aliases",
    {
      "myfw": (
        "myfw.mod",
        "myfw",
      )
    },
    raising=False,
  )

  # Let's override resolve_inheritance
  from unittest.mock import patch

  with patch.object(sm, "_resolve_inheritance", side_effect=lambda x: "flax" if x == "custom" else None):
    res2 = sm.get_import_map("custom")
    assert "torch.nn" in res2
    assert res2["torch.nn"][0] == "flax.linen"


def test_manager_resolve_variant() -> None:
  """Verify `SemanticsManager` handles manager resolve variant as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Mock data
  sm.data = {"Abs": {"variants": {"torch": {"api": "torch.abs"}, "numpy": {"api": "np.abs"}}}}

  # Direct match
  torch_var = sm.resolve_variant("Abs", "torch")
  assert torch_var is not None
  assert torch_var["api"] == "torch.abs"

  # Missing
  assert sm.resolve_variant("Unknown", "torch") is None

  # Fallback inheritance match
  from unittest.mock import patch

  with patch.object(sm, "_resolve_inheritance", side_effect=lambda x: "numpy" if x == "custom_numpy" else None):
    custom_var = sm.resolve_variant("Abs", "custom_numpy")
    assert custom_var is not None
    assert custom_var["api"] == "np.abs"

    # Fallback missing
    assert sm.resolve_variant("Abs", "unknown_fw") is None


def test_manager_get_definition_missing() -> None:
  """Verify `SemanticsManager` handles manager get definition missing as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  assert sm.get_definition("unknown.api") is None


def test_manager_get_framework_config() -> None:
  """Verify `SemanticsManager` handles manager get framework config as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm._providers = {"jax": {"core": {"provider": "val"}}}

  res = sm.get_framework_config("jax")
  assert res is not None

  from unittest.mock import patch

  with patch.object(sm, "_resolve_inheritance", side_effect=lambda x: "jax" if x == "custom" else None):
    sm.get_framework_config("custom")
    assert res is not None

    assert sm.get_framework_config("unknown") == {}


def test_manager_resolve_inheritance() -> None:
  """Verify `SemanticsManager` handles manager resolve inheritance as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.framework_configs = {"custom": {"extends": "base_custom"}}
  assert sm._resolve_inheritance("custom") == "base_custom"

  # testing adapter inheriting
  # We mock get_adapter to return a mock adapter with inherits_from
  from unittest.mock import MagicMock, patch

  mock_ad = MagicMock()
  mock_ad.inherits_from = "base_ad"
  with patch("ml_switcheroo.semantics.manager.get_adapter", return_value=mock_ad):
    assert sm._resolve_inheritance("custom2") == "base_ad"


def test_manager_resolve_variant_limit() -> None:
  """Verify `SemanticsManager` handles manager resolve variant limit as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"Abs": {"variants": {"root": {"api": "root.api"}}}}
  # Create an inheritance cycle
  from unittest.mock import patch

  with patch.object(sm, "_resolve_inheritance", side_effect=lambda x: "b" if x == "a" else "a"):
    assert sm.resolve_variant("Abs", "a") is None


def test_manager_is_verified() -> None:
  """Verify `SemanticsManager` handles manager is verified as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm._validation_status = {"Abs": False, "Relu": True}
  assert sm.is_verified("Abs") is False
  assert sm.is_verified("Relu") is True
  assert sm.is_verified("Unknown") is True


def test_manager_get_definition_by_id() -> None:
  """Verify `SemanticsManager` handles manager get definition by id as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"Abs": {"foo": "bar"}}
  assert sm.get_definition_by_id("Abs") == {"foo": "bar"}


def test_manager_get_definition() -> None:
  """Verify `SemanticsManager` handles manager get definition as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {"torch.abs": {"api": "torch.abs"}}
  sm._reverse_index = {"torch.abs": ("Abs", {"api": "torch.abs"})}
  res = sm.get_definition("torch.abs")
  assert res is not None
  assert res[0] == "Abs"


def test_manager_get_definition_missing_real() -> None:
  """Verify `SemanticsManager` handles manager get definition missing real as expected."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  sm.data = {}
  sm._reverse_index = {}
  assert sm.get_definition("completely.unknown.api") is None


def test_semantic_manager_inherit_fallback() -> None:
  """Verify `SemanticsManager` handles semantic manager inherit fallback as expected."""
  manager = SemanticsManager()
  res = manager._resolve_inheritance("unknown_fw")
  assert res is None


def test_semantic_manager_reverse_lookup_fallback() -> None:
  """Verify `SemanticsManager` handles semantic manager reverse lookup fallback as expected."""
  manager = SemanticsManager()
  manager.data = {"AbstractOp": {"description": "A test op"}}

  res = manager.get_definition("AbstractOp")
  assert res == ("AbstractOp", {"description": "A test op"})


def test_get_framework_ecosystem() -> None:
  """Verify `SemanticsManager` handles get framework ecosystem as expected."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  # Mock config and resolve inheritance
  with (
    patch.object(
      sm, "get_framework_config", side_effect=lambda fw: {"alias": {"module": "myalias"}} if fw == "my_fw" else {}
    ),
    patch.object(sm, "_resolve_inheritance", side_effect=lambda fw: "base_fw" if fw == "my_fw" else None),
  ):
    ecosystem = sm.get_framework_ecosystem("my_fw")
    assert ecosystem == {"my_fw", "myalias", "base_fw"}


def test_manager_cache_branch() -> None:
  """Verify ``SemanticsManager`` ``lru_cache`` hit, ``clear_semantics_cache``, and ``empty()`` constructor."""
  from ml_switcheroo.semantics.manager import SemanticsManager, _get_cached_default_snapshot, clear_semantics_cache

  clear_semantics_cache()
  first = SemanticsManager()
  second = SemanticsManager()
  info = _get_cached_default_snapshot.cache_info()
  assert info.hits >= 1
  assert first.data.keys() == second.data.keys()

  empty_mgr = SemanticsManager.empty()
  assert empty_mgr.data == {}
  clear_semantics_cache()
  assert _get_cached_default_snapshot.cache_info().currsize == 0


def test_build_index_priority() -> None:
  """Verify `SemanticsManager` emits `torch.abs` for build index priority."""
  from ml_switcheroo.semantics.manager import SemanticsManager, SemanticTier

  sm = SemanticsManager()
  sm._reverse_index.clear()
  sm.data = {
    "abs1": {"variants": {"torch": {"api": "torch.abs"}}},
    "abs2": {"variants": {"torch": {"api": "torch.abs"}}},
    "abs3": {"variants": {"torch": {"api": "torch.abs"}}},
    "abs4": {"variants": {"torch": {"api": "torch.abs"}}},
  }
  # Set origins to exercise all priority branches
  sm._key_origins = {
    "abs1": SemanticTier.ARRAY_API.value,
    "abs2": SemanticTier.NEURAL.value,
    "abs3": SemanticTier.EXTRAS.value,
    "abs4": "unknown",
  }
  sm._build_index()
  # It should just not crash and should exercise get_priority branches
  assert "torch.abs" in sm._reverse_index


def test_get_known_apis(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `SemanticsManager` handles get known apis as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  expected_data: dict[str, dict[str, JsonValue]] = {"test": {"val": 123}}
  monkeypatch.setattr(
    sm,
    "data",
    expected_data,
    raising=False,
  )
  assert sm.get_known_apis() == expected_data


def test_manager_get_definition_branches() -> None:
  """Test reverse index lookup and candidate fallback without variants in get_definition."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  # Found in reverse index (line 357)
  defn = sm.get_definition("torch.abs")
  assert defn is not None

  # Missing _reverse_index attribute initializes it (line 354)
  del sm._reverse_index
  assert sm.get_definition("torch.abs") is None

  # Candidate in data without variants (line 367)
  sm.data["dummy_no_var"] = {"operation": "dummy_no_var"}
  defn_no_var = sm.get_definition("dummy_no_var")
  assert defn_no_var is not None
  assert defn_no_var[0] == "dummy_no_var"

  # Candidate in data with variants (line 363)
  sm.data["dummy_with_var"] = {"variants": {"jax": {"api": "jax.dummy"}}}
  defn_with_var = sm.get_definition("dummy_with_var")
  assert defn_with_var is not None
  assert defn_with_var[0] == "dummy_with_var"

  # Case-insensitive resolution when defn is found but lacks variant (lines 305-314)
  sm.data["MyOp"] = {"variants": {"torch": {"api": "torch.my_op"}}}
  sm.data["myop"] = {"variants": {"jax": {"api": "jax.my_op"}}}
  res_var = sm.resolve_variant("MyOp", "jax")
  assert res_var is not None
  assert res_var["api"] == "jax.my_op"

  # Missing variant fallback when operation has empty variants
  sm.data["EmptyVarOp"] = {"variants": {}}
  assert sm.resolve_variant("EmptyVarOp", "jax") is None


def test_manager_build_index_no_attrs() -> None:
  """Test _build_index when _reverse_index and _variant_cache are missing."""
  from ml_switcheroo.semantics.manager import SemanticsManager

  class DummySubclass(SemanticsManager):
    """Minimal `SemanticsManager` test double used in manager tests."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.data = {}
      self.framework_configs = {}

  dummy = DummySubclass()
  dummy._build_index()
  assert isinstance(dummy._reverse_index, dict)
  assert dummy._reverse_index == {}
  assert isinstance(dummy._variant_cache, dict)
  assert dummy._variant_cache == {}


def test_lookup_and_search_snapshot_symbols() -> None:
  """Test lookup_snapshot_symbol and search_snapshot_symbols on SemanticsManager."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Real lookup
  res = sm.lookup_snapshot_symbol("torch", "torch.add")
  assert res is None or isinstance(res, dict)
  assert sm.lookup_snapshot_symbol("unknown", "unknown") is None

  # Real search
  search_res = sm.search_snapshot_symbols("add", framework="torch", limit=5)
  assert isinstance(search_res, list)

  # Non-dict return from lookup
  with patch("ml_ecosystem_snapshots.index.lookup_symbol", return_value=None):
    assert sm.lookup_snapshot_symbol("torch", "unknown") is None

  # Non-list return from search
  with patch("ml_ecosystem_snapshots.index.search_index", return_value=None):
    assert sm.search_snapshot_symbols("unknown") == []

  # Exception handling in lookup
  with patch("ml_ecosystem_snapshots.index.lookup_symbol", side_effect=RuntimeError("DB Error")):
    assert sm.lookup_snapshot_symbol("torch", "torch.add") is None

  # Exception handling in search
  with patch("ml_ecosystem_snapshots.index.search_index", side_effect=RuntimeError("DB Error")):
    assert sm.search_snapshot_symbols("add") == []


def test_validate_python_call() -> None:
  """Test validate_python_call on SemanticsManager."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  report = sm.validate_python_call("torch", "torch.add", args=["x", "y"])
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.python_fw.validate_python_call", side_effect=RuntimeError("Err")):
    assert sm.validate_python_call("torch", "torch.add") is None


def test_diff_snapshot_and_changelog() -> None:
  """Test diff_snapshot_versions and generate_snapshot_changelog on SemanticsManager."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()
  snap1 = {"categories": {"math": [{"name": "foo", "api_path": "test.foo"}]}}
  snap2 = {"categories": {"math": [{"name": "foo", "api_path": "test.foo"}, {"name": "bar", "api_path": "test.bar"}]}}

  diff = sm.diff_snapshot_versions(snap1, snap2)
  assert diff is not None
  changelog = sm.generate_snapshot_changelog(diff)
  assert isinstance(changelog, str)
  assert "Added" in changelog or "test.bar" in changelog

  # Exception branches
  with patch("ml_ecosystem_snapshots.diff.diff_snapshots", side_effect=RuntimeError("Diff Err")):
    assert sm.diff_snapshot_versions(snap1, snap2) is None

  with patch("ml_ecosystem_snapshots.diff.generate_changelog", side_effect=RuntimeError("Changelog Err")):
    assert sm.generate_snapshot_changelog(diff) == ""


def test_extract_framework_isolated() -> None:
  """Test extract_framework_isolated on SemanticsManager."""
  from unittest.mock import patch

  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # Success branch with mock
  with patch("ml_ecosystem_snapshots.api.extract_snapshot_isolated", return_value={"status": "ok"}):
    res = sm.extract_framework_isolated("torch")
    assert res == {"status": "ok"}

  # Non-dict return
  with patch("ml_ecosystem_snapshots.api.extract_snapshot_isolated", return_value="invalid"):
    assert sm.extract_framework_isolated("torch") == {}

  # Exception branch
  with patch("ml_ecosystem_snapshots.api.extract_snapshot_isolated", side_effect=RuntimeError("Extraction Error")):
    assert sm.extract_framework_isolated("torch") == {}


def test_manager_alias_case_insensitive_variant_and_update_definition(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verify incomplete `alias` configs, case-insensitive `resolve_variant`, and `update_definition` fields.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  import ml_switcheroo.semantics.manager as mgr_mod

  mgr = SemanticsManager(load_defaults=False)
  mgr.framework_configs["no_alias_fw"] = {"some_other_key": "val"}
  mgr.framework_configs["missing_alias_fields_fw"] = {"alias": {"module": "only_mod"}}
  mgr.framework_configs["missing_mod_alias_fw"] = {"alias": {"name": "only_name"}}
  mgr._build_index()
  aliases = mgr.get_framework_aliases()
  assert "no_alias_fw" not in aliases
  assert "missing_alias_fields_fw" not in aliases
  assert "missing_mod_alias_fw" not in aliases

  mgr.data["relu"] = {"operation": "relu", "variants": {"torch": {"api": "torch.relu"}}}
  mgr.data["RELU"] = {"operation": "RELU", "variants": {"jax": {"api": "jax.nn.relu"}}}
  mgr._variant_cache.clear()
  assert mgr.resolve_variant("RELU", "torch") == {"api": "torch.relu"}

  mgr.data["OTHER"] = {"operation": "OTHER", "variants": {}}
  mgr.data["other"] = {"operation": "other", "variants": {"flax": {"api": "flax.other"}}}
  mgr._variant_cache.clear()
  assert mgr.resolve_variant("OTHER", "torch") is None

  mgr.data["lower_only"] = {"operation": "lower_only", "variants": {}}
  mgr._variant_cache.clear()
  assert mgr.resolve_variant("lower_only", "torch") is None

  monkeypatch.setattr(mgr_mod, "resolve_semantics_dir", lambda: tmp_path)
  mgr.update_definition(
    "custom_op",
    {
      "operation": "custom_op",
      "variants": {"torch": {"api": "torch.custom_op"}},
      "description": "Full description",
      "std_args": ["x"],
      "frameworks": {},
    },
  )
  assert mgr.data["custom_op"]["description"] == "Full description"
  assert mgr.data["custom_op"]["std_args"] == ["x"]

  mgr.update_definition(
    "test.op",
    {"operation": "test.op", "variants": {}, "description": "Provided description", "std_args": [], "frameworks": {}},
  )
  assert mgr.data["test.op"]["description"] == "Provided description"
