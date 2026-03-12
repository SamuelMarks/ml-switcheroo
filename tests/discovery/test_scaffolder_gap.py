import pytest
from ml_switcheroo.discovery.scaffolder import Scaffolder
from unittest.mock import patch, MagicMock


def test_scaffolder_get_close_matches():
  scaffolder = Scaffolder()
  # Mock difflib.get_close_matches to return ["foo"]
  with patch("difflib.get_close_matches", return_value=["foo"]):
    assert scaffolder._match_spec_op("fob", ["foo", "bar"]) == "foo"


def test_scaffolder_candidates_empty():
  scaffolder = Scaffolder()
  # Mock find_matches so it returns empty candidates

  # Make search return empty candidates
  scaffolder._catalog_indices = {"jax": {"bar": []}}
  assert scaffolder._find_fuzzy_match({}, "foo", [], fw_key="jax") is None
  assert scaffolder._find_fuzzy_match({}, "foo", []) is None


def test_scaffolder_lazy_load_error():
  scaffolder = Scaffolder()
  # Mock adapter heuristics with invalid regex
  with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["jax"]):
    with patch("ml_switcheroo.discovery.scaffolder.get_adapter") as MockGet:
      MockGet.return_value.heuristics = {"neural": ["["]}
      scaffolder._lazy_load_heuristics()


def test_scaffolder_scaffold_no_root_dir():
  scaffolder = Scaffolder()
  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=None):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=None):
      with pytest.raises(Exception):
        # Try to run it and it should hit the else branch
        scaffolder.scaffold(["jax"])


def test_scaffolder_inspect_exception():
  scaffolder = Scaffolder()
  # Mock inspector to raise exception
  scaffolder.inspector = MagicMock()
  scaffolder.inspector.inspect.side_effect = Exception("test")
  # Mock get_adapter to provide a valid scan_targets
  with patch("ml_switcheroo.discovery.scaffolder.get_adapter") as MockGet:
    MockGet.return_value.search_modules = ["foo"]
    scaffolder.scaffold(["jax"], root_dir=MagicMock())
