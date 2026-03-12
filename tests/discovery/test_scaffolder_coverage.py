import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.enums import SemanticTier
import re
import json


def test_scaffolder_specific_coverage():
  scaffolder = Scaffolder()

  # 96-97: Invalid regex pattern
  with patch("ml_switcheroo.discovery.scaffolder.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.discovery_heuristics = {"neural": ["["]}  # Invalid regex pattern
    mock_get.return_value = mock_adapter
    with patch("ml_switcheroo.frameworks.available_frameworks", return_value=["dummy"]):
      scaffolder._lazy_load_heuristics()  # Should log and continue

  # 246: flax_nnx version
  scaffolder.staged_mappings = {"flax_nnx": {"op": "test"}}
  scaffolder.staged_specs = {}
  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=MagicMock()):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=MagicMock()):
      with patch("builtins.open"):
        with patch("json.load", return_value={}):
          with patch("json.dump"):
            try:
              scaffolder.scaffold(["dummy"])
            except Exception:
              pass

  # 435-437: mappings in existing and new_data
  path_mock = MagicMock()
  path_mock.exists.return_value = True
  new_data = {"mappings": {"b": 2}, "other": 1}
  with patch("builtins.open"):
    with patch("json.load", return_value={"mappings": {"a": 1}}):
      with patch("json.dump"):
        scaffolder._write_json(path_mock, new_data, merge=True)

  # 441-442: JSONDecodeError
  with patch("builtins.open"):
    with patch("json.load", side_effect=json.JSONDecodeError("msg", "doc", 0)):
      with patch("json.dump"):
        scaffolder._write_json(path_mock, new_data, merge=True)

  # 257: _get_known_tier without _key_origins
  class DummySemantics:
    pass

  scaffolder.semantics = DummySemantics()
  assert scaffolder._get_ops_by_tier(SemanticTier.NEURAL) == set()

  # 279, 281: _is_structurally_layer
  with patch.object(scaffolder, "_lazy_load_heuristics", return_value={"neural": [re.compile(r"conv")]}):
    assert scaffolder._is_structurally_neural("foo.conv2d", "function") is True
    assert scaffolder._is_structurally_neural("foo.Layer", "class") is True

  # 329-330: primary_path in other_cat
  scaffolder.semantics = MagicMock()
  scaffolder.semantics.data = {}
  scaffolder.staged_specs = {"foo.json": {}}
  scaffolder.staged_mappings = {"jax": {}, "other": {}}
  scaffolder._register_mapping = MagicMock()
  scaffolder._register_entry(
    "foo.json", "bar", "jax", "foo.bar", {"name": "bar"}, {"jax": {}, "other": {"foo.bar": {"name": "bar"}}}
  )

  # 379-380: fuzzy match without fw_key
  cat = {"path1": {"name": "test_op"}}
  scaffolder._find_fuzzy_match(cat, "test_op", [])
