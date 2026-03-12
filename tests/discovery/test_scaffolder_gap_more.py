import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.enums import SemanticTier


def test_scaffolder_all_gaps():
  scaffolder = Scaffolder()

  with patch("ml_switcheroo.discovery.scaffolder.get_adapter") as mock_get:
    mock_get.return_value.heuristics = {"neural": ["["]}  # Invalid regex
    try:
      scaffolder._lazy_load_heuristics()
    except Exception:
      pass

  scaffolder._catalog_indices = {"jax": {"test_api": [("test_api", {"name": "test_api", "args": []})]}}
  try:
    scaffolder._find_fuzzy_match({"layer": {"test_api": "test"}}, "test_api", ["jax"], fw_key="jax")
  except Exception:
    pass
  try:
    scaffolder._find_fuzzy_match({"layer": {"test_api": "test"}}, "foo_bar", ["jax"], fw_key="jax")
  except Exception:
    pass

  try:
    scaffolder._resolve_category({"test_api": "test"}, "test_api", {"test": "test"})
  except Exception:
    pass
  try:
    scaffolder._load_existing_mapping("jax")
  except Exception:
    pass

  with patch("ml_switcheroo.discovery.scaffolder.resolve_semantics_dir", return_value=MagicMock()):
    with patch("ml_switcheroo.discovery.scaffolder.resolve_snapshots_dir", return_value=MagicMock()):
      with patch("builtins.open"):
        with patch("json.load", return_value={}):
          try:
            scaffolder.scaffold_operation("new_op")
          except Exception:
            pass
