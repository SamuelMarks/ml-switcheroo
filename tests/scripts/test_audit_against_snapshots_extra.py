"""Extra tests for audit_against_snapshots.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path("src").resolve()))
from scripts.audit_against_snapshots import load_snapshots, load_snapshots_multi


def test_load_snapshots_branches() -> None:
  """Test load_snapshots branches."""
  mock_dir = MagicMock()
  mock_file1 = MagicMock()
  mock_file1.name = "torch_map.json"
  mock_file2 = MagicMock()
  mock_file2.name = "torch_vunknown.json"
  mock_file3 = MagicMock()
  mock_file3.name = "torch_v1.json"

  mock_dir.glob.return_value = [mock_file1, mock_file2, mock_file3]

  dummy_data = {
    "categories": {
      "list_cat": [{"api_path": "a"}, {"name": "b"}, {"aliases": ["c", "d"]}],
      "dict_cat": {"e": {"api": "e"}},
    },
    "functions": {"f": {}},
    "classes": {"g": {}},
    "extra_item": {"args": []},
    "extra_item2": {},
  }

  with patch("builtins.open", new_callable=MagicMock):
    with patch("json.load", return_value=dummy_data):
      snapshots = load_snapshots(mock_dir)
      assert "torch" in snapshots
      t = snapshots["torch"]
      assert "a" in t and "b" in t and "c" in t and "d" in t
      assert "e" in t
      assert "f" in t
      assert "g" in t
      assert "extra_item" in t
      assert "extra_item2" not in t


def test_load_snapshots_multi_branches() -> None:
  """Test load_snapshots_multi branches."""
  mock_dir1 = MagicMock()
  mock_dir1.exists.return_value = False

  mock_dir2 = MagicMock()
  mock_dir2.exists.return_value = True

  mock_file1 = MagicMock()
  mock_file1.name = "torch_map.json"
  mock_file2 = MagicMock()
  mock_file2.name = "torch_vunknown.json"
  mock_file3 = MagicMock()
  mock_file3.name = "torch_v1.json"

  mock_dir2.glob.return_value = [mock_file1, mock_file2, mock_file3]

  dummy_data = {
    "categories": {
      "list_cat": [{"api_path": "a"}, {"name": "b"}, {"aliases": ["c", "d"]}],
      "dict_cat": {"e": {"api": "e"}},
    },
    "functions": {"f": {}},
    "classes": {"g": {}},
    "extra_item": {"args": []},
    "extra_item2": {},
  }

  with patch("builtins.open", new_callable=MagicMock):
    with patch("json.load", return_value=dummy_data):
      snapshots = load_snapshots_multi([mock_dir1, mock_dir2])
      assert "torch" in snapshots
      t = snapshots["torch"]
      assert "a" in t and "b" in t and "c" in t and "d" in t
      assert "e" in t
      assert "f" in t
      assert "g" in t
      assert "extra_item" in t
      assert "extra_item2" not in t
