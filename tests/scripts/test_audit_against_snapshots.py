"""Tests for audit_against_snapshots.py."""

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from scripts.audit_against_snapshots import audit_frameworks, load_snapshots, load_snapshots_multi

# Provide access to the module
sys.path.insert(0, str(Path("src").resolve()))


@patch("scripts.audit_against_snapshots.sys")
def test_load_snapshots(mock_sys: MagicMock) -> None:
  """Test loading snapshots from a directory.

  Args:
      mock_sys (MagicMock): Mocked sys module.
  """
  mock_path_obj: MagicMock = MagicMock()
  mock_file: MagicMock = MagicMock()
  mock_file.name = "torch_v1.0.json"
  mock_path_obj.glob.return_value = [mock_file]

  with patch("builtins.open", new_callable=MagicMock):
    with patch("json.load", return_value={"functions": {"test_api": {}}}):
      snapshots: Dict[str, Dict[str, Any]] = load_snapshots(mock_path_obj)
      assert set(snapshots.keys()) == {"torch"}
      assert "test_api" in snapshots["torch"]


def test_load_snapshots_empty() -> None:
  """Docstring."""
  from scripts.audit_against_snapshots import load_snapshots

  mock_path_obj: MagicMock = MagicMock()
  mock_path_obj.glob.return_value = []
  assert load_snapshots(mock_path_obj) == {}


def test_audit_frameworks_coverage() -> None:
  """Docstring."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}},
        "torch": {"api": "torch.missing"},
        "jax": {},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}, "torch": {}, "jax": {}}
  errors: List[str] = audit_frameworks(manager, snapshots)
  assert len(errors) == 3

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "kwargs", "kind": "VAR_KEYWORD"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0

  manager.data = {
    "flatten": {
      "variants": {
        "missing": {
          "api": "missing.flatten",
        }
      }
    }
  }
  snapshots = {"missing": {}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}, {"name": "start_axis"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}},
        "torch": {"api": "torch.missing"},
        "jax": {},
      }
    }
  }
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}, "torch": {}, "jax": {}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 3

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager.data = {"flatten": {"variants": {"missing": {}}}}
  assert audit_frameworks(manager, {}) == []


def test_main_block() -> None:
  """Docstring."""
  from scripts.audit_against_snapshots import main

  assert callable(main)


def test_audit_frameworks_coverage2() -> None:
  """Docstring."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {
          "api": "mlx.flatten",
        }
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors: List[str] = audit_frameworks(manager, snapshots)
  assert len(errors) == 0


def test_audit_frameworks() -> None:
  """Docstring."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}},
        "torch": {"api": "torch.missing"},
        "jax": {},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}, "torch": {}, "jax": {}}
  errors: List[str] = audit_frameworks(manager, snapshots)
  assert len(errors) == 3

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "kwargs", "kind": "VAR_KEYWORD"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0

  manager.data = {
    "flatten": {
      "variants": {
        "missing": {
          "api": "missing.flatten",
        }
      }
    }
  }
  snapshots = {"missing": {}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}, {"name": "start_axis"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}},
        "torch": {"api": "torch.missing"},
        "jax": {},
      }
    }
  }
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}, "torch": {}, "jax": {}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 3

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager.data = {
    "flatten": {
      "variants": {
        "torch": {"api": "torch.flatten", "args": {"start_dim": "start_dim"}},
        "mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}},
        "missing": {
          "api": "missing.flatten",
        },
      }
    },
    "missing_op": {"variants": {"torch": {"api": "torch.missing", "args": {}}}},
  }

  snapshots = {
    "torch": {"torch.flatten": {"args": [{"name": "input"}, {"name": "start_dim"}]}},
    "mlx": {"mlx.flatten": {"args": [{"name": "input"}, {"name": "start_axis"}]}},
  }

  errors = audit_frameworks(manager, snapshots)

  # "missing" framework is skipped because we don't have a snapshot for it in this test setup

  assert "[mlx] 'flatten' maps to hallucinated argument: 'missing_axis' for API 'mlx.flatten'" in errors
  assert len(errors) == 3

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 2
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "kwargs", "kind": "VAR_KEYWORD"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots_multi")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_success(
  mock_reg: MagicMock, mock_kb: MagicMock, mock_mgr: MagicMock, mock_load: MagicMock, mock_audit: MagicMock
) -> None:
  """Test main function when there are no errors.

  Args:
      mock_reg (MagicMock): Mock argument.
      mock_kb (MagicMock): Mock argument.
      mock_mgr (MagicMock): Mock argument.
      mock_load (MagicMock): Mock argument.
      mock_audit (MagicMock): Mock argument.
  """
  import sys

  from scripts.audit_against_snapshots import main

  mock_audit.return_value = []

  with patch.object(sys, "argv", ["audit_against_snapshots.py"]):
    assert main() == 0


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots_multi")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_failure_strict(
  mock_reg: MagicMock, mock_kb: MagicMock, mock_mgr: MagicMock, mock_load: MagicMock, mock_audit: MagicMock
) -> None:
  """Test main function when there are errors and strict mode is on.

  Args:
      mock_reg (MagicMock): Mock argument.
      mock_kb (MagicMock): Mock argument.
      mock_mgr (MagicMock): Mock argument.
      mock_load (MagicMock): Mock argument.
      mock_audit (MagicMock): Mock argument.
  """
  import sys

  from scripts.audit_against_snapshots import main

  mock_audit.return_value = ["error"]

  with patch.object(sys, "argv", ["audit_against_snapshots.py", "--strict"]):
    assert main() == 1


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots_multi")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_failure_not_strict(
  mock_reg: MagicMock, mock_kb: MagicMock, mock_mgr: MagicMock, mock_load: MagicMock, mock_audit: MagicMock
) -> None:
  """Test main function when there are errors and strict mode is off.

  Args:
      mock_reg (MagicMock): Mock argument.
      mock_kb (MagicMock): Mock argument.
      mock_mgr (MagicMock): Mock argument.
      mock_load (MagicMock): Mock argument.
      mock_audit (MagicMock): Mock argument.
  """
  import sys

  from scripts.audit_against_snapshots import main

  mock_audit.return_value = ["error"]

  with patch.object(sys, "argv", ["audit_against_snapshots.py"]):
    assert main() == 0


# --- Merged from test_audit_against_snapshots_extra.py ---


sys.path.insert(0, str(Path("src").resolve()))


def test_load_snapshots_branches() -> None:
  """Docstring."""
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
  """Docstring."""
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
