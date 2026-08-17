"""Tests for audit_against_snapshots.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide access to the module
sys.path.insert(0, str(Path("src").resolve()))
from scripts.audit_against_snapshots import audit_frameworks, load_snapshots


@patch("scripts.audit_against_snapshots.sys")
def test_load_snapshots(mock_sys) -> None:
  """Test loading snapshots from a directory."""
  mock_path_obj = MagicMock()
  mock_file = MagicMock()
  mock_file.name = "torch_v1.0.json"
  mock_path_obj.glob.return_value = [mock_file]

  with patch("builtins.open", new_callable=MagicMock):
    with patch("json.load", return_value={"functions": {"test_api": {}}}):
      snapshots = load_snapshots(mock_path_obj)
      assert set(snapshots.keys()) == {"torch"}
      assert "test_api" in snapshots["torch"]


def test_load_snapshots_empty() -> None:
  """Test loading snapshots empty."""
  from scripts.audit_against_snapshots import load_snapshots

  mock_path_obj = MagicMock()
  mock_path_obj.glob.return_value = []
  assert load_snapshots(mock_path_obj) == {}


def test_audit_frameworks_coverage() -> None:
  """Test auditing framework coverage."""
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
  assert len(errors) == 2

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 1
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
  assert len(errors) == 1
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
  assert len(errors) == 2

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 1
  manager.data = {"flatten": {"variants": {"missing": {}}}}
  assert audit_frameworks(manager, {}) == []


def test_main_block():
  """Test that the main block is callable."""
  from scripts.audit_against_snapshots import main

  assert callable(main)


def test_audit_frameworks_coverage2() -> None:
  """Test another coverage case for audit frameworks."""
  manager = MagicMock()
  manager.data = {
    "flatten": {
      "variants": {
        "mlx": {
          "api": "mlx.flatten",
        }
      }
    }
  }
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0


def test_audit_frameworks() -> None:
  """Test auditing framework definitions against snapshots."""
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
  assert len(errors) == 2

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 1
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
  assert len(errors) == 1
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
  assert len(errors) == 2

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 1
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
  assert "[torch] 'missing_op' maps to hallucinated API: 'torch.missing'" in errors
  assert len(errors) == 2

  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "input"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 1
  manager.data = {"flatten": {"variants": {"mlx": {"api": "mlx.flatten", "args": {"start_dim": "missing_axis"}}}}}
  snapshots = {"mlx": {"mlx.flatten": {"args": [{"name": "kwargs", "kind": "VAR_KEYWORD"}]}}}
  errors = audit_frameworks(manager, snapshots)
  assert len(errors) == 0


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_success(mock_reg, mock_kb, mock_mgr, mock_load, mock_audit) -> None:
  """Test main function when there are no errors."""
  from scripts.audit_against_snapshots import main
  import sys

  mock_audit.return_value = []

  with patch.object(sys, "argv", ["audit_against_snapshots.py"]):
    assert main() == 0


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_failure_strict(mock_reg, mock_kb, mock_mgr, mock_load, mock_audit) -> None:
  """Test main function when there are errors and strict mode is on."""
  from scripts.audit_against_snapshots import main
  import sys

  mock_audit.return_value = ["error"]

  with patch.object(sys, "argv", ["audit_against_snapshots.py", "--strict"]):
    assert main() == 1


@patch("scripts.audit_against_snapshots.audit_frameworks")
@patch("scripts.audit_against_snapshots.load_snapshots")
@patch("scripts.audit_against_snapshots.SemanticsManager")
@patch("scripts.audit_against_snapshots.KnowledgeBaseLoader")
@patch("scripts.audit_against_snapshots.RegistryLoader")
def test_main_failure_not_strict(mock_reg, mock_kb, mock_mgr, mock_load, mock_audit) -> None:
  """Test main function when there are errors and strict mode is off."""
  from scripts.audit_against_snapshots import main
  import sys

  mock_audit.return_value = ["error"]

  with patch.object(sys, "argv", ["audit_against_snapshots.py"]):
    assert main() == 0
