"""Tests for audit_against_snapshots.py."""

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from scripts.snapshots.ast_auditor import audit_frameworks  # noqa: E402
from scripts.snapshots.loader import load_snapshots, load_snapshots_multi  # noqa: E402

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
  """Verify load snapshots empty behavior."""
  from scripts.snapshots.loader import load_snapshots

  mock_path_obj: MagicMock = MagicMock()
  mock_path_obj.glob.return_value = []
  assert load_snapshots(mock_path_obj) == {}


def test_audit_frameworks_coverage() -> None:
  """Verify audit frameworks coverage behavior."""
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
  """Verify main block behavior."""
  from scripts.audit_against_snapshots import main

  assert callable(main)


def test_audit_frameworks_coverage2() -> None:
  """Verify audit frameworks coverage2 behavior."""
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
  """Verify audit frameworks behavior."""
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
  """Verify load snapshots branches behavior."""
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
  """Verify load snapshots multi branches behavior."""
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


def test_flatten_single_framework_edge_cases() -> None:
  """Test edge cases for _flatten_single_framework with non-dict/non-mnemonic items."""
  from scripts.snapshots.loader import _flatten_single_framework

  flat: Dict[str, Dict[str, Any]] = {}
  _flatten_single_framework("hw", ["string_item", {"no_mnemonic": 1}, {"mnemonic": "v_add"}], flat)
  assert "v_add" in flat["hw"]

  _flatten_single_framework("invalid", 12345, flat)
  assert flat["invalid"] == {}


def test_load_snapshots_filename_branches(tmp_path: Path) -> None:
  """Test filename parsing branches in load_snapshots.

  Args:
      tmp_path: Temporary directory fixture.
  """
  snap_dir = tmp_path / "snaps"
  snap_dir.mkdir()

  (snap_dir / "test_sass_exhaustive.json").write_text(json.dumps([{"mnemonic": "FADD"}]))
  (snap_dir / "test_rdna_exhaustive.json").write_text(json.dumps([{"mnemonic": "v_add"}]))
  (snap_dir / "other_exhaustive.json").write_text(json.dumps([{"mnemonic": "CUSTOM"}]))

  result = load_snapshots(snap_dir)
  assert "nvidia_sass" in result
  assert "rdna" in result
  assert "other_exhaustive" in result


def test_load_snapshots_parent_frameworks_exist() -> None:
  """Test load_snapshots_multi when parent snapshots and frameworks directories exist."""
  with patch("pathlib.Path.exists", return_value=True):
    with patch("pathlib.Path.glob", return_value=[]):
      result = load_snapshots_multi(None)
      assert isinstance(result, dict)


def test_audit_frameworks_macro_and_unknown_framework() -> None:
  """Test audit_frameworks filtering for macros, semicolons, ignore_list, and unknown frameworks."""
  manager = MagicMock()
  manager.data = {
    "op1": {
      "variants": {
        "unknown_fw": {"api": "unknown.api"},
        "torch": {"api": "Macro.conv2d"},
        "jax": {"api": "; inline sass"},
        "mlx": {"api": "torch.int64"},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"unknown_fw": {}, "torch": {}, "jax": {}, "mlx": {}}
  errors = audit_frameworks(manager, snapshots)
  assert errors == []


def test_import_grounding_engine_fallback() -> None:
  """Test _resolve_grounding_engine across fallback chains."""
  import importlib

  from scripts.snapshots.loader import _resolve_grounding_engine

  orig_import = importlib.import_module
  mock_cls = MagicMock()
  mock_mod = MagicMock()
  mock_mod.GroundingEngine = mock_cls

  # 1. Success on first module
  def mock_import_1(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute mock_import_1 test helper."""
    if "grounding.engine" in name:
      return mock_mod
    return orig_import(name, *args, **kwargs)

  with patch("importlib.import_module", side_effect=mock_import_1):
    assert _resolve_grounding_engine() == mock_cls

  # 2. First raises ImportError, second succeeds
  call_count = 0

  def mock_import_2(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute mock_import_2 test helper."""
    nonlocal call_count
    if "grounding.engine" in name:
      call_count += 1
      if call_count == 1:
        raise ImportError("No mod1")
      return mock_mod
    return orig_import(name, *args, **kwargs)

  with patch("importlib.import_module", side_effect=mock_import_2):
    assert _resolve_grounding_engine() == mock_cls

  # 3. Both in installed loop fail, sibling path exists and parent loop succeeds
  call_count = 0

  def mock_import_3(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute mock_import_3 test helper."""
    nonlocal call_count
    if "grounding.engine" in name:
      call_count += 1
      if call_count <= 2:
        raise ImportError("No installed mod")
      return mock_mod
    return orig_import(name, *args, **kwargs)

  with patch("importlib.import_module", side_effect=mock_import_3):
    with patch("pathlib.Path.exists", return_value=True):
      assert _resolve_grounding_engine() == mock_cls

  # 4. All fail -> returns None
  def mock_import_4(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute mock_import_4 test helper."""
    if "grounding.engine" in name:
      raise ImportError("None found")
    return orig_import(name, *args, **kwargs)

  with patch("importlib.import_module", side_effect=mock_import_4):
    with patch("pathlib.Path.exists", return_value=True):
      assert _resolve_grounding_engine() is None


def test_audit_frameworks_flax_nnx_non_jnp_branch() -> None:
  """Test audit_frameworks when flax_nnx API does not start with jax.numpy or jnp."""
  manager = MagicMock()
  manager.data = {
    "custom_op": {
      "variants": {
        "flax_nnx": {"api": "flax.nnx.CustomLinear"},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"flax_nnx": {}}
  mock_ge = MagicMock()
  mock_ge.has_symbol.return_value = True
  mock_ge._discover_target_files.return_value = True

  errors = audit_frameworks(manager, snapshots, grounding_engine=mock_ge)
  assert errors == []
  mock_ge.has_symbol.assert_called_with("flax_nnx", "flax.nnx.CustomLinear")


def test_audit_frameworks_flax_nnx_jnp_not_in_jax_snapshot() -> None:
  """Test audit_frameworks when flax_nnx API starts with jax.numpy but is missing in jax snapshot."""
  manager = MagicMock()
  manager.data = {
    "op": {
      "variants": {
        "flax_nnx": {"api": "jax.numpy.missing_fn"},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"flax_nnx": {}, "jax": {}}
  mock_ge = MagicMock()
  mock_ge.has_symbol.return_value = True
  mock_ge._discover_target_files.return_value = True

  errors = audit_frameworks(manager, snapshots, grounding_engine=mock_ge)
  assert errors == []
  mock_ge.has_symbol.assert_called_with("flax_nnx", "jax.numpy.missing_fn")


def test_flatten_single_framework_flax_nnx_prefix() -> None:
  """Test flax_nnx api_path prefix handling."""
  from scripts.snapshots.loader import _flatten_single_framework

  flat: Dict[str, Dict[str, Any]] = {}
  snap = {"categories": {"default": [{"api_path": "flax.nnx.Linear", "name": "Linear"}]}}
  _flatten_single_framework("flax_nnx", snap, flat)
  assert "nnx.Linear" in flat["flax_nnx"]


def test_load_snapshots_multi_optax_shim(tmp_path: Path) -> None:
  """Test load_snapshots_multi handling of optax_shim files.

  Args:
      tmp_path (Path): Temporary directory fixture.
  """
  snap_dir: Path = tmp_path / "snaps"
  snap_dir.mkdir()
  data = {"categories": {"default": [{"name": "adam", "api_path": "optax.adam"}]}}
  (snap_dir / "optax_shim_v0.1.json").write_text(json.dumps(data))

  result: Dict[str, Dict[str, Any]] = load_snapshots_multi([snap_dir])
  assert "optax" in result
  assert "adam" in result["optax"]


def test_audit_frameworks_optax_fallback() -> None:
  """Test audit_frameworks matching optax APIs for jax and flax_nnx."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "adam": {
      "variants": {
        "jax": {"api": "optax.adam", "args": {}},
        "flax_nnx": {"api": "optax.sgd", "args": {}},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "jax": {},
    "flax_nnx": {},
    "optax": {"optax.adam": {"args": []}},
    "optax_shim": {"optax.sgd": {"args": []}},
  }
  errors: List[str] = audit_frameworks(manager, snapshots)
  assert errors == []


def test_audit_frameworks_ignore_args() -> None:
  """Test audit_frameworks ignoring known special arguments and dropped None arguments."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "sum": {
      "variants": {
        "torch": {
          "api": "torch.sum",
          "args": {"input": "input", "dropped_arg": None, "keepdim": "keepdim"},
        }
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "torch": {
      "torch.sum": {
        "args": [{"name": "input", "kind": "POSITIONAL_OR_KEYWORD"}],
      }
    }
  }
  errors: List[str] = audit_frameworks(manager, snapshots)
  assert errors == []


def test_audit_new_targets_integration(tmp_path: Path) -> None:
  """Test integration of array_api, scipy, and safetensors targets in audit.

  Args:
      tmp_path: Temporary directory fixture.
  """
  from scripts.audit_against_snapshots import compute_snapshot_checksums, generate_audit_report, load_snapshots_multi

  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()

  array_api_data = {
    "categories": {
      "elementwise": [{"name": "add", "api_path": "array_api.add", "params": []}],
    }
  }
  (snap_dir / "array_api_v2024.12.json").write_text(json.dumps(array_api_data))

  scipy_data = {
    "categories": {
      "special": [{"name": "erf", "api_path": "scipy.special.erf", "params": []}],
    }
  }
  (snap_dir / "scipy_v1.13.1.json").write_text(json.dumps(scipy_data))

  safetensors_data = {
    "categories": {
      "io": [{"name": "save_file", "api_path": "safetensors.torch.save_file", "params": []}],
    }
  }
  (snap_dir / "safetensors_v0.7.0.json").write_text(json.dumps(safetensors_data))

  snapshots = load_snapshots_multi([snap_dir])
  assert "array_api" in snapshots
  assert "scipy" in snapshots
  assert "safetensors" in snapshots

  checksums = compute_snapshot_checksums([snap_dir])
  assert "array_api_v2024.12.json" in checksums
  assert "scipy_v1.13.1.json" in checksums
  assert "safetensors_v0.7.0.json" in checksums

  manager = MagicMock()
  manager.data = {
    "add": {"variants": {"array_api": {"api": "array_api.add"}}},
    "erf": {"variants": {"scipy": {"api": "scipy.special.erf"}}},
    "save": {"variants": {"safetensors": {"api": "safetensors.torch.save_file"}}},
  }

  report = generate_audit_report(manager, snapshots, errors=[], checksums=checksums)
  assert report["status"] == "pass"
  assert "array_api" in report["targets"]
  assert "scipy" in report["targets"]
  assert "safetensors" in report["targets"]
  assert report["targets"]["array_api"]["mapped_operations"] == 1
  assert report["targets"]["scipy"]["mapped_operations"] == 1
  assert report["targets"]["safetensors"]["mapped_operations"] == 1


def test_flatten_single_framework_operations_branches() -> None:
  """Test _flatten_single_framework branches for operations list handling."""
  from scripts.snapshots.loader import _flatten_single_framework

  flat: Dict[str, Dict[str, Any]] = {}
  snap: Dict[str, Any] = {
    "operations": [
      "not_a_dict",
      {"api_path": "op.path"},
      {"class_name": "OpClass"},
      {"name": "OpName"},
      {"name": "OpWithAliases", "aliases": ["alias_1", "alias_2"]},
      {"name": "OpBadAliases", "aliases": "not_a_list"},
    ]
  }
  _flatten_single_framework("torch", snap, flat)
  assert "op.path" in flat["torch"]
  assert "OpClass" in flat["torch"]
  assert "OpName" in flat["torch"]
  assert "alias_1" in flat["torch"]
  assert "alias_2" in flat["torch"]


def test_load_snapshots_stablehlo_exhaustive(tmp_path: Path) -> None:
  """Test load_snapshots for stablehlo_exhaustive snapshot files.

  Args:
      tmp_path (Path): Temporary directory fixture.
  """
  from scripts.snapshots.loader import load_snapshots

  snap_dir: Path = tmp_path / "snapshots"
  snap_dir.mkdir()
  stablehlo_data: Dict[str, Any] = {"operations": [{"name": "custom_call", "api_path": "stablehlo.custom_call"}]}
  (snap_dir / "stablehlo_exhaustive.json").write_text(json.dumps(stablehlo_data))

  snapshots: Dict[str, Dict[str, Any]] = load_snapshots(snap_dir)
  assert "stablehlo_exhaustive" in snapshots
  assert "stablehlo" in snapshots
  assert "stablehlo.custom_call" in snapshots["stablehlo"]


def test_audit_frameworks_ir_canonical_onnx_exception() -> None:
  """Test audit_frameworks when loading canonical ONNX ops raises an exception."""
  manager: MagicMock = MagicMock()
  manager.data = {
    "add": {
      "variants": {
        "ir": {"api": "Add", "args": {}},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {"ir": {}}

  with patch("ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops", side_effect=RuntimeError("Simulated failure")):
    errors: List[str] = audit_frameworks(manager, snapshots)
    assert errors == []


def test_snapshots_and_audit_remaining_coverage(tmp_path: Path) -> None:
  """Cover remaining branches in audit_against_snapshots, validate_snapshot_grounding, and snapshots submodules.

  Args:
      tmp_path: Temporary directory fixture.
  """
  import importlib
  import runpy

  import scripts.audit_against_snapshots as aas
  import scripts.check_snapshot_regressions as csr
  import scripts.snapshots.ast_auditor as ast_aud
  import scripts.snapshots.loader as snap_loader
  from scripts.snapshots.loader import compute_snapshot_checksums

  # 1. check_snapshot_regressions ImportError branch
  with patch.dict(sys.modules, {"ml_ecosystem_snapshots.api": None, "ml_ecosystem_snapshots.diff": None}):
    importlib.reload(csr)
    assert csr.diff_snapshots is None
  importlib.reload(csr)

  # 2. validate_snapshot_grounding import and __main__
  import scripts.validate_snapshot_grounding as vsg

  assert hasattr(vsg, "main")
  with patch("scripts.audit_against_snapshots.main", return_value=0):
    try:
      runpy.run_module("scripts.validate_snapshot_grounding", run_name="__main__")
    except SystemExit as exc:
      assert exc.code == 0

  # 3. audit_against_snapshots.main branches: importlib.resources error, GroundingEngine None/Error, report_path
  report_file = tmp_path / "audit_report.json"
  with (
    patch("sys.argv", ["audit_against_snapshots.py", "--report", str(report_file)]),
    patch("importlib.resources.files", side_effect=RuntimeError("no pkg")),
    patch.object(aas, "GroundingEngine", side_effect=RuntimeError("ge fail")),
    patch.object(aas, "load_snapshots_multi", return_value={}),
    patch.object(aas, "audit_frameworks", return_value=[]),
    patch.object(aas, "audit_python_ast", return_value=[]),
    patch.object(aas, "audit_inline_snippets", return_value=[]),
    patch.object(aas, "compute_snapshot_checksums", return_value={}),
  ):
    assert aas.main() == 0
    assert report_file.exists()

  with (
    patch("sys.argv", ["audit_against_snapshots.py"]),
    patch.object(aas, "GroundingEngine", None),
    patch.object(aas, "load_snapshots_multi", return_value={}),
    patch.object(aas, "audit_frameworks", return_value=[]),
    patch.object(aas, "audit_python_ast", return_value=[]),
    patch.object(aas, "audit_inline_snippets", return_value=[]),
  ):
    assert aas.main() == 0

  # 4. ast_auditor: GroundingEngine() init failure, has_symbol exception, suggest_closest_symbol branches, non-identifier comma param
  with patch.object(ast_aud, "GroundingEngine", side_effect=RuntimeError("init fail")):
    mgr_empty = MagicMock()
    mgr_empty.data = {
      "op_flax": {"variants": {"flax_nnx": {"api": "jax.numpy.sin"}}},
      "op_ir_bad": {"variants": {"ir": {"api": "UnknownIrOp"}}},
    }
    assert len(audit_frameworks(mgr_empty, {"flax_nnx": {}, "jax": {"jax.numpy.sin": {}}, "ir": {}})) == 1

  bad_ge = MagicMock()
  bad_ge._discover_target_files.side_effect = [True, False]
  bad_ge.has_symbol.side_effect = RuntimeError("sym fail")
  mgr_err = MagicMock()
  mgr_err.data = {"op1": {"variants": {"torch": {"api": "torch.bad_op"}}}}
  errs = audit_frameworks(mgr_err, {"torch": {}}, grounding_engine=bad_ge)
  assert len(errs) == 1

  bad_ge2 = MagicMock()
  bad_ge2._discover_target_files.side_effect = [False, RuntimeError("suggest fail")]
  errs2 = audit_frameworks(mgr_err, {"torch": {}}, grounding_engine=bad_ge2)
  assert len(errs2) == 1

  bad_ge3 = MagicMock()
  bad_ge3._discover_target_files.return_value = True
  bad_ge3.has_symbol.return_value = False
  bad_ge3.suggest_closest_symbol.return_value = ""
  errs3 = audit_frameworks(mgr_err, {"torch": {}}, grounding_engine=bad_ge3)
  assert len(errs3) == 1

  # Non-identifier token in comma-separated param name & non-dict arg in snapshot_args
  mgr_comma = MagicMock()
  mgr_comma.data = {
    "op_c": {
      "variants": {
        "torch": {
          "api": "torch.comma_op",
          "args": {"x": "valid_tok"},
        }
      }
    }
  }
  snaps_comma: Dict[str, Dict[str, Any]] = {
    "torch": {
      "torch.comma_op": {
        "params": ["not_a_dict", {"name": "valid_tok, 123invalid"}],
      }
    }
  }
  assert audit_frameworks(mgr_comma, snaps_comma) == []

  # 5. loader: load_snapshots_multi(None) exception branches & compute_snapshot_checksums branches
  with (
    patch("importlib.resources.files", side_effect=RuntimeError("no res")),
    patch.object(snap_loader, "Path", side_effect=RuntimeError("path fail")),
  ):
    assert load_snapshots_multi(None) == {}

  assert isinstance(compute_snapshot_checksums(None), dict)

  cs_dir = tmp_path / "cs_dir"
  cs_dir.mkdir()
  (cs_dir / "torch_v1_map.json").write_text("{}", encoding="utf-8")
  (cs_dir / "empty_v1.json").write_text("", encoding="utf-8")
  unreadable = cs_dir / "err_v1.json"
  unreadable.write_text("{}", encoding="utf-8")

  orig_read_bytes = Path.read_bytes

  def mock_read_bytes(self_path: Path) -> bytes:
    """Raise OSError for err_v1.json."""
    if self_path.name == "err_v1.json":
      raise OSError("cannot read")
    return orig_read_bytes(self_path)

  with patch.object(Path, "read_bytes", mock_read_bytes):
    cs = compute_snapshot_checksums([cs_dir, tmp_path / "missing_cs_dir"])
    assert "err_v1.json" not in cs
    assert "empty_v1.json" not in cs
