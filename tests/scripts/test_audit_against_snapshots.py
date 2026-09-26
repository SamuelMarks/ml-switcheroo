"""Tests for audit_against_snapshots.py."""

import json
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


def test_flatten_single_framework_edge_cases() -> None:
  """Test edge cases for _flatten_single_framework with non-dict/non-mnemonic items."""
  from scripts.audit_against_snapshots import _flatten_single_framework

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
  from scripts.audit_against_snapshots import _resolve_grounding_engine

  orig_import = importlib.import_module
  mock_cls = MagicMock()
  mock_mod = MagicMock()
  mock_mod.GroundingEngine = mock_cls

  # 1. Success on first module
  def mock_import_1(name: str, *args: Any, **kwargs: Any) -> Any:
    if "grounding.engine" in name:
      return mock_mod
    return orig_import(name, *args, **kwargs)

  with patch("importlib.import_module", side_effect=mock_import_1):
    assert _resolve_grounding_engine() == mock_cls

  # 2. First raises ImportError, second succeeds
  call_count = 0

  def mock_import_2(name: str, *args: Any, **kwargs: Any) -> Any:
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
  from scripts.audit_against_snapshots import _flatten_single_framework

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
  from scripts.audit_against_snapshots import (
    compute_snapshot_checksums,
    generate_audit_report,
    load_snapshots_multi,
  )

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
