"""Tests for audit_against_snapshots script."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock
import json

import pytest

import scripts.audit_against_snapshots


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
  """Fixture to provide a temporary workspace."""
  return tmp_path


def test_extract_api_calls(temp_workspace: Path) -> None:
  """Test extract_api_calls functionality."""
  file_path = temp_workspace / "test.py"

  # Valid syntax
  file_path.write_text("import torch\ntorch.nn.Conv2d(1, 2, 3)")
  calls = scripts.audit_against_snapshots.extract_api_calls(file_path)
  assert "torch.nn.Conv2d" in calls

  # Valid from import syntax
  file_path.write_text("from torch import nn\nnn.Conv2d(1, 2, 3)")
  calls = scripts.audit_against_snapshots.extract_api_calls(file_path)
  assert "torch.nn.Conv2d" in calls

  # Invalid syntax
  file_path.write_text("import torch\nbroken [ syntax")
  calls = scripts.audit_against_snapshots.extract_api_calls(file_path)
  assert not calls


def test_load_snapshots(temp_workspace: Path) -> None:
  """Test load_snapshots functionality."""
  snap_dir = temp_workspace / "snapshots"
  snap_dir.mkdir()

  # Needs to match the snapshot data format script expects: {"categories": {"some_cat": [{"name": "api"}]}}
  data = {"categories": {"my_cat": [{"name": "torch.nn.Conv2d", "params": [{"name": "in_channels"}]}]}}
  (snap_dir / "torch_v1.0.0.json").write_text(json.dumps(data))

  # File to ignore
  (snap_dir / "torch_map.json").write_text("{}")

  # Not a json
  (snap_dir / "ignore.txt").write_text("ignore me")

  # Test _vunknown
  (snap_dir / "unknown_vunknown.json").write_text("{}")

  # Test larger snapshot replacement
  (snap_dir / "torch_v2.0.0.json").write_text(
    json.dumps({"name": "torch.nn.Conv2d", "args": [{"name": "in_channels"}, {"name": "larger"}]})
  )

  snapshots = scripts.audit_against_snapshots.load_snapshots(snap_dir)
  assert "torch" in snapshots
  assert "torch.nn.Conv2d" in snapshots["torch"]


def test_load_snapshots_formats(temp_workspace: Path) -> None:
  """Test load_snapshots various internal formats."""
  snap_dir = temp_workspace / "snapshots2"
  snap_dir.mkdir()

  data = {
    "categories": {
      "cat1": [
        {"api_path": "torch.nn.Linear", "aliases": ["torch.nn.modules.linear.Linear"]},
        {"name": "torch.nn.Conv2d"},
      ],
      "cat2": {"torch.nn.ReLU": {}},
      "cat3": "ignore this",
    },
    "functions": {"torch.add": {}},
    "classes": {"torch.Tensor": {}},
    "other": {"args": {}},
    "ignore_me": "value",
  }
  (snap_dir / "torch_v1.0.0.json").write_text(json.dumps(data))

  snapshots = scripts.audit_against_snapshots.load_snapshots(snap_dir)
  assert "torch.nn.Linear" in snapshots["torch"]
  assert "torch.nn.modules.linear.Linear" in snapshots["torch"]
  assert "torch.nn.ReLU" in snapshots["torch"]
  assert "torch.add" in snapshots["torch"]
  assert "torch.Tensor" in snapshots["torch"]
  assert "other" in snapshots["torch"]
  assert "ignore_me" not in snapshots["torch"]


def test_load_snapshots_multi_formats(temp_workspace: Path) -> None:
  """Test load_snapshots_multi various internal formats."""
  dir1 = temp_workspace / "dir1"
  dir1.mkdir()

  data = {
    "categories": {
      "cat1": [
        {"api_path": "torch.nn.Linear", "aliases": ["torch.nn.modules.linear.Linear"]},
        {"name": "torch.nn.Conv2d"},
      ],
      "cat2": {"torch.nn.ReLU": {}},
      "cat3": "ignore this",
    },
    "functions": {"torch.add": {}},
    "classes": {"torch.Tensor": {}},
    "other": {"args": {}},
    "ignore_me": "value",
  }
  (dir1 / "torch_v1.0.0.json").write_text(json.dumps(data))

  snapshots = scripts.audit_against_snapshots.load_snapshots_multi([dir1])
  assert "torch.nn.Linear" in snapshots["torch"]
  assert "torch.nn.modules.linear.Linear" in snapshots["torch"]
  assert "torch.nn.ReLU" in snapshots["torch"]
  assert "torch.add" in snapshots["torch"]
  assert "torch.Tensor" in snapshots["torch"]
  assert "other" in snapshots["torch"]
  assert "ignore_me" not in snapshots["torch"]


def test_load_snapshots_multi(temp_workspace: Path) -> None:
  """Test load_snapshots_multi functionality."""
  dir1 = temp_workspace / "dir1"
  dir1.mkdir()
  dir2 = temp_workspace / "dir2"
  dir2.mkdir()
  dir3 = temp_workspace / "dir3"  # Doesn't exist

  (dir1 / "torch_v1.0.0.json").write_text(json.dumps({"functions": {"torch.nn.Conv2d": {}}}))
  (dir2 / "mlx_v0.1.0.json").write_text(json.dumps({"functions": {"mlx.nn.Conv2d": {}}}))

  # Ignore map
  (dir1 / "torch_map.json").write_text("{}")

  # Test _vunknown
  (dir1 / "unknown_vunknown.json").write_text("{}")

  # Test larger snapshot replacement
  (dir2 / "torch_v2.0.0.json").write_text(json.dumps({"functions": {"torch.nn.Conv2d": {}, "torch.larger": {}}}))

  snapshots = scripts.audit_against_snapshots.load_snapshots_multi([dir1, dir2, dir3])
  assert "torch" in snapshots
  assert "mlx" in snapshots


def test_audit_frameworks() -> None:
  """Test audit_frameworks functionality."""
  mock_manager = MagicMock()
  mock_manager.data = {
    "Conv2d": {
      "variants": {
        "torch": {"api": "torch.nn.Conv2d", "args": {"std_arg": "in_channels"}},
        "broken": {"api": "broken.api", "args": {"std_arg": "in_channels"}},
        "torch_hallucinated_api": {"api": "torch.hallucinated", "args": {}},
        "torch_hallucinated_arg": {"api": "torch.nn.Conv2d", "args": {"std_arg": "hallucinated_arg"}},
        "torch_missing_arg": {"api": "torch.nn.Linear", "args": {}},
        "torch_macro": {"api": "torch.nn.Linear", "macro_template": "...", "args": {}},
      }
    },
    "MissingVar": {},
  }

  snapshots = {
    "torch": {
      "torch.nn.Conv2d": {"params": [{"name": "in_channels"}]},
      "torch.nn.Linear": {"params": [{"name": "in_features"}]},
    },
    "broken": {"broken.api": {"args": [{"name": "in_channels"}]}},
  }

  errors = scripts.audit_against_snapshots.audit_frameworks(mock_manager, snapshots)
  # the script keys on the fw_name, so "torch_hallucinated_api" fails at fw_name not in ["mlx", "torch"...]
  # let's modify the manager data to test this properly
  mock_manager.data = {
    "Conv2d": {
      "variants": {
        "torch": {"api": "torch.nn.Conv2d", "args": {"std_arg": "in_channels"}},
        "mlx": {"api": "mlx.core.hallucinated", "args": {}},  # hallucinated api
        "jax": {"api": "torch.nn.Conv2d", "args": {"std_arg": "hallucinated_arg"}},  # hallucinated arg
        "tensorflow": {"api": "tensorflow.random.normal", "args": {"arg1": "provided"}},  # missing required args
        "numpy": {"api": "torch.nn.Linear", "macro_template": "...", "args": {}},  # ignores required args if macro
        "flax": {"api": "", "args": {}},  # No API
        "keras": {"api": "torch.nn.Conv2d", "args": {}},  # ignored API
        "paxml": {"api": "ignore.abs", "args": {}},  # Framework not strictly checked for hallucinated API
        "stablehlo": {"api": "numpy.add", "args": {}},  # triggers ignore list in missing required args!
        "rdna": {"api": ";comment_api", "args": {}},  # comment api check
        "torch_ignore": {"api": "torch.relu", "args": {}},  # triggers ignore list in hallucinated API
        "jax_kwargs": {
          "api": "jax.nn.Conv2d",
          "args": {"std_arg": "hallucinated_arg"},
        },  # hallucinated arg with kwargs in api
        "unknown_fw": {
          "api": "unknown.api",
          "args": {"std_arg": "hallucinated_arg"},
        },  # fw not strictly checked for hallucinated arg AND hallucinated api
        "paxml2": {
          "api": "ignore.abs",
          "args": {"arg": "hallucinated"},
        },  # Framework not strictly checked for hallucinated API AND hallucinated arg
        "torch_ignore_args": {"api": "numpy.add", "args": {}},  # triggers ignore list in missing args correctly mapped!
        "torch_ignore_arg_match": {
          "api": "numpy.add",
          "args": {"ignored_arg": "test"},
        },  # strictly checked fw but ignores due to ignore_list
        "keras2": {"api": ";comment_api", "args": {}},  # strictly checked fw but ignores due to ;
        "torch_ignore2": {"api": "numpy.add", "args": {}},  # strictly checked, in ignore list, not in snapshot
        "unknown_fw2": {"api": "completely_missing", "args": {}},  # completely missing api for unknown fw
        "numpy_ignore_missing": {"api": "numpy.add", "args": {}},  # missing arg but in ignore list
        "test_356_false": {"api": "missing", "args": {}},
      }
    }
  }

  snapshots = {
    "torch": {
      "torch.nn.Conv2d": {"params": [{"name": "in_channels"}]},
      "torch.nn.Linear": {"params": [{"name": "in_features"}]},
      "numpy.add": {"params": [{"name": "x"}, {"name": "y"}]},  # added for torch_ignore_args to match strictly checked fw
    },
    "mlx": {"mlx.core.valid": {}},
    "jax": {"torch.nn.Conv2d": {"params": [{"name": "in_channels"}]}},
    "tensorflow": {"tensorflow.random.normal": {"params": [{"name": "provided"}, {"name": "random_missing_param"}]}},
    "numpy": {
      "torch.nn.Linear": {"params": [{"name": "random_missing_param"}]},
      "numpy.add": {"params": [{"name": "my_missing_arg"}]},
    },
    "stablehlo": {"numpy.add": {"params": [{"name": "x", "kind": "VAR_KEYWORD"}, {"name": "y"}]}},
    "torch_ignore": {},
    "torch_ignore_arg_match": {"numpy.add": {"params": [{"name": "x"}]}},
    "keras": {},
    "jax_kwargs": {"jax.nn.Conv2d": {"params": [{"name": "kwargs"}]}},
    "unknown_fw": {"something": {}},
    "unknown_fw2": {"something": {}},
    "torch_ignore_args": {"numpy.add": {"params": [{"name": "my_missing_arg"}]}},
    "flax": {},
    "paxml": {"something": {}},
    "rdna": {"something": {}},
    "test_356_false": {"something": {}},
  }

  errors = scripts.audit_against_snapshots.audit_frameworks(mock_manager, snapshots)
  print("ERRORS:", errors)
  assert any("hallucinated API" in err for err in errors if "mlx" in err)
  assert any("hallucinated argument" in err for err in errors if "jax" in err)

  # Check that missing args was triggered.
  assert any("missing required arguments" in err for err in errors if "tensorflow" in err)


def test_audit_inline_snippets() -> None:
  """Test audit_inline_snippets functionality."""
  mock_manager = MagicMock()
  mock_manager.data = {
    "ReLU": {
      "variants": {
        "torch": {"macro_template": "torch.nn.functional.relu(x)"},
        "mlx": {"macro_template": "mlx.core.not_exist(x)"},
        "broken_ast": {"macro_template": "broken ["},
        "ignore_list": {"macro_template": "numpy.add(x)"},
        "unsupported_fw": {"macro_template": "unsupported.api(x)"},
      }
    }
  }

  snapshots = {"torch": {"torch.nn.functional.relu": {}}, "mlx": {"mlx.core.relu": {}}}

  errors = scripts.audit_against_snapshots.audit_inline_snippets(mock_manager, snapshots)
  assert any("mlx.core.not_exist" in err for err in errors)


def test_audit_python_ast(temp_workspace: Path) -> None:
  """Test audit_python_ast functionality."""
  src_dir = temp_workspace / "src"
  src_dir.mkdir()
  (src_dir / "test.py").write_text(
    "import torch\nimport mlx\ntorch.nn.Linear()\nmlx.core.broken()\nimport numpy\nnumpy.ndarray()\nimport unknown\nunknown.module.call()\nimport tensorflow\ntensorflow.random.normal()"
  )

  # Dir doesn't exist
  bad_dir = temp_workspace / "bad"

  snapshots = {"torch": {"torch.nn.Linear": {}}, "mlx": {"mlx.core.valid": {}}}

  errors = scripts.audit_against_snapshots.audit_python_ast([src_dir, bad_dir], snapshots)
  assert any("mlx.core.broken" in err for err in errors)


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Setup a fake workspace and chdir to it."""
  snap_dir = tmp_path / "ml-framework-snapshots" / "src" / "ml_framework_snapshots" / "snapshots"
  snap_dir.mkdir(parents=True)
  (tmp_path / "ml-compiler-snapshots").mkdir()

  src_dir = tmp_path / "src" / "ml_switcheroo"
  (src_dir / "frameworks").mkdir(parents=True)
  (src_dir / "plugins").mkdir(parents=True)

  monkeypatch.chdir(tmp_path)
  return tmp_path


@patch("sys.argv", ["audit_against_snapshots.py"])
def test_main_execution() -> None:
  """Test module execution block."""
  source_code = open("scripts/audit_against_snapshots.py").read()

  with patch("sys.exit") as mock_exit:
    with patch.object(scripts.audit_against_snapshots, "__name__", "__main__"):
      code = compile(source_code, "scripts/audit_against_snapshots.py", "exec")
      exec(code, scripts.audit_against_snapshots.__dict__)
    mock_exit.assert_called()


def test_main_success(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main success branch."""
  with patch("sys.argv", ["audit_against_snapshots.py"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = []
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 0


def test_main_failure(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main failure branch."""
  with patch("sys.argv", ["audit_against_snapshots.py", "--strict"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = ["error 1"]
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 1


def test_main_failure_not_strict(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main failure branch not strict."""
  with patch("sys.argv", ["audit_against_snapshots.py"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = ["error 1"]
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 0


def test_audit_against_snapshots_extra_branches(temp_workspace: Path) -> None:
  """Test relative imports, alias prefixes, ignore_args, and fallback loader."""
  # 1. extract_api_calls with relative import (branch 37->32)
  rel_file = temp_workspace / "rel_test.py"
  rel_file.write_text("from . import local_mod\nlocal_mod.foo()\n")
  assert scripts.audit_against_snapshots.extract_api_calls(rel_file) == set()

  # 2. load_snapshots and load_snapshots_multi alias prefixes (lines 236, 238, 240, 301, 303, 305)
  snap_dir = temp_workspace / "prefix_snaps"
  snap_dir.mkdir()
  snap_data = {
    "categories": {
      "cat": [
        {"api_path": "jax.numpy.sin"},
        {"api_path": "mlx.core.cos"},
        {"api_path": "torch.nn.functional.relu"},
      ]
    }
  }
  (snap_dir / "all_v1.json").write_text(json.dumps(snap_data))
  # Smaller snapshots to guarantee 288->279 branch is hit regardless of glob order
  (snap_dir / "all_v2.json").write_text(json.dumps({"categories": {}}))
  (snap_dir / "all_v3.json").write_text(json.dumps({}))
  flat1 = scripts.audit_against_snapshots.load_snapshots(snap_dir)
  assert "jnp.sin" in flat1["all"]
  assert "mx.cos" in flat1["all"]
  assert "F.relu" in flat1["all"]

  flat2 = scripts.audit_against_snapshots.load_snapshots_multi([snap_dir])
  assert "jnp.sin" in flat2["all"]
  assert "mx.cos" in flat2["all"]
  assert "F.relu" in flat2["all"]

  # 3. load_snapshots_multi default snapshot_dirs (lines 269-273)
  # Success branch
  flat_default = scripts.audit_against_snapshots.load_snapshots_multi(None)
  assert isinstance(flat_default, dict)
  # Exception branch
  with patch("scripts.audit_against_snapshots.importlib.resources.files", side_effect=Exception("no resources")):
    flat_except = scripts.audit_against_snapshots.load_snapshots_multi(None)
    assert isinstance(flat_except, dict)

  # Test _flatten_single_framework existing fw branch and non-dict branch
  flat_test: Dict[str, Dict[str, Any]] = {"existing": {"old": 1}}
  scripts.audit_against_snapshots._flatten_single_framework("existing", {"categories": {}}, flat_test)
  assert "old" in flat_test["existing"]
  scripts.audit_against_snapshots._flatten_single_framework("non_dict", 12345, flat_test)
  assert flat_test["non_dict"] == {}

  # Test load_snapshots filenames: sass, rdna, and other
  snap_fn_dir = temp_workspace / "fn_snaps"
  snap_fn_dir.mkdir()
  (snap_fn_dir / "sass_exhaustive.json").write_text(json.dumps([{"mnemonic": "FADD"}]))
  (snap_fn_dir / "rdna_exhaustive.json").write_text(json.dumps([{"mnemonic": "v_add"}]))
  (snap_fn_dir / "other_exhaustive.json").write_text(json.dumps([{"mnemonic": "custom"}]))
  fn_res = scripts.audit_against_snapshots.load_snapshots(snap_fn_dir)
  assert "nvidia_sass" in fn_res and "rdna" in fn_res and "other_exhaustive" in fn_res

  # Test load_snapshots_multi parent_snap branches (false conditions and exception)
  with patch("scripts.audit_against_snapshots.importlib.resources.files", side_effect=Exception("no resources")):
    with patch("pathlib.Path.exists", return_value=False):
      flat_no_parent = scripts.audit_against_snapshots.load_snapshots_multi(None)
      assert isinstance(flat_no_parent, dict)

    with patch("scripts.audit_against_snapshots.Path.resolve", side_effect=Exception("resolve fail")):
      flat_resolve_fail = scripts.audit_against_snapshots.load_snapshots_multi(None)
      assert isinstance(flat_resolve_fail, dict)

  # Test audit_frameworks with unknown fw and ignored/macro apis
  mock_mgr_extra = MagicMock()
  mock_mgr_extra.data = {
    "op1": {
      "variants": {
        "unknown_fw": {"api": "unknown.api"},
        "torch": {"api": "Macro.conv2d"},
        "jax": {"api": "; inline sass"},
        "mlx": {"api": "torch.int64"},
      }
    },
    "op_hallucinated": {
      "variants": {
        "torch": {"api": "torch.nonexistent_function"},
      }
    },
  }
  snap_extra: Dict[str, Dict[str, Any]] = {"unknown_fw": {}, "torch": {}, "jax": {}, "mlx": {}}
  extra_errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr_extra, snap_extra)
  assert any("maps to hallucinated API" in e for e in extra_errors)

  # 4. audit_frameworks ignore_args (line 433) and ignore_list in missing args (branch 507->397)
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "Sum": {
      "variants": {
        "torch": {
          "api": "torch.sum",
          "args": {"input": "input", "dim": "dim"},
        }
      }
    },
    "Float": {
      "variants": {
        "torch": {
          "api": "torch.float32",
          "args": {"x": "req_arg1"},
        }
      }
    },
  }
  snapshots = {
    "torch": {
      "torch.sum": {
        "params": [{"name": "input"}],
      },
      "torch.float32": {
        "params": [
          {"name": "req_arg1", "kind": "POSITIONAL_OR_KEYWORD"},
          {"name": "req_arg2", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
      },
    }
  }
  errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr, snapshots)
  assert not any("torch.sum" in e for e in errors)
  assert not any("torch.float32" in e for e in errors)

  # 5. main importlib.resources.files exception (lines 534-535)
  with patch("sys.argv", ["audit_against_snapshots.py"]):
    with (
      patch("scripts.audit_against_snapshots.SemanticsManager"),
      patch("scripts.audit_against_snapshots.RegistryLoader"),
    ):
      with patch.object(scripts.audit_against_snapshots.importlib.resources, "files", side_effect=Exception("fail")):
        with patch("scripts.audit_against_snapshots.load_snapshots_multi", return_value={}):
          with patch("scripts.audit_against_snapshots.audit_frameworks", return_value=[]):
            with patch("scripts.audit_against_snapshots.audit_python_ast", return_value=[]):
              with patch("scripts.audit_against_snapshots.audit_inline_snippets", return_value=[]):
                assert scripts.audit_against_snapshots.main() == 0


def test_generate_audit_report_and_cli(tmp_path: Path) -> None:
  """Test generate_audit_report function and report export CLI flag."""
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "op1": {"variants": {"torch": {"api": "torch.add"}, "keras": {"api": "keras.ops.add"}}},
    "op2": {"variants": {"rdna": {"api": "v_add_f32"}}},
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "torch": {"torch.add": {}},
    "keras": {"keras.ops.add": {}},
    "rdna": {"v_add_f32": {}},
  }
  errors = ["[torch] Some test error"]
  report = scripts.audit_against_snapshots.generate_audit_report(mock_mgr, snapshots, errors)
  assert report["total_operations"] == 2
  assert report["total_errors"] == 1
  assert report["status"] == "fail"
  assert report["targets"]["torch"]["mapped_operations"] == 1
  assert report["targets"]["torch"]["errors"] == 1
  assert report["targets"]["torch"]["status"] == "mismatched"
  assert report["targets"]["keras"]["mapped_operations"] == 1
  assert report["targets"]["keras"]["status"] == "valid"

  report_file = tmp_path / "out_report.json"
  with patch("sys.argv", ["audit_against_snapshots.py", "--report", str(report_file)]):
    with (
      patch("scripts.audit_against_snapshots.SemanticsManager", return_value=mock_mgr),
      patch("scripts.audit_against_snapshots.KnowledgeBaseLoader"),
      patch("scripts.audit_against_snapshots.RegistryLoader"),
      patch("scripts.audit_against_snapshots.load_snapshots_multi", return_value=snapshots),
      patch("scripts.audit_against_snapshots.audit_frameworks", return_value=[]),
      patch("scripts.audit_against_snapshots.audit_python_ast", return_value=[]),
      patch("scripts.audit_against_snapshots.audit_inline_snippets", return_value=[]),
    ):
      assert scripts.audit_against_snapshots.main() == 0
      assert report_file.exists()
      saved = json.loads(report_file.read_text())
      assert saved["status"] == "pass"


def test_audit_frameworks_ungrounded_sass_instruction() -> None:
  """Test that audit_frameworks flags an ungrounded SASS instruction."""
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "fake_op": {"variants": {"nvidia_sass": {"api": "HALLUCINATED_SASS_OP"}}},
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "nvidia_sass": {"FFMA": {}, "FADD": {}},
  }
  errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr, snapshots)
  assert any("HALLUCINATED_SASS_OP" in err and "[nvidia_sass]" in err for err in errors)


def test_audit_frameworks_ungrounded_rdna_instruction() -> None:
  """Test that audit_frameworks flags an ungrounded RDNA instruction."""
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "fake_op": {"variants": {"rdna": {"api": "v_fake_hallucinated_inst"}}},
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "rdna": {"v_fmac_f32": {}, "v_add_f32": {}},
  }
  errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr, snapshots)
  assert any("v_fake_hallucinated_inst" in err and "[rdna]" in err for err in errors)


def test_audit_frameworks_with_framework_filter() -> None:
  """Test that audit_frameworks respects the framework parameter filter."""
  mock_mgr = MagicMock()
  mock_mgr.data = {
    "op1": {
      "variants": {
        "rdna": {"api": "v_fake_rdna"},
        "nvidia_sass": {"api": "HALLUCINATED_SASS"},
      }
    }
  }
  snapshots: Dict[str, Dict[str, Any]] = {
    "rdna": {"v_add_f32": {}},
    "nvidia_sass": {"FADD": {}},
  }
  # Only audit rdna
  rdna_errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr, snapshots, framework="rdna")
  assert len(rdna_errors) == 1
  assert "[rdna]" in rdna_errors[0]
  assert "[nvidia_sass]" not in rdna_errors[0]

  # Only audit nvidia_sass
  sass_errors = scripts.audit_against_snapshots.audit_frameworks(mock_mgr, snapshots, framework="nvidia_sass")
  assert len(sass_errors) == 1
  assert "[nvidia_sass]" in sass_errors[0]
  assert "[rdna]" not in sass_errors[0]


def test_main_cli_framework_filter() -> None:
  """Test CLI argument parsing for --framework option."""
  with patch("sys.argv", ["audit_against_snapshots.py", "--framework", "rdna"]):
    with (
      patch("scripts.audit_against_snapshots.SemanticsManager"),
      patch("scripts.audit_against_snapshots.KnowledgeBaseLoader"),
      patch("scripts.audit_against_snapshots.RegistryLoader"),
      patch("scripts.audit_against_snapshots.load_snapshots_multi", return_value={"rdna": {}}),
      patch("scripts.audit_against_snapshots.audit_frameworks", return_value=[]) as mock_audit,
      patch("scripts.audit_against_snapshots.audit_python_ast", return_value=[]),
      patch("scripts.audit_against_snapshots.audit_inline_snippets", return_value=[]),
    ):
      assert scripts.audit_against_snapshots.main() == 0
      mock_audit.assert_called_once()
      assert mock_audit.call_args[1].get("framework") == "rdna"


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test __main__ execution block for audit_against_snapshots."""
  import runpy

  monkeypatch.setattr("sys.argv", ["audit_against_snapshots.py"])
  with patch("scripts.audit_against_snapshots.main", return_value=0):
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.audit_against_snapshots", run_name="__main__")
    assert excinfo.value.code == 0
