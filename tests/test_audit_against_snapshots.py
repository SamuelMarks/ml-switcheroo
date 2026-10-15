"""Test suite for audit against snapshots."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import scripts.audit_against_snapshots
from scripts.snapshots.ast_auditor import audit_frameworks, extract_api_calls
from scripts.snapshots.loader import load_snapshots


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
  """Fixture to provide a temporary workspace."""
  return tmp_path


def test_extract_api_calls(temp_workspace: Path) -> None:
  """Test extract_api_calls functionality."""
  file_path = temp_workspace / "test.py"

  # Valid syntax
  file_path.write_text("import torch\ntorch.nn.Conv2d(1, 2, 3)")
  calls = extract_api_calls(file_path)
  assert "torch.nn.Conv2d" in calls

  # Valid from import syntax
  file_path.write_text("from torch import nn\nnn.Conv2d(1, 2, 3)")
  calls = extract_api_calls(file_path)
  assert "torch.nn.Conv2d" in calls

  # Invalid syntax
  file_path.write_text("import torch\nbroken [ syntax")
  calls = extract_api_calls(file_path)
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

  snapshots = load_snapshots(snap_dir)
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
        {"mnemonic": "MNEMO_OP"},
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

  snapshots = load_snapshots(snap_dir)
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
        {"mnemonic": "MNEMO_OP"},
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

  # Test ISA and exhaustive filenames
  (dir1 / "rdna_isa.json").write_text(json.dumps({"functions": {"v_add_f32": {}}}))
  (dir1 / "nvidia_sass_isa.json").write_text(json.dumps({"functions": {"FADD": {}}}))
  (dir1 / "custom_exhaustive.json").write_text(json.dumps({"functions": {"custom_op": {}}}))

  snapshots = scripts.audit_against_snapshots.load_snapshots_multi([dir1, dir2, dir3])
  assert "torch" in snapshots
  assert "mlx" in snapshots
  assert "rdna" in snapshots
  assert "nvidia_sass" in snapshots
  assert "custom_exhaustive" in snapshots


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

  errors = audit_frameworks(mock_manager, snapshots)
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

  errors = audit_frameworks(mock_manager, snapshots)
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

  snapshots: dict[str, dict[str, Any]] = {"torch": {"torch.nn.functional.relu": {}}, "mlx": {"mlx.core.relu": {}}}

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

  snapshots: dict[str, dict[str, Any]] = {"torch": {"torch.nn.Linear": {}}, "mlx": {"mlx.core.valid": {}}}

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
  assert extract_api_calls(rel_file) == set()

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
  flat1 = load_snapshots(snap_dir)
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
