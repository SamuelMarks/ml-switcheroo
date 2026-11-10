"""Test suite for audit against snapshots."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.audit_against_snapshots
from scripts.snapshots.ast_auditor import audit_frameworks, extract_api_calls
from scripts.snapshots.loader import load_snapshots


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
  """Fixture to provide a temporary workspace.

  Args:
      tmp_path: Pytest temporary directory path fixture.

  Returns:
      Configured `temp_workspace` fixture value (Path).
  """
  return tmp_path


def test_extract_api_calls(temp_workspace: Path) -> None:
  """Test extract_api_calls functionality.

  Args:
      temp_workspace: The temp workspace (Path) for test extract api calls.
  """
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
  """Test load_snapshots functionality.

  Args:
      temp_workspace: The temp workspace (Path) for test load snapshots.
  """
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
  """Test load_snapshots various internal formats.

  Args:
      temp_workspace: The temp workspace (Path) for test load snapshots formats.
  """
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
  """Test load_snapshots_multi various internal formats.

  Args:
      temp_workspace: The temp workspace (Path) for test load snapshots multi formats.
  """
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
  """Test load_snapshots_multi functionality.

  Args:
      temp_workspace: The temp workspace (Path) for test load snapshots multi.
  """
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

  snapshots: dict[str, RuntimeTensorOrValue] = {
    "torch": {"torch.nn.functional.relu": {}},
    "mlx": {"mlx.core.relu": {}},
  }

  errors = scripts.audit_against_snapshots.audit_inline_snippets(mock_manager, snapshots)
  assert any("mlx.core.not_exist" in err for err in errors)


def test_audit_python_ast(temp_workspace: Path) -> None:
  """Test audit_python_ast functionality.

  Args:
      temp_workspace: The temp workspace (Path) for test audit python ast.
  """
  src_dir = temp_workspace / "src"
  src_dir.mkdir()
  (src_dir / "test.py").write_text(
    "import torch\nimport mlx\ntorch.nn.Linear()\nmlx.core.broken()\nimport numpy\nnumpy.ndarray()\nimport unknown\nunknown.module.call()\nimport tensorflow\ntensorflow.random.normal()"
  )

  # Dir doesn't exist
  bad_dir = temp_workspace / "bad"

  snapshots: dict[str, RuntimeTensorOrValue] = {
    "torch": {"torch.nn.Linear": {}},
    "mlx": {"mlx.core.valid": {}},
  }

  errors = scripts.audit_against_snapshots.audit_python_ast([src_dir, bad_dir], snapshots)
  assert any("mlx.core.broken" in err for err in errors)


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Setup a fake workspace and chdir to it.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.

  Returns:
      Configured `setup_workspace` fixture value (Path).
  """
  snap_dir = tmp_path / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
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
  import runpy

  with patch("sys.exit") as mock_exit:
    runpy.run_path("scripts/audit_against_snapshots.py", run_name="__main__")
    mock_exit.assert_called()


def test_main_success(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main success branch.

  Args:
      setup_workspace: The setup workspace (Path) for test main success.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  with patch("sys.argv", ["audit_against_snapshots.py"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = []
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 0


def test_main_failure(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main failure branch.

  Args:
      setup_workspace: The setup workspace (Path) for test main failure.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  with patch("sys.argv", ["audit_against_snapshots.py", "--strict"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = ["error 1"]
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 1


def test_main_failure_not_strict(setup_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main failure branch not strict.

  Args:
      setup_workspace: The setup workspace (Path) for test main failure not strict.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  with patch("sys.argv", ["audit_against_snapshots.py"]):
    with patch("scripts.audit_against_snapshots.audit_frameworks") as mock_audit:
      with patch("scripts.audit_against_snapshots.audit_python_ast") as mock_ast:
        with patch("scripts.audit_against_snapshots.audit_inline_snippets") as mock_snip:
          mock_audit.return_value = ["error 1"]
          mock_ast.return_value = []
          mock_snip.return_value = []

          assert scripts.audit_against_snapshots.main() == 0


def test_audit_against_snapshots_extra_branches(temp_workspace: Path) -> None:
  """Test relative imports, alias prefixes, ignore_args, and fallback loader.

  Args:
      temp_workspace: The temp workspace (Path) for test audit against snapshots extra branches.
  """
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


def test_main_strict_empty_snapshots(setup_workspace: Path) -> None:
  """Test main strict failure when no snapshots are loaded.

  Args:
      setup_workspace: The setup workspace (Path) for test main strict empty snapshots.
  """
  with patch("sys.argv", ["audit_against_snapshots.py", "--strict"]):
    with patch("scripts.audit_against_snapshots.load_snapshots_multi", return_value={}):
      assert scripts.audit_against_snapshots.main() == 1


def test_main_strict_missing_framework(setup_workspace: Path, tmp_path: Path) -> None:
  """Test main strict failure when requested --framework is not in loaded snapshots.

  Args:
      setup_workspace: The setup workspace (Path) for test main strict missing framework.
      tmp_path: Pytest temporary directory path fixture.
  """
  with patch("sys.argv", ["audit_against_snapshots.py", "--strict", "--framework", "nonexistent_fw"]):
    with patch("scripts.audit_against_snapshots.load_snapshots_multi", return_value={"torch": {"torch.add": {}}}):
      assert scripts.audit_against_snapshots.main() == 1


def test_loader_and_ast_auditor_edge_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Cover remaining defensive branches in `scripts.snapshots.loader` and `scripts.snapshots.ast_auditor`.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  import sys
  from types import ModuleType

  from pydantic import JsonValue

  from ml_switcheroo.semantics.manager import SemanticsManager
  from scripts.snapshots.ast_auditor import audit_frameworks, audit_inline_snippets
  from scripts.snapshots.loader import (
    GroundingEngineProtocol,
    _augment_framework_aliases,
    _flatten_single_framework,
    _hydrate_ir_snapshot,
    _normalize_snapshot_entry,
    _resolve_grounding_engine,
    load_snapshots,
  )

  # 1. _normalize_snapshot_entry edge branches (82->81, 84->81, 98->97, 100->97, 130, 153-154)
  item: dict[str, JsonValue] = {
    "docstring": "fn(a)",
    "params": [
      "not_a_dict",
      {"name": "req_param", "default": None, "kind": "POSITIONAL_OR_KEYWORD"},
    ],
    "overloads": [
      "not_a_dict",
      {"params": "not_a_list"},
      {"params": ["not_a_dict", {"name": "req_param"}]},
    ],
  }
  _normalize_snapshot_entry(item)

  item_colon: dict[str, JsonValue] = {
    "docstring": "fn(opt_colon)\nArgs:\n  opt_colon: int, optional\n",
    "params": [
      {"name": "opt_colon", "default": None, "kind": "POSITIONAL_OR_KEYWORD", "description": "val"},
    ],
  }
  _normalize_snapshot_entry(item_colon)
  assert isinstance(item_colon["params"], list)
  first_param = item_colon["params"][0]
  assert isinstance(first_param, dict)
  assert first_param["default"] == "None"

  # 2. _augment_framework_aliases branches without optional keys (231->exit, 340->342, 342->exit, 351->359)
  np_snap: dict[str, dict[str, JsonValue]] = {"numpy.add": {"api_path": "numpy.add"}}
  _augment_framework_aliases("numpy", np_snap)
  tf_snap: dict[str, dict[str, JsonValue]] = {"tf.add": {"api_path": "tf.add"}}
  _augment_framework_aliases("tensorflow", tf_snap)
  keras_snap: dict[str, dict[str, JsonValue]] = {"keras.add": {"api_path": "keras.add"}}
  _augment_framework_aliases("keras", keras_snap)

  # 3. _resolve_grounding_engine non-callable GroundingEngine attribute (390->395, 401->406)
  class _FakeEngineModule(ModuleType):
    """Module stub with a non-callable GroundingEngine attribute."""

    GroundingEngine: int = 123

  fake_engine_mod = _FakeEngineModule("ml_ecosystem_snapshots.grounding.engine")
  with patch("importlib.import_module", return_value=fake_engine_mod):
    assert _resolve_grounding_engine() is None

  # 4. _hydrate_ir_snapshot sibling_ir_src insertion (423)
  sibling_ir_src = Path(__file__).resolve().parent.parent / "ml-switcheroo-ir" / "src"
  sibling_ir_src_up = Path(__file__).resolve().parent.parent.parent / "ml-switcheroo-ir" / "src"
  monkeypatch.setattr(
    sys,
    "path",
    [p for p in sys.path if p not in (str(sibling_ir_src), str(sibling_ir_src_up))],
  )
  ir_flat: dict[str, dict[str, dict[str, JsonValue]]] = {}
  _hydrate_ir_snapshot(ir_flat)
  assert "ir" in ir_flat

  # 5. _flatten_single_framework non-dict/non-list/non-str branches
  flat: dict[str, dict[str, dict[str, JsonValue]]] = {}
  _flatten_single_framework(
    "custom_fw",
    {
      "categories": "not_a_dict",
      "functions": "not_a_dict",
      "classes": "not_a_dict",
      "operations": "not_a_list",
    },
    flat,
  )
  _flatten_single_framework(
    "custom_fw",
    {
      "categories": {
        "list_cat": ["not_a_dict", {"api_path": "custom_fw.op", "aliases": [123, "custom_fw.alias"]}],
        "dict_cat": {"bad_k": "not_a_dict", "good_k": {"api_path": "custom_fw.good"}},
      },
      "functions": {"bad_fn": "not_a_dict"},
      "classes": {"bad_cls": "not_a_dict"},
      "operations": [{"api_path": "custom_fw.op2", "aliases": [123]}],
    },
    flat,
  )
  assert "custom_fw.alias" in flat["custom_fw"]

  # 6. load_snapshots with stablehlo_exhaustive.json and pax_v1.4.0.json (lines 640, 642)
  shlo_dir = tmp_path / "shlo_snaps"
  shlo_dir.mkdir()
  (shlo_dir / "stablehlo_exhaustive.json").write_text(
    json.dumps({"operations": [{"name": "stablehlo.add", "api_path": "stablehlo.add"}]}),
    encoding="utf-8",
  )
  (shlo_dir / "pax_v1.4.0.json").write_text(
    json.dumps({"operations": [{"name": "praxis.base_layer", "api_path": "praxis.base_layer"}]}),
    encoding="utf-8",
  )
  shlo_loaded = load_snapshots(shlo_dir)
  assert "stablehlo" in shlo_loaded
  assert "paxml" in shlo_loaded

  # 7. audit_inline_snippets and audit_frameworks edge branches in ast_auditor.py
  class _DummyManager(SemanticsManager):
    """Dummy semantics manager stub for testing audit edge branches."""

    def __init__(self) -> None:
      """Initialize dummy data without loading full catalog."""
      self.data: dict[str, dict[str, JsonValue]] = {
        "NonDictVariants": {"variants": "invalid"},
        "NonStrSnippet": {"variants": {"torch": {"macro_template": 123}}},
        "EdgeOp": {
          "std_args": [{"name": "x"}],
          "variants": {
            "bad_mapping": "not_a_dict",
            "non_mapping_snap": {"api": "non_mapping_snap.op"},
            "torch": {"api": "torch.missing_op"},
            "jax": {"api": "jax.op_non_list_args", "args": {"x": "x"}},
          },
        },
      }

  class _EngineWithoutDiscover(GroundingEngineProtocol):
    """Engine stub raising AttributeError on _discover_target_files."""

    @property
    def _discover_target_files(self) -> None:
      """Raise AttributeError to simulate missing private method."""
      raise AttributeError("_discover_target_files not present")

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return False for symbol lookup.

      Args:
          target: Target framework name.
          symbol: Symbol name.

      Returns:
          Always False.
      """
      return False

    def suggest_closest_symbol(self, target: str, symbol: str, top_k: int = 1) -> list[str]:
      """Return a suggestion.

      Args:
          target: Target framework name.
          symbol: Symbol name.
          top_k: Maximum suggestions.

      Returns:
          Single suggestion list.
      """
      return ["torch.add"]

  mgr = _DummyManager()
  assert audit_inline_snippets(mgr, {}) == []
  aud_snaps: dict[str, RuntimeTensorOrValue] = {
    "non_mapping_snap": 123,
    "torch": {},
    "jax": {
      "jax.op_non_list_args": {
        "params": "not_a_list",
        "overloads": "not_a_list",
      }
    },
  }
  errs = audit_frameworks(mgr, aud_snaps, grounding_engine=_EngineWithoutDiscover())
  assert any("did you mean" in e for e in errs)

  aud_snaps2: dict[str, RuntimeTensorOrValue] = {
    "jax": {
      "jax.op_non_list_args": {
        "params": [{"name": "x", "default": None}],
        "overloads": ["not_a_dict", {"params": "not_a_list"}],
      }
    },
  }
  errs2 = audit_frameworks(mgr, aud_snaps2, framework="jax")
  assert errs2 == []
