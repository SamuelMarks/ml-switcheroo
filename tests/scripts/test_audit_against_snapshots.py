"""Unit tests for `scripts/audit_against_snapshots.py` and `scripts/snapshots/` using `tmp_path` fixtures."""

import importlib
import json
import runpy
import sys
import types
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue

import scripts.audit_against_snapshots as aas
import scripts.check_snapshot_regressions as csr
import scripts.snapshots.ast_auditor as ast_aud
import scripts.snapshots.loader as snap_loader
import scripts.validate_snapshot_grounding as vsg
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from scripts.audit_against_snapshots import (
  audit_inline_snippets,
  audit_python_ast,
  compute_snapshot_checksums,
  generate_audit_report,
)
from scripts.snapshots.ast_auditor import audit_frameworks, extract_api_calls
from scripts.snapshots.loader import (
  GroundingEngineProtocol,
  _augment_framework_aliases,
  _flatten_single_framework,
  _hydrate_ir_snapshot,
  _normalize_snapshot_entry,
  _resolve_grounding_engine,
  load_snapshots,
  load_snapshots_multi,
)

_NL = chr(10)


@pytest.fixture
def setup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Create an isolated temporary repository workspace and `chdir` into it.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Workspace root `Path`.
  """
  snap_dir = tmp_path / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
  snap_dir.mkdir(parents=True)
  (tmp_path / "ml-compiler-snapshots").mkdir()

  src_dir = tmp_path / "src" / "ml_switcheroo"
  (src_dir / "frameworks").mkdir(parents=True)
  (src_dir / "plugins").mkdir(parents=True)

  monkeypatch.chdir(tmp_path)
  return tmp_path


def test_extract_api_calls_and_audit_python_ast(tmp_path: Path) -> None:
  """Verify `extract_api_calls` and `audit_python_ast` against real `tmp_path` Python source files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  py_file = tmp_path / "sample.py"
  py_file.write_text(f"import torch{_NL}torch.nn.Conv2d(1, 2, 3){_NL}(lambda: 1)().attr(){_NL}", encoding="utf-8")
  assert "torch.nn.Conv2d" in extract_api_calls(py_file)

  py_file.write_text(f"from torch import nn{_NL}nn.Conv2d(1, 2, 3){_NL}", encoding="utf-8")
  assert "torch.nn.Conv2d" in extract_api_calls(py_file)

  py_file.write_text(f"from . import local_mod{_NL}local_mod.foo(){_NL}", encoding="utf-8")
  assert extract_api_calls(py_file) == set()

  py_file.write_text(f"import torch{_NL}broken [ syntax", encoding="utf-8")
  assert extract_api_calls(py_file) == set()

  src_dir = tmp_path / "src_scan"
  src_dir.mkdir()
  (src_dir / "check.py").write_text(
    _NL.join(
      [
        "import torch",
        "import mlx",
        "import numpy",
        "import unknown",
        "import tensorflow",
        "torch.nn.Linear()",
        "torch.Tensor.model_validate()",
        "mlx.core.broken()",
        "numpy.ndarray()",
        "unknown.module.call()",
        "tensorflow.random.normal()",
      ]
    ),
    encoding="utf-8",
  )
  snapshots: dict[str, RuntimeTensorOrValue] = {
    "torch": {"torch.nn.Linear": {}},
    "mlx": {"mlx.core.valid": {}},
  }
  ast_errors = audit_python_ast([src_dir, tmp_path / "nonexistent"], snapshots)
  assert any("mlx.core.broken" in err for err in ast_errors)


def test_load_snapshots_and_multi_formats_from_disk(tmp_path: Path) -> None:
  """Verify `load_snapshots` and `load_snapshots_multi` across real JSON snapshot files on disk.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  empty_dir = tmp_path / "empty_snaps"
  empty_dir.mkdir()
  assert load_snapshots(empty_dir) == {}

  snap_dir = tmp_path / "snaps1"
  snap_dir.mkdir()
  rich_data = {
    "categories": {
      "cat1": [
        {"api_path": "torch.nn.Linear", "aliases": ["torch.nn.modules.linear.Linear"]},
        {"name": "torch.nn.Conv2d", "params": [{"name": "in_channels"}]},
        {"mnemonic": "MNEMO_OP"},
      ],
      "cat2": {"torch.nn.ReLU": {}},
      "cat3": "ignored_str",
    },
    "functions": {"torch.add": {}},
    "classes": {"torch.Tensor": {}},
    "other": {"args": {}},
    "unindexed_dict": {"unrelated": 1},
    "ignore_me": "value",
  }
  (snap_dir / "torch_v1.0.0.json").write_text(json.dumps(rich_data), encoding="utf-8")
  (snap_dir / "torch_map.json").write_text("{}", encoding="utf-8")
  (snap_dir / "ignore.txt").write_text("ignore", encoding="utf-8")
  (snap_dir / "unknown_vunknown.json").write_text("{}", encoding="utf-8")
  (snap_dir / "torch_v2.0.0.json").write_text(
    json.dumps(
      {
        **rich_data,
        "functions": {"torch.add": {}, "torch.sub": {}},
      }
    ),
    encoding="utf-8",
  )

  loaded1 = load_snapshots(snap_dir)
  assert {"torch.nn.Linear", "torch.nn.modules.linear.Linear", "torch.nn.ReLU", "torch.add", "torch.sub", "other"} <= set(
    loaded1["torch"].keys()
  )
  assert "ignore_me" not in loaded1["torch"]

  # Multi-directory loading, ISA/exhaustive/optax/pax/stablehlo filenames, and alias prefixes
  dir2 = tmp_path / "snaps2"
  dir2.mkdir()
  (dir2 / "mlx_v0.1.0.json").write_text(json.dumps({"functions": {"mlx.nn.Conv2d": {}}}), encoding="utf-8")
  (dir2 / "optax_v0.1.json").write_text(
    json.dumps({"categories": {"default": [{"name": "adam", "api_path": "optax.adam"}]}}),
    encoding="utf-8",
  )
  (dir2 / "rdna_isa.json").write_text(json.dumps({"functions": {"v_add_f32": {}}}), encoding="utf-8")
  (dir2 / "nvidia_sass_isa.json").write_text(json.dumps({"functions": {"FADD": {}}}), encoding="utf-8")
  (dir2 / "test_sass_exhaustive.json").write_text(json.dumps([{"mnemonic": "FADD"}]), encoding="utf-8")
  (dir2 / "test_rdna_exhaustive.json").write_text(json.dumps([{"mnemonic": "v_add"}]), encoding="utf-8")
  (dir2 / "custom_exhaustive.json").write_text(json.dumps({"functions": {"custom_op": {}}}), encoding="utf-8")
  (dir2 / "stablehlo_exhaustive.json").write_text(
    json.dumps({"operations": [{"name": "stablehlo.custom_call", "api_path": "stablehlo.custom_call"}]}),
    encoding="utf-8",
  )
  (dir2 / "pax_v1.4.0.json").write_text(
    json.dumps({"operations": [{"name": "praxis.base_layer", "api_path": "praxis.base_layer"}]}),
    encoding="utf-8",
  )
  (dir2 / "all_v1.json").write_text(
    json.dumps(
      {
        "categories": {
          "cat": [
            {"api_path": "jax.numpy.sin"},
            {"api_path": "mlx.core.cos"},
            {"api_path": "torch.nn.functional.relu"},
          ]
        }
      }
    ),
    encoding="utf-8",
  )
  (dir2 / "all_v2.json").write_text(json.dumps({"categories": {}}), encoding="utf-8")

  loaded2 = load_snapshots(dir2)
  assert {"nvidia_sass", "rdna", "stablehlo", "paxml"} <= set(loaded2.keys())
  assert {"jnp.sin", "mx.cos", "F.relu"} <= set(loaded2["all"].keys())

  multi = load_snapshots_multi([snap_dir, dir2, tmp_path / "missing_dir"])
  assert {"torch", "mlx", "optax", "rdna", "nvidia_sass", "custom_exhaustive"} <= set(multi.keys())
  assert isinstance(load_snapshots_multi(None), dict)

  with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.glob", return_value=[]):
    assert isinstance(load_snapshots_multi(None), dict)


def test_audit_frameworks_and_inline_snippets() -> None:
  """Verify `audit_frameworks` and `audit_inline_snippets` across hallucinated APIs, args, macros, and optax/flax."""
  mgr = SemanticsManager(load_defaults=False)
  mgr.data = {
    "Conv2d": {
      "variants": {
        "torch": {"api": "torch.nn.Conv2d", "args": {"std_arg": "in_channels", "dropped": None}},
        "mlx": {"api": "mlx.core.hallucinated", "args": {}},
        "jax": {"api": "torch.nn.Conv2d", "args": {"std_arg": "hallucinated_arg"}},
        "tensorflow": {"api": "tensorflow.random.normal", "args": {"arg1": "provided"}},
        "numpy": {"api": "torch.nn.Linear", "macro_template": "...", "args": {}},
        "flax": {"api": "", "args": {}},
        "keras": {"api": ";comment_api", "args": {}},
        "paxml": {"api": "ignore.abs", "args": {}},
        "stablehlo": {"api": "numpy.add", "args": {}},
        "nvidia_sass": {"api": "FADD", "args": {}},
        "rdna": {"api": "v_add_f32", "args": {}},
        "unknown_fw": {"api": "unknown.api", "args": {"std_arg": "hallucinated_arg"}},
      }
    },
    "MacroWithArgs": {
      "variants": {
        "torch": {"api": "torch.nn.Conv2d", "macro_template": "torch.nn.Conv2d({x})", "args": {"x": "in_channels"}},
      }
    },
    "OverloadOp": {
      "variants": {
        "torch": {"api": "torch.overload_fn", "args": {"x": "a"}},
      }
    },
    "Adam": {
      "variants": {
        "jax": {"api": "optax.adam", "args": {}},
        "flax_nnx": {"api": "optax.missing_opt", "args": {}},
      }
    },
    "MissingVar": {},
  }
  snapshots: dict[str, RuntimeTensorOrValue] = {
    "torch": {
      "torch.nn.Conv2d": {"params": [{"name": "in_channels"}]},
      "torch.nn.Linear": {"params": [{"name": "in_features"}]},
      "torch.overload_fn": {
        "api_path": "torch.overload_fn",
        "params": [{"name": "a"}, {"name": "b"}],
        "overloads": [
          {"params": [{"name": "a"}]},
          {"params": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        ],
      },
    },
    "mlx": {"mlx.core.valid": {}},
    "jax": {"torch.nn.Conv2d": {"params": [{"name": "in_channels"}]}},
    "flax_nnx": {},
    "optax": {"optax.adam": {"args": []}, "optax.sgd": {"args": []}},
    "tensorflow": {"tensorflow.random.normal": {"params": [{"name": "provided"}, {"name": "missing_req_param"}]}},
    "numpy": {"torch.nn.Linear": {"params": [{"name": "missing_req_param"}]}},
    "stablehlo": {"numpy.add": {"params": [{"name": "x", "kind": "VAR_KEYWORD"}, {"name": "y"}]}},
    "keras": {},
    "flax": {},
    "paxml": {"something": {}},
    "nvidia_sass": {},
    "rdna": {"something": {}},
  }
  errors = audit_frameworks(mgr, snapshots)
  assert any("hallucinated API" in e and "mlx" in e for e in errors)
  assert any("hallucinated argument" in e and "jax" in e for e in errors)
  assert any("missing required arguments" in e and "tensorflow" in e for e in errors)

  # `flax_nnx` grounding engine symbol checks and IR canonical ONNX fallback
  flax_mgr = SemanticsManager(load_defaults=False)
  flax_mgr.data = {
    "op1": {"variants": {"flax_nnx": {"api": "flax.nnx.CustomLinear"}}},
    "op2": {"variants": {"flax_nnx": {"api": "jax.numpy.missing_fn"}}},
  }
  mock_ge = MagicMock()
  mock_ge.has_symbol.return_value = True
  mock_ge._discover_target_files.return_value = True
  assert audit_frameworks(flax_mgr, {"flax_nnx": {}, "jax": {}}, grounding_engine=mock_ge) == []

  ir_mgr = SemanticsManager(load_defaults=False)
  ir_mgr.data = {"add": {"variants": {"ir": {"api": "Add", "args": {}}}}}
  with patch("ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops", side_effect=RuntimeError("onnx fail")):
    assert audit_frameworks(ir_mgr, {"ir": {}}) == []

  # Inline snippets audit
  snip_mgr = SemanticsManager(load_defaults=False)
  snip_mgr.data = {
    "ReLU": {
      "variants": {
        "torch": {"macro_template": "torch.nn.functional.relu(x) + (lambda: 1)().attr()"},
        "mlx": {"macro_template": "mlx.core.not_exist(x)"},
        "broken_ast": {"macro_template": "broken ["},
        "ignore_list": {"macro_template": "numpy.add(x)"},
        "unsupported_fw": {"macro_template": "unsupported.api(x)"},
      }
    }
  }
  snip_errs = audit_inline_snippets(
    snip_mgr,
    {"torch": {"torch.nn.functional.relu": {}}, "mlx": {"mlx.core.relu": {}}},
  )
  assert any("mlx.core.not_exist" in err for err in snip_errs)


def test_audit_report_checksums_and_loader_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `generate_audit_report`, `compute_snapshot_checksums`, and snapshot normalizer helpers.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  snap_dir = tmp_path / "target_snaps"
  snap_dir.mkdir()
  (snap_dir / "array_api_v2024.12.json").write_text(
    json.dumps({"categories": {"elementwise": [{"name": "add", "api_path": "array_api.add", "params": []}]}}),
    encoding="utf-8",
  )
  (snap_dir / "scipy_v1.13.1.json").write_text(
    json.dumps({"categories": {"special": [{"name": "erf", "api_path": "scipy.special.erf", "params": []}]}}),
    encoding="utf-8",
  )
  (snap_dir / "safetensors_v0.7.0.json").write_text(
    json.dumps({"categories": {"io": [{"name": "save_file", "api_path": "safetensors.torch.save_file", "params": []}]}}),
    encoding="utf-8",
  )
  snapshots = load_snapshots_multi([snap_dir])
  checksums = compute_snapshot_checksums([snap_dir])
  assert {"array_api_v2024.12.json", "scipy_v1.13.1.json", "safetensors_v0.7.0.json"} <= set(checksums.keys())

  mgr = SemanticsManager(load_defaults=False)
  mgr.data = {
    "add": {"variants": {"array_api": {"api": "array_api.add"}}},
    "erf": {"variants": {"scipy": {"api": "scipy.special.erf"}}},
    "save": {"variants": {"safetensors": {"api": "safetensors.torch.save_file"}}},
  }
  report = generate_audit_report(mgr, snapshots, errors=[], checksums=checksums)
  assert report["status"] == "pass"
  targets = report["targets"]
  assert isinstance(targets, dict)
  for tgt in ("array_api", "scipy", "safetensors"):
    tgt_entry = targets[tgt]
    assert isinstance(tgt_entry, dict)
    assert tgt_entry["mapped_operations"] == 1

  # Normalizer, alias augmenter, IR hydration, and `_flatten_single_framework` edge shapes
  item_overloads: dict[str, JsonValue] = {
    "docstring": "fn(a)",
    "params": ["not_a_dict", {"name": "req_param", "default": None, "kind": "POSITIONAL_OR_KEYWORD"}],
    "overloads": ["not_a_dict", {"params": "not_a_list"}, {"params": ["not_a_dict", {"name": "req_param"}]}],
  }
  _normalize_snapshot_entry(item_overloads)

  item_colon: dict[str, JsonValue] = {
    "docstring": f"fn(opt_colon){_NL}Args:{_NL}  opt_colon: int, optional{_NL}",
    "params": [{"name": "opt_colon", "default": None, "kind": "POSITIONAL_OR_KEYWORD", "description": "val"}],
  }
  _normalize_snapshot_entry(item_colon)
  params_list = item_colon["params"]
  assert isinstance(params_list, list)
  first_p = params_list[0]
  assert isinstance(first_p, dict)
  assert first_p["default"] == "None"

  _augment_framework_aliases("numpy", {"numpy.add": {"api_path": "numpy.add"}})
  _augment_framework_aliases("tensorflow", {"tf.add": {"api_path": "tf.add"}})
  _augment_framework_aliases("keras", {"keras.add": {"api_path": "keras.add"}})

  sibling_ir = Path(__file__).resolve().parent.parent.parent / "ml-switcheroo-ir" / "src"
  sibling_ir_up = sibling_ir.parent.parent.parent / "ml-switcheroo-ir" / "src"
  monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in (str(sibling_ir), str(sibling_ir_up))])
  ir_flat: dict[str, dict[str, dict[str, JsonValue]]] = {}
  _hydrate_ir_snapshot(ir_flat)
  assert "ir" in ir_flat

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "ml_switcheroo_ir", None)
    empty_ir: dict[str, dict[str, dict[str, JsonValue]]] = {}
    _hydrate_ir_snapshot(empty_ir)
    assert empty_ir == {}

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "ml_ecosystem_snapshots.frameworks.ir", None)
    no_collect_ir: dict[str, dict[str, dict[str, JsonValue]]] = {}
    _hydrate_ir_snapshot(no_collect_ir)
    assert "ir" in no_collect_ir

  with pytest.raises(FileNotFoundError, match="No valid framework snapshots"):
    load_snapshots_multi([tmp_path / "missing_strict_dir"], strict=True)

  flat: dict[str, dict[str, dict[str, JsonValue]]] = {}
  _flatten_single_framework("hw", ["str_item", {"no_mnemonic": 1}, {"mnemonic": "v_add"}], flat)
  assert "v_add" in flat["hw"]
  _flatten_single_framework("invalid_scalar", 12345, flat)
  assert flat["invalid_scalar"] == {}
  _flatten_single_framework(
    "flax_nnx",
    {"categories": {"default": [{"api_path": "flax.nnx.Linear", "name": "Linear"}]}},
    flat,
  )
  assert "nnx.Linear" in flat["flax_nnx"]
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
      "operations": [
        "not_a_dict",
        {"api_path": "op.path"},
        {"class_name": "OpClass"},
        {"name": "OpName"},
        {"name": "OpWithAliases", "aliases": [123, "alias_1"]},
        {"name": "OpBadAliases", "aliases": "not_a_list"},
      ],
    },
    flat,
  )
  assert {"custom_fw.alias", "op.path", "OpClass", "OpName", "alias_1"} <= set(flat["custom_fw"].keys())


def test_grounding_engine_resolution_and_main_entrypoints(setup_workspace: Path, tmp_path: Path) -> None:
  """Verify `_resolve_grounding_engine` and CLI `main()` modes.

  Args:
      setup_workspace: Workspace directory fixture.
      tmp_path: Pytest temporary directory fixture.
  """
  _ = setup_workspace
  orig_import = importlib.import_module

  class _MockGroundingEngine(GroundingEngineProtocol):
    """Stub `GroundingEngine` class implementing `GroundingEngineProtocol`."""

    def __init__(self, base_dirs: typing.Sequence[str | Path] | None = None) -> None:
      """Initialize stub engine.

      Args:
          base_dirs: Optional snapshot directories.
      """
      self.base_dirs = base_dirs

    def _discover_target_files(self, target: str) -> list[Path]:
      """Return empty list of target files.

      Args:
          target: Target framework key.

      Returns:
          Empty list.
      """
      _ = target
      return []

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return `True`.

      Args:
          target: Target framework key.
          symbol: Symbol path.

      Returns:
          Always `True`.
      """
      _ = (target, symbol)
      return True

    def suggest_closest_symbol(self, target: str, symbol: str) -> str | None:
      """Return `None`.

      Args:
          target: Target framework key.
          symbol: Symbol path.

      Returns:
          Always `None`.
      """
      _ = (target, symbol)
      return None

  class _EngineModWithCls(types.ModuleType):
    """Module stub exposing callable `GroundingEngine` class."""

    def __init__(self, name: str) -> None:
      """Initialize module stub and bind `self.GroundingEngine`.

      Args:
          name: Module name.
      """
      super().__init__(name)
      self.GroundingEngine = _MockGroundingEngine

  class _EngineModNonCallable(types.ModuleType):
    """Module stub exposing non-callable `GroundingEngine`."""

    def __init__(self, name: str) -> None:
      """Initialize module stub with integer `self.GroundingEngine`.

      Args:
          name: Module name.
      """
      super().__init__(name)
      self.GroundingEngine = 123

  with patch(
    "importlib.import_module",
    side_effect=lambda n, *a: _EngineModWithCls(n) if "grounding.engine" in n else orig_import(n, *a),
  ):
    assert _resolve_grounding_engine() is _MockGroundingEngine

  attempts = 0

  def _second_succeeds(name: str, *args: str) -> types.ModuleType:
    """Fail first `grounding.engine` import and succeed on second.

    Args:
        name: Module path.
        *args: Optional package arguments.

    Returns:
        Imported module.

    Raises:
        ImportError: On first `grounding.engine` attempt.
    """
    nonlocal attempts
    if "grounding.engine" in name:
      attempts += 1
      if attempts == 1:
        raise ImportError("first failed")
      return _EngineModWithCls(name)
    return orig_import(name, *args)

  with patch("importlib.import_module", side_effect=_second_succeeds), patch("pathlib.Path.exists", return_value=True):
    assert _resolve_grounding_engine() is _MockGroundingEngine

  with (
    patch(
      "importlib.import_module",
      side_effect=lambda n, *a: (
        (_ for _ in ()).throw(ImportError("none")) if "grounding.engine" in n else orig_import(n, *a)
      ),
    ),
    patch("pathlib.Path.exists", return_value=True),
  ):
    assert _resolve_grounding_engine() is None

  with patch("importlib.import_module", return_value=_EngineModNonCallable("ml_ecosystem_snapshots.grounding.engine")):
    assert _resolve_grounding_engine() is None

  # `audit_against_snapshots.main` across strict/non-strict/empty/missing-fw/report/GroundingEngine failure
  with (
    patch("sys.argv", ["audit_against_snapshots.py"]),
    patch.object(aas, "audit_frameworks", return_value=[]),
    patch.object(aas, "audit_python_ast", return_value=[]),
    patch.object(aas, "audit_inline_snippets", return_value=[]),
  ):
    assert aas.main() == 0

  for argv, expected_rc in ((["audit_against_snapshots.py", "--strict"], 1), (["audit_against_snapshots.py"], 0)):
    with (
      patch("sys.argv", argv),
      patch.object(aas, "load_snapshots_multi", return_value={"torch": {"torch.add": {}}}),
      patch.object(aas, "audit_frameworks", return_value=["err"]),
      patch.object(aas, "audit_python_ast", return_value=[]),
      patch.object(aas, "audit_inline_snippets", return_value=[]),
    ):
      assert aas.main() == expected_rc

  with (
    patch("sys.argv", ["audit_against_snapshots.py", "--strict"]),
    patch.object(aas, "load_snapshots_multi", return_value={}),
  ):
    assert aas.main() == 1

  with (
    patch("sys.argv", ["audit_against_snapshots.py", "--strict", "--framework", "nonexistent_fw"]),
    patch.object(aas, "load_snapshots_multi", return_value={"torch": {"torch.add": {}}}),
  ):
    assert aas.main() == 1

  report_file = tmp_path / "audit_report.json"
  with (
    patch("sys.argv", ["audit_against_snapshots.py", "--report", str(report_file)]),
    patch("importlib.resources.files", side_effect=OSError("no pkg")),
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

  repo_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "audit_against_snapshots.py"
  with (
    patch("sys.argv", ["audit_against_snapshots.py"]),
    patch("scripts.snapshots.loader.load_snapshots_multi", return_value={}),
    patch("scripts.snapshots.ast_auditor.audit_frameworks", return_value=[]),
    patch("scripts.snapshots.ast_auditor.audit_python_ast", return_value=[]),
    patch("scripts.snapshots.ast_auditor.audit_inline_snippets", return_value=[]),
    patch("sys.exit") as mock_exit,
  ):
    runpy.run_path(str(repo_script), run_name="__main__")
    mock_exit.assert_called()


def test_companion_scripts_and_defensive_auditor_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify companion regression/grounding scripts and defensive `ast_auditor` / checksum branches.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  # Companion scripts & defensive auditor/checksum branches
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "ml_ecosystem_snapshots.api", None)
    mp.setitem(sys.modules, "ml_ecosystem_snapshots.diff", None)
    importlib.reload(csr)
    assert csr.diff_snapshots is None
  importlib.reload(csr)

  assert callable(vsg.main)
  with patch("scripts.audit_against_snapshots.main", return_value=0), pytest.raises(SystemExit) as exc_info:
    runpy.run_module("scripts.validate_snapshot_grounding", run_name="__main__")
  assert exc_info.value.code == 0

  class _EngineWithoutDiscover(GroundingEngineProtocol):
    """Engine stub raising `AttributeError` on `_discover_target_files`."""

    def _discover_target_files(self, target: str) -> list[Path]:
      """Raise `AttributeError`.

      Args:
          target: Target framework key.

      Returns:
          Never returns; raises `AttributeError`.

      Raises:
          AttributeError: Always raised.
      """
      _ = target
      raise AttributeError("missing _discover_target_files")

    def has_symbol(self, target: str, symbol: str) -> bool:
      """Return `False`.

      Args:
          target: Target framework key.
          symbol: Qualified symbol string.

      Returns:
          Always `False`.
      """
      _ = (target, symbol)
      return False

    def suggest_closest_symbol(self, target: str, symbol: str) -> str | None:
      """Suggest `'torch.add'`.

      Args:
          target: Target framework key.
          symbol: Qualified symbol string.

      Returns:
          Literal `'torch.add'`.
      """
      _ = (target, symbol)
      return "torch.add"

  edge_mgr = SemanticsManager(load_defaults=False)
  edge_mgr.data = {
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
  assert audit_inline_snippets(edge_mgr, {}) == []
  errs = audit_frameworks(
    edge_mgr,
    {"non_mapping_snap": 123, "torch": {}, "jax": {"jax.op_non_list_args": {"params": "bad", "overloads": "bad"}}},
    grounding_engine=_EngineWithoutDiscover(),
  )
  assert any("did you mean" in e for e in errs)
  assert (
    audit_frameworks(
      edge_mgr,
      {
        "jax": {
          "jax.op_non_list_args": {
            "params": [{"name": "x", "default": None}],
            "overloads": ["not_a_dict", {"params": "not_a_list"}],
          }
        }
      },
      framework="jax",
    )
    == []
  )

  with patch.object(ast_aud, "GroundingEngine", side_effect=RuntimeError("init fail")):
    mgr_ge = SemanticsManager(load_defaults=False)
    mgr_ge.data = {
      "op_flax": {"variants": {"flax_nnx": {"api": "jax.numpy.sin"}}},
      "op_ir_bad": {"variants": {"ir": {"api": "UnknownIrOp"}}},
    }
    assert len(audit_frameworks(mgr_ge, {"flax_nnx": {}, "jax": {"jax.numpy.sin": {}}, "ir": {}})) == 1

  mgr_err = SemanticsManager(load_defaults=False)
  mgr_err.data = {"op1": {"variants": {"torch": {"api": "torch.bad_op"}}}}
  for ge_cfg in (
    ([True, False], RuntimeError("sym fail"), None),
    ([False, RuntimeError("sug fail")], False, None),
    (True, False, ""),
  ):
    disc_se, has_se, sug_ret = ge_cfg
    bad_ge = MagicMock()
    if isinstance(disc_se, list):
      bad_ge._discover_target_files.side_effect = disc_se
    else:
      bad_ge._discover_target_files.return_value = disc_se
    if isinstance(has_se, Exception):
      bad_ge.has_symbol.side_effect = has_se
    else:
      bad_ge.has_symbol.return_value = has_se
    bad_ge.suggest_closest_symbol.return_value = sug_ret
    assert len(audit_frameworks(mgr_err, {"torch": {}}, grounding_engine=bad_ge)) == 1

  mgr_comma = SemanticsManager(load_defaults=False)
  mgr_comma.data = {"op_c": {"variants": {"torch": {"api": "torch.comma_op", "args": {"x": "valid_tok"}}}}}
  assert (
    audit_frameworks(
      mgr_comma, {"torch": {"torch.comma_op": {"params": ["not_a_dict", {"name": "valid_tok, 123invalid"}]}}}
    )
    == []
  )

  with (
    patch("importlib.resources.files", side_effect=ImportError("no res")),
    patch.object(snap_loader, "Path", return_value=tmp_path / "nonexistent_root"),
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

  def _guarded_read_bytes(self_path: Path) -> bytes:
    """Raise `OSError` for `err_v1.json`.

    Args:
        self_path: File path being read.

    Returns:
        Raw file bytes.

    Raises:
        OSError: When `self_path.name == 'err_v1.json'`.
    """
    if self_path.name == "err_v1.json":
      raise OSError("cannot read")
    return orig_read_bytes(self_path)

  monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)
  cs = compute_snapshot_checksums([cs_dir, tmp_path / "missing_cs_dir"])
  assert "err_v1.json" not in cs
  assert "empty_v1.json" not in cs
