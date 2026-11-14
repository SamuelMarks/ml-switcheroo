"""Unit tests for ``scripts/lint_architecture.py`` and ``ml_switcheroo.core.bootstrap``."""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from ml_switcheroo.core.bootstrap import ensure_compiler_and_zero_on_path, is_package_installed
from ml_switcheroo.frameworks.base import resolve_snapshot_dir
from ml_switcheroo.generated_tests.runtime import seed_framework_rngs
from scripts.lint_architecture import (
  _ArchitectureVisitor,
  _contains_framework_literal,
  _extract_var_leaf,
  check_source,
  iter_python_files,
  main,
)


def test_lint_architecture_core_literals_and_docstrings() -> None:
  """Verify docstrings and allowlisted core files pass while leaked framework literals fail."""
  core_file = Path("src/ml_switcheroo/core/rewriter/some_pass.py")
  compliant_src = "\x0a".join(
    [
      '"""Module docstring mentioning torch and jax."""',
      "",
      "class Helper:",
      '  """Class docstring mentioning mlx."""',
      "",
      "  def run(self) -> str:",
      '    """Function docstring mentioning keras."""',
      '    return "neutral_value"',
      "",
      "  async def arun(self) -> int:",
      '    """Async docstring mentioning rdna."""',
      "    return 1",
    ]
  )
  assert check_source(compliant_src, core_file) == []

  leaked_literal = "x = 'nvidia_sass'\x0a"
  res_lit = check_source(leaked_literal, core_file)
  assert len(res_lit) == 1
  assert "Hardcoded framework literal found" in res_lit[0]

  leaked_dict = "d = {'stablehlo': 1, 2: 'ok'}\x0a"
  res_dict = check_source(leaked_dict, core_file)
  assert len(res_dict) == 1
  assert "Hardcoded framework dictionary key found" in res_dict[0]

  plugin_file = Path("src/ml_switcheroo/plugins/custom_plugin.py")
  res_plugin_dict = check_source("d = {'wasm': 'x'}\x0a", plugin_file)
  assert len(res_plugin_dict) == 1
  assert "Hardcoded framework dictionary key found" in res_plugin_dict[0]

  # Allowlisted registry file permits dialect registration tables
  registry_file = Path("src/ml_switcheroo/core/compiler/registry.py")
  assert check_source("TABLE = {'nvidia_sass': 1, 'rdna': 2}\x0a", registry_file) == []


def test_lint_architecture_cross_layer_imports() -> None:
  """Verify cross-layer imports from cli or testing into core are rejected."""
  core_file = Path("src/ml_switcheroo/core/some_core.py")
  bad_imports = "\x0a".join(
    [
      "import ml_switcheroo.cli.commands",
      "from ml_switcheroo.testing.linter import StructuralLinter",
      "from . import local_helper",
    ]
  )
  violations = check_source(bad_imports, core_file)
  assert len(violations) == 2
  assert all("Cross-layer import found" in v for v in violations)


def test_lint_architecture_routing_comparisons() -> None:
  """Verify framework routing comparisons on LHS/RHS, str(...), and collections are caught."""
  outside_file = Path("src/ml_switcheroo/cli/handlers/some_handler.py")
  routing_cases = [
    "if str(self.target_fw) == 'tikz':\x0a  pass\x0a",
    "if 'cpp' == fw:\x0a  pass\x0a",
    "if source_framework in ('latex_dsl', 'html'):\x0a  pass\x0a",
    "if target != 'mlir':\x0a  pass\x0a",
    "if 1 < 2 and x == 'other':\x0a  pass\x0a",
  ]
  assert len(check_source(routing_cases[0], outside_file)) == 1
  assert len(check_source(routing_cases[1], outside_file)) == 1
  assert len(check_source(routing_cases[2], outside_file)) == 1
  assert len(check_source(routing_cases[3], outside_file)) == 1
  assert check_source(routing_cases[4], outside_file) == []
  assert check_source("if fw == 'torch':\x0a  pass\x0a", Path("src/ml_switcheroo/config.py")) == []

  # Helper edge cases for _extract_var_leaf and _contains_framework_literal
  import ast

  assert _extract_var_leaf(ast.Constant(value=1)) is None
  assert _contains_framework_literal(ast.Constant(value=42)) is None
  assert _contains_framework_literal(ast.Tuple(elts=[ast.Constant(value="not_fw"), ast.Constant(value=1)])) is None


def test_lint_architecture_sys_mutations_and_evasions() -> None:
  """Verify unauthorized sys.modules / sys.path mutations and reflection evasions are caught."""
  app_file = Path("src/ml_switcheroo/utils/helper.py")
  bad_sys = "\x0a".join(
    [
      "import sys as _s",
      "from sys import modules as _mods, path as _path, version",
      "_s.path.append('/tmp')",
      "_mods['x'] = None",
      "del _s.modules['x']",
      "_path += ['/opt']",
      "_s.path = []",
      "y = _s.version",
      "arr = [1]",
      "arr[0] = 2",
    ]
  )
  violations = check_source(bad_sys, app_file)
  assert len(violations) == 5
  assert all("Unauthorized sys mutation found" in v for v in violations)

  # Bootstrap isolation file is allowed to register explicit paths
  bootstrap_file = Path("src/ml_switcheroo/core/bootstrap.py")
  assert check_source("import sys\x0asys.path.append('/x')\x0asys.modules['a'] = None\x0a", bootstrap_file) == []

  # Reflection dunder evasion via builtins.getattr and direct getattr
  evasion = "\x0a".join(
    [
      "import builtins",
      "builtins.getattr({'v': 1}, 'keys')",
      "getattr(obj, '__setitem__')",
      "getattr(obj, 'normal_attr')",
    ]
  )
  ev_violations = check_source(evasion, app_file)
  assert len(ev_violations) == 2
  assert all("Reflection type-checker evasion found" in v for v in ev_violations)

  # Fallback lineno on non-stmt/expr AST node in _ArchitectureVisitor._add
  import ast

  vis = _ArchitectureVisitor(app_file, set())
  vis._add(ast.Module(body=[], type_ignores=[]), "Error: Test", "detail")
  assert vis.violations == [f"Error: Test in {app_file}:1: detail"]


def test_lint_architecture_cli_and_file_iteration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """Verify iter_python_files and main CLI exit codes across files, dirs, and missing paths.

  Args:
      tmp_path: Pytest temporary directory fixture.
      capsys: Pytest stdout/stderr capture fixture.
  """
  sub = tmp_path / "pkg"
  sub.mkdir()
  good_py = sub / "good.py"
  good_py.write_text("x = 1\x0a", encoding="utf-8")
  txt_file = sub / "notes.txt"
  txt_file.write_text("ignored", encoding="utf-8")
  missing = tmp_path / "does_not_exist.py"

  files = iter_python_files([str(good_py), str(txt_file), str(sub), str(missing)])
  assert good_py in files
  assert txt_file not in files

  assert main([]) == 0
  assert main([str(good_py)]) == 0

  bad_py = sub / "bad.py"
  bad_py.write_text("if framework == 'torch':\x0a  pass\x0a", encoding="utf-8")
  assert main([str(bad_py)]) == 1
  captured = capsys.readouterr()
  assert "Hardcoded framework routing found" in captured.out

  with mock.patch.object(sys, "argv", ["lint_architecture.py"]):
    assert main(None) == 0


def test_bootstrap_snapshot_resolution_and_rng_seeding(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify bootstrap path discovery, snapshot directory resolution, and adapter RNG seeding.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert is_package_installed("sys") is True
  assert is_package_installed("nonexistent_pkg_xyz_12345") is False
  with mock.patch("importlib.util.find_spec", side_effect=ValueError("bad spec")):
    assert is_package_installed("bad") is False

  # Without workspace_parent or env var, ensure_compiler_and_zero_on_path returns [] without mutating sys.path
  monkeypatch.delenv("ML_SWITCHEROO_WORKSPACE_PATH", raising=False)
  assert ensure_compiler_and_zero_on_path() == []

  # With ML_SWITCHEROO_WORKSPACE_PATH env var, registers existing <repo>/src
  comp_src = tmp_path / "zero-jax" / "src"
  comp_src.mkdir(parents=True)
  monkeypatch.setenv("ML_SWITCHEROO_WORKSPACE_PATH", str(tmp_path))
  saved_sys_path = list(sys.path)
  try:
    added = ensure_compiler_and_zero_on_path()
    assert str(comp_src) in added
    assert ensure_compiler_and_zero_on_path(workspace_parent=tmp_path) == []
  finally:
    sys.path[:] = saved_sys_path

  # resolve_snapshot_dir precedence: explicit arg -> SNAPSHOT_DIR -> env var -> find_spec -> None
  assert resolve_snapshot_dir(tmp_path) == tmp_path
  monkeypatch.setenv("ML_SWITCHEROO_SNAPSHOT_DIR", str(tmp_path))
  with mock.patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", None):
    assert resolve_snapshot_dir() == tmp_path
  monkeypatch.delenv("ML_SWITCHEROO_SNAPSHOT_DIR", raising=False)
  with mock.patch("ml_switcheroo.frameworks.base.SNAPSHOT_DIR", None):
    with mock.patch("importlib.util.find_spec", side_effect=ValueError("err")):
      assert resolve_snapshot_dir() is None
    fake_spec = mock.MagicMock()
    fake_spec.origin = None
    with mock.patch("importlib.util.find_spec", return_value=fake_spec):
      assert resolve_snapshot_dir() is None

  # Exercise Keras utils.set_random_seed and adapter None branch in seed_framework_rngs
  class KerasStubModule(types.ModuleType):
    """Typed Keras module stub exposing ``utils``."""

    utils: types.SimpleNamespace

  keras_mod = KerasStubModule("keras")
  keras_utils = types.SimpleNamespace(set_random_seed=mock.MagicMock())
  keras_mod.utils = keras_utils
  with mock.patch.dict(sys.modules, {"keras": keras_mod}):
    seed_framework_rngs(99)
  keras_utils.set_random_seed.assert_called_once_with(99)

  with mock.patch("ml_switcheroo.frameworks.base.get_adapter", return_value=None):
    seed_framework_rngs(42)

  # Cover _execute_autodiff_strategy explicit autodiff_module_key and non-matching module fallback
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
  from ml_switcheroo.core.ir_bridge import synthesize_backward_vjp_graph
  from ml_switcheroo.core.runtime_tensor import OnDeviceTraceTensor
  from ml_switcheroo.core.runtime_tracer import _execute_autodiff_strategy

  loss_t = OnDeviceTraceTensor(data=[1.0], shape=(1,), requires_grad=True)
  plain_mod = types.ModuleType("plain_mod")
  assert (
    _execute_autodiff_strategy(
      "functional_value_and_grad",
      loss_t,
      {"plain": plain_mod},
      autodiff_module_key="plain",
    )
    == "functional_value_and_grad"
  )

  # Cover synthesize_backward_vjp_graph when ml_switcheroo_compiler.autodiff returns a valid LogicalGraph
  fwd_g = LogicalGraph(name="Net")
  fwd_g.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(2,))
  fwd_g.nodes["loss"] = LogicalNode(id="loss", op_type="Sum", inputs=["x"], shape_metadata=(1,))

  class FakeAutodiffMod(types.ModuleType):
    """Typed fake ``ml_switcheroo_compiler.autodiff`` module."""

    def grad(self, graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
      """Return ``graph`` copy as synthesized VJP graph.

      Args:
          graph: Forward LogicalGraph.
          wrt: Differentiated parameter IDs.
          output_id: Scalar output node ID.

      Returns:
          LogicalGraph: Cloned VJP graph.
      """
      return graph.model_copy(deep=True)

  from ml_switcheroo.config import LossReduction

  fake_pkg = types.ModuleType("ml_switcheroo_compiler")
  fake_transforms = types.ModuleType("ml_switcheroo_compiler.transforms")
  fake_ad = FakeAutodiffMod("ml_switcheroo_compiler.transforms.autodiff")
  with mock.patch.dict(
    sys.modules,
    {
      "ml_switcheroo_compiler": fake_pkg,
      "ml_switcheroo_compiler.transforms": fake_transforms,
      "ml_switcheroo_compiler.transforms.autodiff": fake_ad,
    },
  ):
    bwd = synthesize_backward_vjp_graph(fwd_g, LossReduction.SUM, observed_shapes={"x": [2]})
    assert "loss" in bwd.nodes
    assert "grad_loss" not in bwd.nodes
