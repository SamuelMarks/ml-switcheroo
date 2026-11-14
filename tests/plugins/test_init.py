"""Unit tests for lazy plugin discovery, import purity, and registered hook smoke execution."""

import contextlib
import importlib
from pathlib import Path
from unittest.mock import patch

import libcst as cst

import ml_switcheroo.core.hooks_registry as hr
import ml_switcheroo.plugins as plugins_pkg
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext, get_default_hook_registry
from ml_switcheroo.semantics.manager import SemanticsManager


def test_plugins_init_lazy_discovery(tmp_path: Path) -> None:
  """Verify importing `ml_switcheroo.plugins` has no side effects and `ensure_loaded` loads plugins lazily.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  hr.clear_hooks(mark_loaded=False)
  reg = hr.get_default_hook_registry()

  importlib.reload(plugins_pkg)
  assert hr.get_default_hook_registry().plugins_loaded is False
  assert len(reg.hooks) == 0

  reg.ensure_loaded()
  assert hr.get_default_hook_registry().plugins_loaded is True
  assert len(reg.hooks) > 0

  (tmp_path / "_protected.py").write_text("x = 1", encoding="utf-8")
  (tmp_path / "some_utils.py").write_text("x = 1", encoding="utf-8")
  (tmp_path / "my_helpers.py").write_text("x = 1", encoding="utf-8")
  (tmp_path / "valid_plugin.py").write_text("x = 1", encoding="utf-8")
  (tmp_path / "broken_plugin.py").write_text("raise RuntimeError('broken')", encoding="utf-8")

  with patch("logging.Logger.warning") as mock_warn:
    loaded = hr._import_from_dir(tmp_path, base_package=None)
    assert loaded == 1
    mock_warn.assert_called_once()


def test_registered_plugins_diverse_cst_node_smoke() -> None:
  """Verify registered hooks handle non-matching and varied CST nodes across target frameworks."""
  sem = SemanticsManager()
  targets = ["torch", "jax", "mlx", "tensorflow", "keras", "source_placeholder"]
  contexts: list[HookContext] = []
  for tgt in targets:
    ctx = HookContext(semantics=sem, config=RuntimeConfig(source_framework="jax", target_framework=tgt))
    ctx.current_op_id = "Conv2d"
    contexts.append(ctx)

  nodes: list[cst.CSTNode] = [
    cst.Call(func=cst.Name("func")),
    cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))),
    cst.Name("some_var"),
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("attr")),
    cst.Call(
      func=cst.Name("func"),
      args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"), keyword=cst.Name("kw"))],
    ),
    cst.Call(
      func=cst.Name("func"),
      args=[
        cst.Arg(value=cst.Integer("1")),
        cst.Arg(value=cst.List([cst.Element(cst.Integer("2"))]), keyword=cst.Name("padding")),
      ],
    ),
    cst.Call(
      func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")),
      args=[
        cst.Arg(value=cst.Integer("1"), keyword=cst.Name("index")),
        cst.Arg(value=cst.Integer("1"), keyword=cst.Name("value")),
      ],
    ),
    cst.Call(
      func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("flatten")),
      args=[cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("1"))],
    ),
    cst.For(
      target=cst.Name("i"),
      iter=cst.Name("iterable"),
      body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
    ),
    cst.FunctionDef(
      name=cst.Name("foo"),
      params=cst.Parameters(),
      body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
    ),
    cst.Assign(targets=[cst.AssignTarget(cst.Name("foo"))], value=cst.Call(func=cst.Name("some_func"))),
  ]

  from libcst._nodes.base import CSTValidationError

  reg = get_default_hook_registry()
  reg.ensure_loaded()
  for hook_func in reg.hooks.values():
    for node in nodes:
      for ctx in contexts:
        with contextlib.suppress(ValueError, TypeError, KeyError, IndexError, AttributeError, CSTValidationError):
          res = hook_func(node, ctx)
          assert isinstance(res, (cst.CSTNode, cst.FlattenSentinel, cst.RemovalSentinel))
  assert len(reg.hooks) > 0
