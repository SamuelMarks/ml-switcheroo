"""
Tests for IO Handler Plugin.

Verifies that:
1. `torch.save` is mapped to `orbax.checkpoint...save`.
2. `torch.load` is mapped to `orbax.checkpoint...restore`.
3. Arguments are correctly swapped and labelled (`obj`, `f` -> `item`, `directory`).
4. Preamble import is injected correctly (requires function context).
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.io_handler import transform_io_calls


# Helper
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook & ensure plugins loaded state is set
  hooks._HOOKS["io_handler"] = transform_io_calls
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  save_def = {
    "requires_plugin": "io_handler",
    "std_args": ["obj", "f"],
    "variants": {
      "torch": {"api": "torch.save"},
      "jax": {"requires_plugin": "io_handler"},
    },
  }

  load_def = {
    "requires_plugin": "io_handler",
    "std_args": ["f"],
    "variants": {
      "torch": {"api": "torch.load"},
      "jax": {"requires_plugin": "io_handler"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.save":
      return "save", save_def
    if name == "torch.load":
      return "load", load_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"save": save_def, "load": load_def}
  mgr.is_verified.return_value = True

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_save_transform_positional(rewriter):
  """
  Input: torch.save(model, 'path')
  Expect: import orbax.checkpoint; orbax.checkpoint.PyTreeCheckpointer().save(directory='path', item=model)
  """
  # Wrap in function to allow preamble injection
  code = "def save_model():\n    torch.save(model, 'ckpt_path')"
  result = rewrite_code(rewriter, code)

  assert "import orbax.checkpoint" in result
  assert "orbax.checkpoint.PyTreeCheckpointer()" in result
  assert ".save(" in result
  assert "directory='ckpt_path'" in result.replace(" ", "")
  assert "item=model" in result.replace(" ", "")


def test_save_transform_keywords(rewriter):
  """
  Input: torch.save(f='path', obj=model)
  Expect: ordered correctly in orbax call
  """
  code = "def save_model():\n    torch.save(f='path', obj=model)"
  result = rewrite_code(rewriter, code)

  clean = result.replace(" ", "")
  assert "directory='path'" in clean
  assert "item=model" in clean


def test_load_transform(rewriter):
  """
  Input: m = torch.load('path')
  Expect: m = orbax.checkpoint.PyTreeCheckpointer().restore('path')
  """
  code = "def load_model():\n    m = torch.load('path')"
  result = rewrite_code(rewriter, code)

  assert "import orbax.checkpoint" in result
  assert ".restore('path')" in result


def test_ignored_if_wrong_target(rewriter):
  """
  Verify pass-through if target is truly supported (e.g. ghost_fw).
  (Numpy is now supported, so we use a dummy name).
  """
  # Manually switch target for this test instance
  rewriter.ctx._runtime_config.target_framework = "ghost_fw"
  rewriter.ctx.target_fw = "ghost_fw"

  code = "def f():\n    torch.save(x, 'f')"
  result = rewrite_code(rewriter, code)

  assert "torch.save" in result
  assert "orbax" not in result
  assert "numpy" not in result
