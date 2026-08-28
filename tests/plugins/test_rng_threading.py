"""Test suite for the Rng Threading module."""

import pytest
import libcst as cst
from typing import Generator, Dict
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.rng_threading import inject_prng_threading
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  return rewriter.convert(tree).code


@pytest.fixture
def rewriter() -> Generator[PivotRewriter, None, None]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["inject_prng"] = inject_prng_threading
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  op_def: Dict[str, Dict[str, Dict[str, str]]] = {"variants": {"jax": {"requires_plugin": "inject_prng"}}}
  mgr.get_definition.return_value = ("dropout", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)
  mgr.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=True)}
  mgr.framework_configs = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  yield PivotRewriter(mgr, cfg)


def test_rng_basic_injection(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of rng basic injection.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f(x):\n  return torch.dropout(x)"
  res: str = rewrite_code(rewriter, code)
  assert "def f(rng, x):" in res or "def f(x, rng):" in res
  assert "rng, key = jax.random.split(rng)" in res
  assert "key=key" in res


def test_rng_custom_configuration(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of rng custom configuration.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.context.config.plugin_settings = {"rng_arg_name": "seed", "key_var_name": "k"}
  code: str = "def f(x):\n  torch.dropout(x)"
  res: str = rewrite_code(rewriter, code)
  assert "def f(seed, x):" in res or "def f(x, seed):" in res
  assert "seed, k = jax.random.split(seed)" in res
  assert "key=k" in res


def test_no_injection_if_traits_disabled(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of no injection if traits disabled.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_explicit_rng=False)}
  code: str = "def f(x):\n  return torch.dropout(x)"
  res: str = rewrite_code(rewriter, code)
  assert "rng" not in res
  assert "split" not in res


def test_rng_deduplication(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of rng deduplication.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f(x):\n  torch.dropout(x)\n  torch.dropout(x)"
  res: str = rewrite_code(rewriter, code)
  assert res.count("split(rng)") == 1


def test_remove_generator_arg(rewriter: PivotRewriter) -> None:
  """Removes generator argument.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  code: str = "def f(x):\n  torch.dropout(x, generator=g)"
  res: str = rewrite_code(rewriter, code)
  assert "generator" not in res
