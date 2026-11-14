"""Test suite for the Rewriter Defaults module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for rewriter defaults tests.

  Args:
      monkeypatch: Pytest monkeypatch fixture.

  Returns:
      Configured `manager` fixture value (SemanticsManager).
  """
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._reverse_index = {}
  mgr._key_origins = {}
  mgr.framework_configs = {}
  monkeypatch.setattr(mgr, "import_data", {}, raising=False)
  op: dict[str, JsonValue] = {
    "std_args": [{"name": "x"}, {"name": "eps", "type": "float", "default": 1e-05}],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm", "args": {"eps": "eps"}},
      "jax": {"api": "jax.nn.layer_norm", "args": {"eps": "epsilon"}},
    },
  }
  mgr.data["LayerNorm"] = op
  mgr._reverse_index["torch.nn.LayerNorm"] = ("LayerNorm", op)
  op_drop: dict[str, JsonValue] = {
    "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": 0.5}],
    "variants": {"torch": {"api": "torch.dropout"}, "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}}},
  }
  mgr.data["Dropout"] = op_drop
  mgr._reverse_index["torch.dropout"] = ("Dropout", op_drop)
  mgr.framework_configs["torch"] = {"alias": {"module": "torch", "name": "t"}}
  mgr.framework_configs["jax"] = {}

  # Bind mocked lookup methods
  def mock_get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from mock get def.
    """
    return mgr._reverse_index.get(name)

  def mock_res_var(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from mock res var.
    """
    op_data = mgr.data.get(aid)
    if op_data is None:
      return None
    variants = op_data.get("variants")
    if not isinstance(variants, dict):
      return None
    res = variants.get(fw)
    return res if isinstance(res, dict) else None

  monkeypatch.setattr(mgr, "get_definition", mock_get_def, raising=False)
  monkeypatch.setattr(mgr, "resolve_variant", mock_res_var, raising=False)
  monkeypatch.setattr(mgr, "is_verified", lambda x: True, raising=False)

  return mgr


@pytest.fixture
def rewriter(manager: SemanticsManager) -> TestRewriter:
  """Create a configured `TestRewriter` fixture for rewriter defaults tests.

  Args:
      manager: Input `manager` parameter value (`SemanticsManager`).

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(manager, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Execute `rewriter.convert` to support rewrite in rewriter defaults tests.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


def test_inject_default_float(rewriter: TestRewriter) -> None:
  """Injects default float.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch\ny = torch.nn.LayerNorm(x)"
  res: str = rewrite(rewriter, code)
  assert "epsilon=1e-05" in res or "epsilon=0.00001" in res


def test_preserve_explicit_eps(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `epsilon=0.1` for preserve explicit eps.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch\ny = torch.nn.LayerNorm(x, eps=0.1)"
  res: str = rewrite(rewriter, code)
  assert "epsilon=0.1" in res
  assert "1e-5" not in res


def test_inject_default_dropout(rewriter: TestRewriter) -> None:
  """Injects default dropout.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "import torch\ny = torch.dropout(x)"
  res: str = rewrite(rewriter, code)
  assert "p=0.5" in res
