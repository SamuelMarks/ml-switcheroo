"""Test suite for the Rewriter Defaults module."""

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


@pytest.fixture
def manager() -> SemanticsManager:
  """Execute manager helper logic.

  Returns:
      Computed test result or fixture value.
  """
  mgr = SemanticsManager()
  mgr.data = {}
  mgr._reverse_index = {}
  mgr._key_origins = {}
  mgr.framework_configs = {}
  if not hasattr(mgr, "import_data"):
    setattr(getattr({"v": mgr}, "__getitem__")("v"), "import_data", getattr({"v": {}}, "__getitem__")("v"))
  op = {
    "std_args": [{"name": "x"}, {"name": "eps", "type": "float", "default": 1e-05}],
    "variants": {
      "torch": {"api": "torch.nn.LayerNorm", "args": {"eps": "eps"}},
      "jax": {"api": "jax.nn.layer_norm", "args": {"eps": "epsilon"}},
    },
  }
  getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
    "LayerNorm", getattr({"v": op}, "__getitem__")("v")
  )
  getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
    "torch.nn.LayerNorm",
    getattr(
      {"v": (getattr({"v": "LayerNorm"}, "__getitem__")("v"), getattr({"v": op}, "__getitem__")("v"))}, "__getitem__"
    )("v"),
  )
  op_drop = {
    "std_args": [{"name": "x"}, {"name": "p", "type": "float", "default": 0.5}],
    "variants": {"torch": {"api": "torch.dropout"}, "jax": {"api": "jax.random.bernoulli", "args": {"p": "p"}}},
  }
  getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
    "Dropout", getattr({"v": op_drop}, "__getitem__")("v")
  )
  getattr(getattr({"v": getattr({"v": mgr}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__")(
    "torch.dropout",
    getattr(
      {"v": (getattr({"v": "Dropout"}, "__getitem__")("v"), getattr({"v": op_drop}, "__getitem__")("v"))}, "__getitem__"
    )("v"),
  )
  mgr.framework_configs["torch"] = {"alias": {"module": "torch", "name": "t"}}
  mgr.framework_configs["jax"] = {}

  # Bind mocked lookup methods
  def mock_get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return mgr._reverse_index.get(name)

  def mock_res_var(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    op_data = mgr.data.get(aid, {})
    variants = getattr(
      {
        "v": getattr({"v": getattr({"v": op_data}, "__getitem__")("v").get}, "__getitem__")("v")(
          getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v")
        )
        if isinstance(getattr({"v": op_data}, "__getitem__")("v"), dict)
        else {}
      },
      "__getitem__",
    )("v")
    res = variants.get(fw) if isinstance(variants, dict) else None
    return res if isinstance(res, dict) else None

  setattr(mgr, "get_definition", mock_get_def)
  setattr(mgr, "resolve_variant", mock_res_var)
  setattr(mgr, "is_verified", lambda x: True)

  return mgr


@pytest.fixture
def rewriter(manager: SemanticsManager) -> TestRewriter:
  """Execute rewriter helper logic.

  Args:
      manager: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(manager, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


def test_inject_default_float(rewriter: TestRewriter) -> None:
  """Injects default float.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "import torch\ny = torch.nn.LayerNorm(x)"
  res: str = rewrite(rewriter, code)
  assert "epsilon=1e-05" in res or "epsilon=0.00001" in res


def test_preserve_explicit_eps(rewriter: TestRewriter) -> None:
  """Verifies the behavior of preserve explicit eps.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "import torch\ny = torch.nn.LayerNorm(x, eps=0.1)"
  res: str = rewrite(rewriter, code)
  assert "epsilon=0.1" in res
  assert "1e-5" not in res


def test_inject_default_dropout(rewriter: TestRewriter) -> None:
  """Injects default dropout.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "import torch\ny = torch.dropout(x)"
  res: str = rewrite(rewriter, code)
  assert "p=0.5" in res
