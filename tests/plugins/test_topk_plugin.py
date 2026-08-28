"""Test suite for the Topk Plugin module."""

import pytest
import libcst as cst
from typing import Callable, Dict, Any, Union
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.topk import transform_topk


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  new_tree: cst.Module = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory() -> Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: Factory instance.
  """
  hooks._HOOKS["topk_adapter"] = transform_topk
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  topk_def: Dict[str, Any] = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"},
      "tensorflow": {"api": "tf.math.top_k", "requires_plugin": "topk_adapter"},
    }
  }
  mgr.get_definition.side_effect = lambda n: ("TopK", topk_def) if "topk" in n else None

  def resolve(aid: str, fw: str) -> Any:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Framework string.

    Returns:
        Any: Variant definition.
    """
    if aid == "TopK" and fw in topk_def["variants"]:
      return topk_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True

  def create(target: str) -> PivotRewriter:
    """Creates rewriter.

    Args:
        target (str): Target framework string.

    Returns:
        PivotRewriter: The rewriter.
    """
    cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_topk_rewrapping_jax(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of topk rewrapping JAX.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("jax")
  code: str = "res = torch.topk(x, 5)"
  res: str = rewrite_code(rewriter, code)
  assert "jax.lax.top_k" in res
  assert "collections.namedtuple" in res


def test_topk_passthrough_missing_target(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of topk passthrough missing target.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rewriter: PivotRewriter = rewriter_factory("numpy")
  rewriter.context.hook_context.target_fw = "numpy"
  code: str = "res = torch.topk(x, 5)"
  res: str = rewrite_code(rewriter, code)
  assert "torch.topk" in res
  assert "collections.namedtuple" not in res


def test_topk_kwargs_stripping(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies that kwargs like largest, sorted, out are stripped.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.topk(5, largest=True, sorted=False, out=None, dim=1)"
  res: str = rewrite_code(r, code)
  assert "largest=" not in res
  assert "sorted=" not in res
  assert "out=" not in res
  assert "dim=1" in res


def test_topk_missing_api() -> None:
  """Verifies behavior when target API is missing."""
  node: cst.Call = cst.Call(func=cst.Name("topk"), args=[])
  ctx: MagicMock = MagicMock()
  ctx.lookup_api.return_value = None
  res: Union[cst.CSTNode, cst.Call] = transform_topk(node, ctx)
  assert res is node
