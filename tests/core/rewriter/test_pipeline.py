"""Unit tests for `RewriterPipeline` and `RewriterPass` (`ml_switcheroo.core.rewriter.pipeline`)."""

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.pipeline import RewriterPipeline
from ml_switcheroo.semantics.manager import SemanticsManager


class _HeaderCommentPass(RewriterPass):
  """Concrete `RewriterPass` appending a labeled comment line to `module.header`."""

  def __init__(self, label: str) -> None:
    """Initialize pass label.

    Args:
        label: Marker label written into the module header comment.
    """
    self.label = label

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Append `# Pass: <label>` to `module.header`.

    Args:
        module: Input CST module.
        context: Active `RewriterContext`.

    Returns:
        Updated `cst.Module` with the header comment appended.
    """
    header = list(module.header)
    header.append(cst.EmptyLine(comment=cst.Comment(f"# Pass: {self.label}"), newline=cst.Newline()))
    return module.with_changes(header=header)


def test_pipeline_ordered_execution_and_empty_noop() -> None:
  """Verify `RewriterPipeline.run` threads the `cst.Module` sequentially through passes and handles `[]`."""
  ctx = RewriterContext(
    semantics=SemanticsManager(),
    config=RuntimeConfig(source_framework="torch", target_framework="jax"),
  )
  module = cst.parse_module("x = 1")

  result = RewriterPipeline([_HeaderCommentPass("A"), _HeaderCommentPass("B")]).run(module, ctx)
  code = result.code
  assert "# Pass: A" in code
  assert "# Pass: B" in code
  assert code.index("# Pass: A") < code.index("# Pass: B")
  assert "x = 1" in code

  empty_res = RewriterPipeline([]).run(module, ctx)
  assert empty_res.code == "x = 1"


def test_rewriter_pass_abc_enforcement() -> None:
  """Verify `RewriterPass` declares `transform` abstract and rejects incomplete subclasses at runtime."""
  assert RewriterPass.__abstractmethods__ == frozenset({"transform"})
  incomplete_cls = type("_IncompletePass", (RewriterPass,), {})
  with pytest.raises(TypeError):
    incomplete_cls()

  ctx = RewriterContext(semantics=SemanticsManager(), config=RuntimeConfig())
  assert RewriterPass.transform(_HeaderCommentPass("Base"), cst.parse_module("x = 1"), ctx) is None

  del ctx.hook_registry
  assert RewriterPipeline([]).run(cst.parse_module("x = 1"), ctx).code == "x = 1"
