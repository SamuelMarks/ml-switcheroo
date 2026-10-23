"""Test suite for api coverage."""

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.manager import SemanticsManager


def test_api_resolution_pass_import_invalid_alias() -> None:
  """Verify api resolution pass import alias tracking behavior."""
  ctx = RewriterContext(SemanticsManager(), RuntimeConfig())
  p = ApiTransformer(ctx)
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  assert p.visit_Import(imp_node) is False
  assert p.context.alias_map.get("torch") == "torch"


def test_api_resolution_pass_import_invalid_alias_name() -> None:
  """Verify api resolution pass import from dotted attribute alias tracking."""
  ctx = RewriterContext(SemanticsManager(), RuntimeConfig())
  p = ApiTransformer(ctx)
  imp_from = cst.ImportFrom(
    module=cst.Name("torch"),
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("nn"), attr=cst.Name("functional")))],
  )
  assert p.visit_ImportFrom(imp_from) is False
  assert p.context.alias_map.get("functional") == "torch.functional"
