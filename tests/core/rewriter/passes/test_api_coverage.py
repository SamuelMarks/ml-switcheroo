"""Module docstring."""

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_resolution_pass_import_invalid_alias():
  """Docstring."""

  class DummyImportNode:
    names = ["not_an_alias"]
    module = None

  ctx = RewriterContext("test", RuntimeConfig())
  p = ApiTransformer(ctx)
  p.visit_Import(DummyImportNode())

  class DummyImportFromNode:
    names = ["not_an_alias"]
    module = cst.Name("dummy")
    relative = []

  p.visit_ImportFrom(DummyImportFromNode())


def test_api_resolution_pass_import_invalid_alias_name():
  """Docstring."""

  class MockAlias:
    name = "not_name_or_attr"

  class DummyImportNode:
    names = [MockAlias()]
    module = None

  ctx = RewriterContext("test", RuntimeConfig())
  p = ApiTransformer(ctx)
  import builtins

  original_isinstance = builtins.isinstance

  def patched_isinstance(obj, class_or_tuple):
    if obj.__class__.__name__ == "MockAlias" and class_or_tuple == cst.ImportAlias:
      return True
    return original_isinstance(obj, class_or_tuple)

  builtins.isinstance = patched_isinstance
  try:
    p.visit_Import(DummyImportNode())

    class DummyImportFromNode:
      module = cst.Name("dummy")
      relative = []
      names = [MockAlias()]

    p.visit_ImportFrom(DummyImportFromNode())
  finally:
    builtins.isinstance = original_isinstance
