"""Test suite for api coverage."""

import typing

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_resolution_pass_import_invalid_alias() -> None:
  """Verify api resolution pass import invalid alias behavior."""

  class DummyImportNode:
    """Test helper class DummyImportNode."""

    names: typing.Any = ["not_an_alias"]
    module: typing.Any = None

  ctx: typing.Any = RewriterContext("test", RuntimeConfig())
  p: typing.Any = ApiTransformer(ctx)
  p.visit_Import(DummyImportNode())

  class DummyImportFromNode:
    """Test helper class DummyImportFromNode."""

    names: typing.Any = ["not_an_alias"]
    module: typing.Any = cst.Name("dummy")
    relative: typing.Any = []

  _res: typing.Any = p.visit_ImportFrom(DummyImportFromNode())
  assert _res is False
  assert p.context.alias_map == {}


def test_api_resolution_pass_import_invalid_alias_name() -> typing.Any:
  """Verify api resolution pass import invalid alias name behavior.

  Returns:
      Computed test result or fixture value.
  """

  class MockAlias:
    """Test helper class MockAlias."""

    name: typing.Any = "not_name_or_attr"

  class DummyImportNode:
    """Test helper class DummyImportNode."""

    names: typing.Any = [MockAlias()]
    module: typing.Any = None

  ctx: typing.Any = RewriterContext("test", RuntimeConfig())
  p: typing.Any = ApiTransformer(ctx)
  import builtins

  original_isinstance: typing.Any = builtins.isinstance

  def patched_isinstance(obj: typing.Any, class_or_tuple: typing.Any) -> typing.Any:
    """Execute patched isinstance helper logic.

    Args:
        obj: Test argument or fixture.
        class_or_tuple: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if obj.__class__.__name__ == "MockAlias" and class_or_tuple == cst.ImportAlias:
      return True
    return original_isinstance(obj, class_or_tuple)

  builtins.isinstance = patched_isinstance
  try:
    p.visit_Import(DummyImportNode())

    class DummyImportFromNode:
      """Test helper class DummyImportFromNode."""

      module: typing.Any = cst.Name("dummy")
      relative: typing.Any = []
      names: typing.Any = [MockAlias()]

    p.visit_ImportFrom(DummyImportFromNode())
  finally:
    builtins.isinstance = original_isinstance
  assert p.context.alias_map == {}
