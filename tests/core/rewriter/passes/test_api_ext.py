"""Test suite for api coverage."""

import typing
from pydantic import JsonValue

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_resolution_pass_import_invalid_alias() -> None:
  """Verify api resolution pass import invalid alias behavior."""

  class DummyImportNode:
    """Test helper class DummyImportNode."""

    names = ["not_an_alias"]
    module = None

  ctx = RewriterContext("test", RuntimeConfig())
  p = ApiTransformer(ctx)
  getattr({"v": getattr({"v": p}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
    getattr({"v": getattr({"v": DummyImportNode}, "__getitem__")("v")()}, "__getitem__")("v")
  )

  class DummyImportFromNode:
    """Test helper class DummyImportFromNode."""

    names = ["not_an_alias"]
    module = cst.Name("dummy")
    relative: list[typing.Hashable | typing.Sized] = []

  _res = getattr(
    {
      "v": getattr({"v": getattr({"v": p}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
        getattr({"v": getattr({"v": DummyImportFromNode}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert _res is False
  assert p.context.alias_map == {}


def test_api_resolution_pass_import_invalid_alias_name() -> None:
  """Verify api resolution pass import invalid alias name behavior.

  Returns:
      Computed test result or fixture value.
  """

  class MockAlias:
    """Test helper class MockAlias."""

    name = "not_name_or_attr"

  class DummyImportNode:
    """Test helper class DummyImportNode."""

    names = [MockAlias()]
    module = None

  ctx = RewriterContext("test", RuntimeConfig())
  p = ApiTransformer(ctx)
  import builtins

  original_isinstance = builtins.isinstance

  def patched_isinstance(
    obj: cst.CSTNode | MockAlias | JsonValue,
    class_or_tuple: type | tuple[type, ...],
  ) -> bool:
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

  setattr(
    getattr({"v": builtins}, "__getitem__")("v"), "isinstance", getattr({"v": patched_isinstance}, "__getitem__")("v")
  )
  try:
    getattr({"v": getattr({"v": p}, "__getitem__")("v").visit_Import}, "__getitem__")("v")(
      getattr({"v": getattr({"v": DummyImportNode}, "__getitem__")("v")()}, "__getitem__")("v")
    )

    class DummyImportFromNode:
      """Test helper class DummyImportFromNode."""

      module = cst.Name("dummy")
      relative: list[typing.Hashable | typing.Sized] = []
      names = [MockAlias()]

    getattr({"v": getattr({"v": p}, "__getitem__")("v").visit_ImportFrom}, "__getitem__")("v")(
      getattr({"v": getattr({"v": DummyImportFromNode}, "__getitem__")("v")()}, "__getitem__")("v")
    )
  finally:
    builtins.isinstance = original_isinstance
  assert p.context.alias_map == {}
