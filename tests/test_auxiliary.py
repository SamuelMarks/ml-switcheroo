"""Tests for the auxiliary transformer."""

from pydantic import JsonValue


import libcst as _cst

from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer

cst = _cst


class MockSemantics:
  """Test helper class MockSemantics."""

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Retrieve a mock definition based on name.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition tuple.
    """
    if name == "my_dec":
      return "id", {"variants": {"other_fw": {}}}
    return None


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize dummy context."""
    self.target_framework = "flax_nnx"
    self.semantics = MockSemantics()


def test_auxiliary_leave_decorator_no_name() -> None:
  """Verify auxiliary leave decorator no name behavior."""
  p = getattr(
    {
      "v": getattr({"v": AuxiliaryTransformer}, "__getitem__")("v")(
        context=getattr({"v": getattr({"v": DummyContext}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                        getattr({"v": "@foo()()\ndef x(): pass"}, "__getitem__")("v")
                      )
                    },
                    "__getitem__",
                  )("v").body
                },
                "__getitem__",
              )("v")[0]
            },
            "__getitem__",
          )("v").decorators
        },
        "__getitem__",
      )("v")[0]
    },
    "__getitem__",
  )("v")

  # We must add an _get_qualified_name method that returns None on the transformer
  # to mock it, or just let it return None naturally for complex calls.
  # Actually _get_qualified_name is a method on AuxiliaryTransformer.
  res = p.leave_Decorator(node, node)
  assert res == node


def test_auxiliary_leave_decorator_fw_not_in_variants() -> None:
  """Verify auxiliary leave decorator fw not in variants behavior."""
  p = getattr(
    {
      "v": getattr({"v": AuxiliaryTransformer}, "__getitem__")("v")(
        context=getattr({"v": getattr({"v": DummyContext}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  node = getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").parse_module}, "__getitem__")("v")(
                        getattr({"v": "@my_dec\ndef x(): pass"}, "__getitem__")("v")
                      )
                    },
                    "__getitem__",
                  )("v").body
                },
                "__getitem__",
              )("v")[0]
            },
            "__getitem__",
          )("v").decorators
        },
        "__getitem__",
      )("v")[0]
    },
    "__getitem__",
  )("v")

  # Needs a mock _get_qualified_name to return "my_dec"
  setattr(p, "_get_qualified_name", lambda x: "my_dec")

  res = p.leave_Decorator(node, node)
  assert res == node
