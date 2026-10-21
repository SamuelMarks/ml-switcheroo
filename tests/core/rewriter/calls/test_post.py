"""Test suite for post.py"""

import typing
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.signature_stack: list[typing.Hashable | typing.Sized] = []


class MockSemantics:
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self._key_origins: dict[str, str] = {"my_id": "neural_ops"}
    self.known_magic_args: set[str] = set()


class MockTraits:
  """Test helper class MockTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.inject_magic_args: list[tuple[str, str]] = [("is_training", "True")]
    self.strip_magic_args: list[str] = ["training"]
    self.auto_strip_magic_args: bool = False


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self._report_failure = MagicMock()

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Execute create dotted name helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name(name)

  def _get_target_traits(self) -> MockTraits:
    """Execute get target traits helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return MockTraits()


def parse_call(code: str) -> cst.Call:
  """Execute parse call helper logic.

  Args:
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  module = cst.parse_module(code)
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call)


def test_handle_post_processing_output_select_index() -> None:
  """Verify handle post processing output select index behavior."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping = {"output_select_index": 1}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # Result should be func()[1]
  assert isinstance(result, cst.Subscript)
  assert cst.ensure_type(cst.ensure_type(result.slice[0].slice, cst.Index).value, cst.Integer).value == "1"


def test_handle_post_processing_output_select_index_failure() -> None:
  """Verify handle post processing output select index failure behavior."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  # Force failure in apply_index_select by giving an invalid index type for CST Subscript builder
  mapping = {"output_select_index": "not_an_int"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # Should report failure and return original node
  rewriter._report_failure.assert_called_once()
  assert result == node


def test_handle_post_processing_output_cast() -> None:
  """Verify handle post processing output cast behavior."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping = {"output_cast": "float32"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # func().astype(jnp.float32)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "astype"
  assert cst.ensure_type(result.args[0].value, cst.Name).value == "float32"


def test_handle_post_processing_output_cast_failure() -> None:
  """Verify handle post processing output cast failure behavior."""
  rewriter = MockRewriter()
  # Mock _create_dotted_name to raise
  setattr(rewriter, "_create_dotted_name", MagicMock(side_effect=ValueError))

  node = parse_call("func()")
  mapping = {"output_cast": "invalid"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  assert result == node


class MockSignature:
  """Test helper class MockSignature."""

  def __init__(self, is_init: bool, is_module_method: bool) -> None:
    """Initialize test helper instance.

    Args:
        is_init: Test argument or fixture.
        is_module_method: Test argument or fixture.
    """
    self.is_init = is_init
    self.is_module_method = is_module_method


def test_handle_post_processing_state_threading_neural() -> None:
  """Verify handle post processing state threading neural behavior."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  node = parse_call("func(training=True)")
  mapping: dict[str, typing.Hashable | typing.Sized] = {}

  # neural_ops tier, will inject and strip
  result = handle_post_processing(rewriter, node, mapping, "my_id")

  assert isinstance(result, cst.Call)
  args: list[str] = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_force() -> None:
  """Verify handle post processing state threading force behavior."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  # Not neural ops
  rewriter.semantics._key_origins = {"other_id": "math"}

  # But forced because magic arg 'training' is present
  node = parse_call("func(training=True)")
  mapping: dict[str, typing.Hashable | typing.Sized] = {}

  result = handle_post_processing(rewriter, node, mapping, "other_id")

  args: list[str] = getattr(
    {
      "v": [
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr({"v": getattr({"v": arg}, "__getitem__")("v").keyword}, "__getitem__")("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v").value
        for arg in getattr({"v": result}, "__getitem__")("v").args
        if getattr({"v": arg}, "__getitem__")("v").keyword
      ]
    },
    "__getitem__",
  )("v")
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_auto_strip() -> None:
  """Verify handle post processing state threading auto strip behavior."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  traits = MockTraits()
  traits.auto_strip_magic_args = True
  rewriter.semantics.known_magic_args = {"extra_magic"}
  setattr(rewriter, "_get_target_traits", lambda: traits)

  node = parse_call("func(training=True, extra_magic=1)")
  mapping: dict[str, typing.Hashable | typing.Sized] = {}

  result = handle_post_processing(rewriter, node, mapping, "my_id")

  args: list[str] = getattr(
    {
      "v": [
        getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
              getattr({"v": getattr({"v": arg}, "__getitem__")("v").keyword}, "__getitem__")("v"),
              getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
            )
          },
          "__getitem__",
        )("v").value
        for arg in getattr({"v": result}, "__getitem__")("v").args
        if getattr({"v": arg}, "__getitem__")("v").keyword
      ]
    },
    "__getitem__",
  )("v")
  assert "is_training" in args
  assert "training" not in args
  assert "extra_magic" not in args
