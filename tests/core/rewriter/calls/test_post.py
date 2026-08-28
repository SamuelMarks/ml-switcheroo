"""Test suite for post.py"""

import libcst as cst
import typing
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.signature_stack: list[typing.Any] = []


class MockSemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self._key_origins: dict[str, str] = {"my_id": "neural_ops"}
    self.known_magic_args: set[str] = set()


class MockTraits:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.inject_magic_args: list[tuple[str, str]] = [("is_training", "True")]
    self.strip_magic_args: list[str] = ["training"]
    self.auto_strip_magic_args: bool = False


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self._report_failure = MagicMock()

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Docstring."""
    return cst.Name(name)

  def _get_target_traits(self) -> MockTraits:
    """Docstring."""
    return MockTraits()


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  expr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr)


def test_handle_post_processing_output_select_index() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping: dict[str, typing.Any] = {"output_select_index": 1}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "id")  # type: ignore
  # Result should be func()[1]
  assert isinstance(result, cst.Subscript)
  assert typing.cast(cst.Integer, typing.cast(cst.Index, result.slice[0].slice).value).value == "1"


def test_handle_post_processing_output_select_index_failure() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  # Force failure in apply_index_select by giving an invalid index type for CST Subscript builder
  mapping: dict[str, typing.Any] = {"output_select_index": "not_an_int"}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "id")  # type: ignore
  # Should report failure and return original node
  rewriter._report_failure.assert_called_once()
  assert result == node


def test_handle_post_processing_output_cast() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping: dict[str, typing.Any] = {"output_cast": "float32"}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "id")  # type: ignore
  # func().astype(jnp.float32)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert typing.cast(cst.Name, result.func.attr).value == "astype"
  assert typing.cast(cst.Name, result.args[0].value).value == "float32"


def test_handle_post_processing_output_cast_failure() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  # Mock _create_dotted_name to raise
  rewriter._create_dotted_name = MagicMock(side_effect=ValueError)  # type: ignore

  node = parse_call("func()")
  mapping: dict[str, typing.Any] = {"output_cast": "invalid"}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "id")  # type: ignore
  assert result == node


class MockSignature:
  """Docstring."""

  def __init__(self, is_init: bool, is_module_method: bool) -> None:
    """Docstring."""
    self.is_init = is_init
    self.is_module_method = is_module_method


def test_handle_post_processing_state_threading_neural() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  node = parse_call("func(training=True)")
  mapping: dict[str, typing.Any] = {}

  # neural_ops tier, will inject and strip
  result: typing.Any = handle_post_processing(rewriter, node, mapping, "my_id")  # type: ignore

  assert isinstance(result, cst.Call)
  args: list[str] = [typing.cast(cst.Name, arg.keyword).value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_force() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  # Not neural ops
  rewriter.semantics._key_origins = {"other_id": "math"}

  # But forced because magic arg 'training' is present
  node = parse_call("func(training=True)")
  mapping: dict[str, typing.Any] = {}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "other_id")  # type: ignore

  args: list[str] = [typing.cast(cst.Name, arg.keyword).value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_auto_strip() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  traits = MockTraits()
  traits.auto_strip_magic_args = True
  rewriter.semantics.known_magic_args = {"extra_magic"}
  rewriter._get_target_traits = lambda: traits  # type: ignore

  node = parse_call("func(training=True, extra_magic=1)")
  mapping: dict[str, typing.Any] = {}

  result: typing.Any = handle_post_processing(rewriter, node, mapping, "my_id")  # type: ignore

  args: list[str] = [typing.cast(cst.Name, arg.keyword).value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args
  assert "extra_magic" not in args
