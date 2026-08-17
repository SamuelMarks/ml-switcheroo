"""Test suite for post.py"""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing


class MockContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.signature_stack = []


class MockSemantics:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self._key_origins = {"my_id": "neural_ops"}
    self.known_magic_args = set()


class MockTraits:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.inject_magic_args = [("is_training", "True")]
    self.strip_magic_args = ["training"]
    self.auto_strip_magic_args = False


class MockRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self._report_failure = MagicMock()

  def _create_dotted_name(self, name):
    return cst.Name(name)

  def _get_target_traits(self):
    return MockTraits()


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  return module.body[0].body[0].value


def test_handle_post_processing_output_select_index():
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping = {"output_select_index": 1}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # Result should be func()[1]
  assert isinstance(result, cst.Subscript)
  assert result.slice[0].slice.value.value == "1"


def test_handle_post_processing_output_select_index_failure():
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  # Force failure in apply_index_select by giving an invalid index type for CST Subscript builder
  mapping = {"output_select_index": "not_an_int"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # Should report failure and return original node
  rewriter._report_failure.assert_called_once()
  assert result == node


def test_handle_post_processing_output_cast():
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("func()")
  mapping = {"output_cast": "float32"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  # func().astype(jnp.float32)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "astype"
  assert result.args[0].value.value == "float32"


def test_handle_post_processing_output_cast_failure():
  """Docstring."""
  rewriter = MockRewriter()
  # Mock _create_dotted_name to raise
  rewriter._create_dotted_name = MagicMock(side_effect=ValueError)

  node = parse_call("func()")
  mapping = {"output_cast": "invalid"}

  result = handle_post_processing(rewriter, node, mapping, "id")
  assert result == node


class MockSignature:
  """Docstring."""

  def __init__(self, is_init, is_module_method):
    """Docstring."""
    self.is_init = is_init
    self.is_module_method = is_module_method


def test_handle_post_processing_state_threading_neural():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  node = parse_call("func(training=True)")
  mapping = {}

  # neural_ops tier, will inject and strip
  result = handle_post_processing(rewriter, node, mapping, "my_id")

  assert isinstance(result, cst.Call)
  args = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_force():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  # Not neural ops
  rewriter.semantics._key_origins = {"other_id": "math"}

  # But forced because magic arg 'training' is present
  node = parse_call("func(training=True)")
  mapping = {}

  result = handle_post_processing(rewriter, node, mapping, "other_id")

  args = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args


def test_handle_post_processing_state_threading_auto_strip():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.signature_stack.append(MockSignature(is_init=True, is_module_method=True))

  traits = MockTraits()
  traits.auto_strip_magic_args = True
  rewriter.semantics.known_magic_args = {"extra_magic"}
  rewriter._get_target_traits = lambda: traits

  node = parse_call("func(training=True, extra_magic=1)")
  mapping = {}

  result = handle_post_processing(rewriter, node, mapping, "my_id")

  args = [arg.keyword.value for arg in result.args if arg.keyword]
  assert "is_training" in args
  assert "training" not in args
  assert "extra_magic" not in args
