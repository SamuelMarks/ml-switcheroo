"""Test suite for the Calls Extra module."""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
  rewrite_as_infix,
)
from ml_switcheroo_ir.schema.ghost import SemanticTier


class MockHookContext:
  """Mock Hook Context class for testing purposes."""

  def __init__(self):
    """Initializes the MockHookContext instance."""
    self.metadata = {}
    self.preambles = []

  def inject_preamble(self, code):
    """Mock implementation of inject preamble."""
    self.preambles.append(code)


class MockContext:
  """Mock Context class for testing purposes."""

  def __init__(self):
    """Initializes the MockContext instance."""
    self.hook_context = MockHookContext()
    self.signature_stack = []


class MockSemantics:
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self._key_origins = {"abs_1": SemanticTier.NEURAL.value}
    self.known_magic_args = {"training"}


class MockTraits:
  """Mock Traits class for testing purposes."""

  def __init__(self):
    """Initializes the MockTraits instance."""
    self.strip_magic_args = ["training"]
    self.auto_strip_magic_args = True
    self.inject_magic_args = [("injected", "True")]


class MockSignature:
  """Mock Signature class for testing purposes."""

  def __init__(self, is_init=True, is_module_method=True):
    """Initializes the MockSignature instance."""
    self.is_init = is_init
    self.is_module_method = is_module_method


class MockRewriter:
  """Mock Rewriter class for testing purposes."""

  def __init__(self):
    """Initializes the MockRewriter instance."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self.failures = []

  def _get_target_traits(self):
    """Mock implementation of  get target traits."""
    return MockTraits()

  def _create_dotted_name(self, name):
    """Mock implementation of  create dotted name."""
    if name == "fail":
      raise ValueError("fail")
    return cst.Name("float32")

  def _report_failure(self, msg):
    """Mock implementation of  report failure."""
    self.failures.append(msg)


def test_apply_strict_guards():
  """Applies strict guards."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(value=cst.Name("x"), keyword=cst.Name("x")), cst.Arg(value=cst.Name("y"), keyword=cst.Name("y"))]
  details = {"std_args": [{"name": "x", "rank": 2}, {"name": "y"}]}
  target_impl = {"args": {"x": "target_x"}}
  assert apply_strict_guards(rewriter, norm_args, {"std_args": []}, {}) == norm_args
  norm_args_2 = [
    cst.Arg(value=cst.Name("a"), keyword=cst.Name("target_x")),
    cst.Arg(value=cst.Name("b"), keyword=cst.Name("x")),
    cst.Arg(value=cst.Name("c")),
  ]
  new_args = apply_strict_guards(rewriter, norm_args_2, details, target_impl)
  assert len(new_args) == 3
  assert isinstance(new_args[0].value, cst.Call)
  assert new_args[0].value.func.value == "_check_rank"
  assert isinstance(new_args[1].value, cst.Call)
  assert isinstance(new_args[2].value, cst.Name)
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected") is True
  assert len(rewriter.context.hook_context.preambles) == 1


def test_handle_post_processing():
  """Handles post processing."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[])
  mapping = {"output_select_index": 0}
  res = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert isinstance(res, cst.Subscript)
  mapping = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, cst.Pass(), mapping, "abs_1")
  assert len(rewriter.failures) > 0
  mapping = {"output_cast": "float32"}
  res3 = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert isinstance(res3, cst.Call)
  assert isinstance(res3.func, cst.Attribute)
  assert res3.func.attr.value == "astype"
  mapping = {"output_cast": "fail"}
  res4 = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert res4 == node
  rewriter.context.signature_stack.append(MockSignature())
  node_with_args = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("True"), keyword=cst.Name("training"))])
  res5 = handle_post_processing(rewriter, node_with_args, {}, "abs_1")
  assert isinstance(res5, cst.Call)
  rewriter.semantics._key_origins["abs_2"] = "other"
  res6 = handle_post_processing(rewriter, node_with_args, {}, "abs_2")
  assert isinstance(res6, cst.Call)


def test_apply_index_select():
  """Applies index select."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  res = apply_index_select(node, 1)
  assert isinstance(res, cst.Subscript)
  assert res.slice[0].slice.value.value == "1"


def test_rewrite_as_inline_lambda():
  """Rewrites as inline lambda."""
  args = [cst.Arg(value=cst.Name("x"))]
  res = rewrite_as_inline_lambda("lambda a: a + 1", args)
  assert isinstance(res, cst.Call)
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda a: +++", args)


def test_rewrite_as_macro():
  """Rewrites as macro."""
  args = [cst.Arg(value=cst.Name("x_val"))]
  res = rewrite_as_macro("{x} * 2", args, ["x"])
  assert isinstance(res, cst.BinaryOperation)
  with pytest.raises(ValueError, match="Macro template requires argument 'y'"):
    rewrite_as_macro("{y} * 2", args, ["x"])
  with pytest.raises(ValueError, match="invalid python"):
    rewrite_as_macro("{x} * +++", args, ["x"])


def test_rewrite_as_infix():
  """Rewrites as infix."""
  original = cst.Call(func=cst.Name("foo"), args=[])
  args_1 = [cst.Arg(value=cst.Name("x"))]
  args_2 = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]
  res1 = rewrite_as_infix(original, args_1, "-", ["x"])
  assert isinstance(res1, cst.UnaryOperation)
  args_bin = [cst.Arg(value=cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b")))]
  res1b = rewrite_as_infix(original, args_bin, "-", ["x"])
  assert isinstance(res1b, cst.UnaryOperation)
  assert len(res1b.expression.lpar) > 0
  with pytest.raises(ValueError, match="expects 1 argument"):
    rewrite_as_infix(original, [], "-", ["x"])
  with pytest.raises(ValueError, match="Unsupported unary"):
    rewrite_as_infix(original, args_1, "???", ["x"])
  res2 = rewrite_as_infix(original, args_2, "+", ["x", "y"])
  assert isinstance(res2, cst.BinaryOperation)
  with pytest.raises(ValueError, match="requires 2 arguments"):
    rewrite_as_infix(original, args_1, "+", ["x", "y"])
  with pytest.raises(ValueError, match="Unsupported binary"):
    rewrite_as_infix(original, args_2, "???", ["x", "y"])
  with pytest.raises(ValueError, match="requires 1 or 2 args"):
    rewrite_as_infix(original, args_2, "+", ["x", "y", "z"])
