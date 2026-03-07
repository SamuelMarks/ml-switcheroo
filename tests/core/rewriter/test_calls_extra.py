"""Module docstring."""

import pytest
import libcst as cst
from typing import Any, Dict

from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
  rewrite_as_infix,
)
from ml_switcheroo.enums import SemanticTier


# --- Mocks ---
class MockHookContext:
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.metadata = {}
    self.preambles = []

  def inject_preamble(self, code):
    """Function docstring."""
    self.preambles.append(code)


class MockContext:
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.hook_context = MockHookContext()
    self.signature_stack = []


class MockSemantics:
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self._key_origins = {"abs_1": SemanticTier.NEURAL.value}
    self.known_magic_args = {"training"}


class MockTraits:
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.strip_magic_args = ["training"]
    self.auto_strip_magic_args = True
    self.inject_magic_args = [("injected", "True")]


class MockSignature:
  """Class docstring."""

  def __init__(self, is_init=True, is_module_method=True):
    """Function docstring."""
    self.is_init = is_init
    self.is_module_method = is_module_method


class MockRewriter:
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self.failures = []

  def _get_target_traits(self):
    """Function docstring."""
    return MockTraits()

  def _create_dotted_name(self, name):
    """Function docstring."""
    if name == "fail":
      raise ValueError("fail")
    return cst.Name("float32")

  def _report_failure(self, msg):
    """Function docstring."""
    self.failures.append(msg)


# --- test_guards.py ---
def test_apply_strict_guards():
  """Function docstring."""
  rewriter = MockRewriter()
  norm_args = [
    cst.Arg(value=cst.Name("x"), keyword=cst.Name("x")),
    cst.Arg(value=cst.Name("y"), keyword=cst.Name("y")),
  ]
  details = {"std_args": [{"name": "x", "rank": 2}, {"name": "y"}]}
  target_impl = {
    "args": {"x": "target_x"}  # Maps std name 'x' to target name 'target_x'
  }

  # 1. Test when guards_map is empty
  assert apply_strict_guards(rewriter, norm_args, {"std_args": []}, {}) == norm_args

  # 2. Test standard logic
  norm_args_2 = [
    cst.Arg(value=cst.Name("a"), keyword=cst.Name("target_x")),
    cst.Arg(value=cst.Name("b"), keyword=cst.Name("x")),
    cst.Arg(value=cst.Name("c")),
  ]
  new_args = apply_strict_guards(rewriter, norm_args_2, details, target_impl)
  assert len(new_args) == 3
  # First arg should be wrapped
  assert isinstance(new_args[0].value, cst.Call)
  assert new_args[0].value.func.value == "_check_rank"
  # Second arg also wrapped (fallback logic `elif arg_key in guards_map`)
  assert isinstance(new_args[1].value, cst.Call)
  # Third arg unmodified
  assert isinstance(new_args[2].value, cst.Name)

  # Check preamble injection
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected") is True
  assert len(rewriter.context.hook_context.preambles) == 1


# --- test_post.py ---
def test_handle_post_processing():
  """Function docstring."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[])

  # 1. Output Adaptation (output_select_index)
  mapping = {"output_select_index": 0}
  res = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert isinstance(res, cst.Subscript)

  # Output selection failure
  mapping = {"output_select_index": "invalid"}  # will raise in apply_index_select maybe? or not
  # apply_index_select safely uses str(index) though. Wait, if we pass something that raises...
  # cst.Subscript requires valid node. If inner_node is a Statement, it raises.
  res2 = handle_post_processing(rewriter, cst.Pass(), mapping, "abs_1")
  assert len(rewriter.failures) > 0

  # 2. Output Casting
  mapping = {"output_cast": "float32"}
  res3 = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert isinstance(res3, cst.Call)
  assert isinstance(res3.func, cst.Attribute)
  assert res3.func.attr.value == "astype"

  # Casting failure (ignored)
  mapping = {"output_cast": "fail"}
  res4 = handle_post_processing(rewriter, node, mapping, "abs_1")
  assert res4 == node

  # 3. State Threading in Constructors
  rewriter.context.signature_stack.append(MockSignature())
  node_with_args = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("True"), keyword=cst.Name("training"))])
  res5 = handle_post_processing(rewriter, node_with_args, {}, "abs_1")
  # training is stripped, and injected back?
  # the code says: A. Inject magic args -> injects training=False.
  # B. Strip magic args -> training is in strip_magic_args but native native is {'training'}, so it does not strip the newly injected one?
  # Wait, native = {'training'}, args_to_strip = {'training'} - {'training'} = set()
  # So it doesn't strip it!
  assert isinstance(res5, cst.Call)

  # Force=True branch (if not neural but has magic arg)
  rewriter.semantics._key_origins["abs_2"] = "other"
  res6 = handle_post_processing(rewriter, node_with_args, {}, "abs_2")
  assert isinstance(res6, cst.Call)


# --- test_transformers.py ---
def test_apply_index_select():
  """Function docstring."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  res = apply_index_select(node, 1)
  assert isinstance(res, cst.Subscript)
  assert res.slice[0].slice.value.value == "1"


def test_rewrite_as_inline_lambda():
  """Function docstring."""
  args = [cst.Arg(value=cst.Name("x"))]
  res = rewrite_as_inline_lambda("lambda a: a + 1", args)
  assert isinstance(res, cst.Call)

  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda a: +++", args)


def test_rewrite_as_macro():
  """Function docstring."""
  args = [cst.Arg(value=cst.Name("x_val"))]
  res = rewrite_as_macro("{x} * 2", args, ["x"])
  assert isinstance(res, cst.BinaryOperation)

  with pytest.raises(ValueError, match="Macro template requires argument 'y'"):
    rewrite_as_macro("{y} * 2", args, ["x"])

  with pytest.raises(ValueError, match="invalid python"):
    rewrite_as_macro("{x} * +++", args, ["x"])


def test_rewrite_as_infix():
  """Function docstring."""
  original = cst.Call(func=cst.Name("foo"), args=[])
  args_1 = [cst.Arg(value=cst.Name("x"))]
  args_2 = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]

  # Arity 1 success
  res1 = rewrite_as_infix(original, args_1, "-", ["x"])
  assert isinstance(res1, cst.UnaryOperation)

  # Arity 1 wrap binary op
  args_bin = [cst.Arg(value=cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b")))]
  res1b = rewrite_as_infix(original, args_bin, "-", ["x"])
  assert isinstance(res1b, cst.UnaryOperation)
  assert len(res1b.expression.lpar) > 0

  # Arity 1 failure
  with pytest.raises(ValueError, match="expects 1 argument"):
    rewrite_as_infix(original, [], "-", ["x"])
  with pytest.raises(ValueError, match="Unsupported unary"):
    rewrite_as_infix(original, args_1, "???", ["x"])

  # Arity 2 success
  res2 = rewrite_as_infix(original, args_2, "+", ["x", "y"])
  assert isinstance(res2, cst.BinaryOperation)

  # Arity 2 failure
  with pytest.raises(ValueError, match="requires 2 arguments"):
    rewrite_as_infix(original, args_1, "+", ["x", "y"])
  with pytest.raises(ValueError, match="Unsupported binary"):
    rewrite_as_infix(original, args_2, "???", ["x", "y"])

  # Arity > 2 failure
  with pytest.raises(ValueError, match="requires 1 or 2 args"):
    rewrite_as_infix(original, args_2, "+", ["x", "y", "z"])
