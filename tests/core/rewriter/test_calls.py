"""Test suite for the Calls Extra module."""

import typing

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.transformers import (
  apply_index_select,
  rewrite_as_infix,
  rewrite_as_inline_lambda,
  rewrite_as_macro,
)


class MockHookContext:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockHookContext instance."""
    self.metadata: dict[str, typing.Any] = {}
    self.preambles: list[str] = []

  def inject_preamble(self, code: str) -> None:
    """Mock implementation of inject preamble."""
    self.preambles.append(code)


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockContext instance."""
    self.hook_context = MockHookContext()
    self.signature_stack: list[typing.Any] = []


class MockSemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self._key_origins = {"abs_1": SemanticTier.NEURAL.value}
    self.known_magic_args = {"training"}


class MockTraits:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockTraits instance."""
    self.strip_magic_args = ["training"]
    self.auto_strip_magic_args = True
    self.inject_magic_args = [("injected", "True")]


class MockSignature:
  """Docstring."""

  def __init__(self, is_init: bool = True, is_module_method: bool = True) -> None:
    """Initializes the MockSignature instance."""
    self.is_init = is_init
    self.is_module_method = is_module_method


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockRewriter instance."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self.failures: list[str] = []

  def _get_target_traits(self) -> MockTraits:
    """Mock implementation of  get target traits."""
    return MockTraits()

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Mock implementation of  create dotted name."""
    if name == "fail":
      raise ValueError("fail")
    return cst.Name("float32")

  def _report_failure(self, msg: str) -> None:
    """Mock implementation of  report failure."""
    self.failures.append(msg)


def test_apply_strict_guards() -> None:
  """Applies strict guards."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(value=cst.Name("x"), keyword=cst.Name("x")), cst.Arg(value=cst.Name("y"), keyword=cst.Name("y"))]
  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}, {"name": "y"}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "target_x"}}
  assert apply_strict_guards(rewriter, norm_args, {"std_args": []}, {}) == norm_args  # type: ignore
  norm_args_2 = [
    cst.Arg(value=cst.Name("a"), keyword=cst.Name("target_x")),
    cst.Arg(value=cst.Name("b"), keyword=cst.Name("x")),
    cst.Arg(value=cst.Name("c")),
  ]
  new_args: list[cst.Arg] = apply_strict_guards(rewriter, norm_args_2, details, target_impl)  # type: ignore
  assert len(new_args) == 3
  assert isinstance(new_args[0].value, cst.Call)
  assert typing.cast(cst.Name, new_args[0].value.func).value == "_check_rank"
  assert isinstance(new_args[1].value, cst.Call)
  assert isinstance(new_args[2].value, cst.Name)
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected") is True
  assert len(rewriter.context.hook_context.preambles) == 1


def test_handle_post_processing() -> None:
  """Handles post processing."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("foo"), args=[])
  mapping: dict[str, typing.Any] = {"output_select_index": 0}
  res: typing.Any = handle_post_processing(rewriter, node, mapping, "abs_1")  # type: ignore
  assert isinstance(res, cst.Subscript)
  mapping = {"output_select_index": "invalid"}
  handle_post_processing(rewriter, cst.Pass(), mapping, "abs_1")  # type: ignore
  assert len(rewriter.failures) > 0
  mapping = {"output_cast": "float32"}
  res3: typing.Any = handle_post_processing(rewriter, node, mapping, "abs_1")  # type: ignore
  assert isinstance(res3, cst.Call)
  assert isinstance(res3.func, cst.Attribute)
  assert typing.cast(cst.Name, res3.func.attr).value == "astype"
  mapping = {"output_cast": "fail"}
  res4: typing.Any = handle_post_processing(rewriter, node, mapping, "abs_1")  # type: ignore
  assert res4 == node
  rewriter.context.signature_stack.append(MockSignature())
  node_with_args = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Name("True"), keyword=cst.Name("training"))])
  res5: typing.Any = handle_post_processing(rewriter, node_with_args, {}, "abs_1")  # type: ignore
  assert isinstance(res5, cst.Call)
  rewriter.semantics._key_origins["abs_2"] = "other"
  res6: typing.Any = handle_post_processing(rewriter, node_with_args, {}, "abs_2")  # type: ignore
  assert isinstance(res6, cst.Call)


def test_apply_index_select() -> None:
  """Applies index select."""
  node = cst.Call(func=cst.Name("foo"), args=[])
  res: typing.Any = apply_index_select(node, 1)
  assert isinstance(res, cst.Subscript)
  assert typing.cast(cst.Integer, res.slice[0].slice.value).value == "1"  # type: ignore


def test_rewrite_as_inline_lambda() -> None:
  """Rewrites as inline lambda."""
  args = [cst.Arg(value=cst.Name("x"))]
  res: typing.Any = rewrite_as_inline_lambda("lambda a: a + 1", args)
  assert isinstance(res, cst.Call)
  with pytest.raises(ValueError, match="Invalid lambda syntax"):
    rewrite_as_inline_lambda("lambda a: +++", args)


def test_rewrite_as_macro() -> None:
  """Rewrites as macro."""
  args = [cst.Arg(value=cst.Name("x_val"))]
  res: typing.Any = rewrite_as_macro("{x} * 2", args, ["x"])
  assert isinstance(res, cst.BinaryOperation)
  with pytest.raises(ValueError, match="Macro template requires argument 'y'"):
    rewrite_as_macro("{y} * 2", args, ["x"])
  with pytest.raises(ValueError, match="invalid python"):
    rewrite_as_macro("{x} * +++", args, ["x"])


def test_rewrite_as_infix() -> None:
  """Rewrites as infix."""
  original = cst.Call(func=cst.Name("foo"), args=[])
  args_1 = [cst.Arg(value=cst.Name("x"))]
  args_2 = [cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))]
  res1: typing.Any = rewrite_as_infix(original, args_1, "-", ["x"])
  assert isinstance(res1, cst.UnaryOperation)
  args_bin = [cst.Arg(value=cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b")))]
  res1b: typing.Any = rewrite_as_infix(original, args_bin, "-", ["x"])
  assert isinstance(res1b, cst.UnaryOperation)
  assert len(res1b.expression.lpar) > 0  # type: ignore
  with pytest.raises(ValueError, match="expects 1 argument"):
    rewrite_as_infix(original, [], "-", ["x"])
  with pytest.raises(ValueError, match="Unsupported unary"):
    rewrite_as_infix(original, args_1, "???", ["x"])
  res2: typing.Any = rewrite_as_infix(original, args_2, "+", ["x", "y"])
  assert isinstance(res2, cst.BinaryOperation)
  with pytest.raises(ValueError, match="requires 2 arguments"):
    rewrite_as_infix(original, args_1, "+", ["x", "y"])
  with pytest.raises(ValueError, match="Unsupported binary"):
    rewrite_as_infix(original, args_2, "???", ["x", "y"])
  with pytest.raises(ValueError, match="requires 1 or 2 args"):
    rewrite_as_infix(original, args_2, "+", ["x", "y", "z"])
