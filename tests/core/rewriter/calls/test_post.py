"""Unit tests for call post-processing (`ml_switcheroo.core.rewriter.calls.post`)."""

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import StructuralTraits


def _parse_call(code: str) -> cst.Call:
  """Parse a Python call expression string into a `cst.Call`.

  Args:
      code: Python call expression string.

  Returns:
      Parsed `cst.Call` node.
  """
  return cst.ensure_type(cst.parse_expression(code), cst.Call)


def _expr_code(node: cst.CSTNode) -> str:
  """Serialize a rewritten CST expression to stripped Python source code.

  Args:
      node: Rewritten CST expression node.

  Returns:
      Formatted Python expression string.
  """
  assert isinstance(node, cst.BaseExpression)
  return cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code.strip()


class _PostContextStub:
  """Typed context stub carrying `signature_stack`."""

  def __init__(self, *, is_init: bool = False, is_module_method: bool = False) -> None:
    """Initialize optional `SignatureContext` frame.

    Args:
        is_init: Whether currently inside `__init__`.
        is_module_method: Whether currently inside a module method.
    """
    self.signature_stack: list[SignatureContext] = []
    if is_init or is_module_method:
      sig = SignatureContext()
      sig.is_init = is_init
      sig.is_module_method = is_module_method
      self.signature_stack.append(sig)


class _PostSemanticsStub:
  """Typed semantics stub exposing `_key_origins` and `known_magic_args`."""

  def __init__(self, origins: dict[str, str] | None = None, magic: set[str] | None = None) -> None:
    """Initialize tier origins and magic argument names.

    Args:
        origins: Mapping from abstract op ID to semantic tier value.
        magic: Known framework magic keyword argument names.
    """
    self._key_origins: dict[str, str] = origins or {}
    self.known_magic_args: set[str] = magic or set()


class _PostRewriterStub:
  """Typed rewriter host stub implementing post-processing hooks."""

  def __init__(
    self,
    context: _PostContextStub | None = None,
    semantics: _PostSemanticsStub | None = None,
    traits: StructuralTraits | None = None,
  ) -> None:
    """Initialize post-processing rewriter host.

    Args:
        context: Optional `_PostContextStub`.
        semantics: Optional `_PostSemanticsStub`.
        traits: Optional target `StructuralTraits`.
    """
    self.context = context or _PostContextStub()
    self.semantics = semantics or _PostSemanticsStub()
    self.traits = traits or StructuralTraits()
    self.failures: list[str] = []

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Construct a CST expression for `name`, raising `ValueError` on `'invalid'`.

    Args:
        name: Dotted target type or attribute string.

    Returns:
        Parsed CST expression node.

    Raises:
        ValueError: When `name == 'invalid'`.
    """
    if name == "invalid":
      raise ValueError("invalid cast target")
    return cst.parse_expression(name)

  def _report_failure(self, msg: str) -> None:
    """Record reported post-processing failure message.

    Args:
        msg: Diagnostic message string.
    """
    self.failures.append(msg)

  def _get_target_traits(self) -> StructuralTraits:
    """Return configured target `StructuralTraits`.

    Returns:
        Target `StructuralTraits` instance.
    """
    return self.traits


def test_output_select_index_and_output_cast() -> None:
  """Verify `output_select_index` and `output_cast` transformations and graceful error recovery."""
  rewriter = _PostRewriterStub()
  call = _parse_call("func(x)")

  assert handle_post_processing(rewriter, call, {}, "op") is call
  assert handle_post_processing(rewriter, call, {"output_select_index": None}, "op") is call

  indexed = handle_post_processing(rewriter, call, {"output_select_index": 1}, "op")
  assert _expr_code(indexed) == "func(x)[1]"

  casted = handle_post_processing(rewriter, call, {"output_cast": "jnp.float32"}, "op")
  assert _expr_code(casted) == "func(x).astype(jnp.float32)"

  chained = handle_post_processing(rewriter, call, {"output_select_index": 0, "output_cast": "float32"}, "op")
  assert _expr_code(chained) == "func(x)[0].astype(float32)"

  # Invalid index reports failure when `_report_failure` exists, and returns unchanged when absent
  bad_idx: dict[str, JsonValue] = {"output_select_index": "not_an_int"}
  assert handle_post_processing(rewriter, call, bad_idx, "op") is call
  assert len(rewriter.failures) == 1

  class _BareRewriter:
    """Host satisfying `PostRewriterProtocol` while lacking `_report_failure` and `context`."""

    def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
      """Construct `cst.Name` from `name_str`.

      Args:
          name_str: Identifier string.

      Returns:
          `cst.Name` node.
      """
      return cst.Name(name_str)

  x_node = cst.Name("x")
  assert handle_post_processing(_BareRewriter(), x_node, bad_idx, "op") is x_node

  # Invalid `output_cast` swallows `ValueError` from `_create_dotted_name` and preserves node
  assert handle_post_processing(rewriter, call, {"output_cast": "invalid"}, "op") is call


def test_state_threading_magic_arg_injection_and_stripping() -> None:
  """Verify magic argument injection/stripping across neural ops, forced magic args, and `auto_strip_magic_args`."""
  traits = StructuralTraits(
    inject_magic_args=[("rngs", "flax.nnx.Rngs(0)")],
    strip_magic_args=["training"],
    auto_strip_magic_args=False,
  )
  ctx = _PostContextStub(is_init=True, is_module_method=True)
  sem = _PostSemanticsStub(
    origins={"neural_id": SemanticTier.NEURAL.value, "math_id": "math"},
    magic={"rngs", "extra_magic"},
  )
  rewriter = _PostRewriterStub(context=ctx, semantics=sem, traits=traits)

  # 1. Neural op inside `__init__`: strips `training` and injects `rngs`
  neural_res = handle_post_processing(rewriter, _parse_call("Linear(10, training=True)"), {}, "neural_id")
  assert _expr_code(neural_res) == "Linear(10, rngs=rngs)"

  # 2. Non-neural op WITH a known magic arg (`rngs`) forces injection/stripping
  forced_res = handle_post_processing(rewriter, _parse_call("helper(x, training=False, rngs=r)"), {}, "math_id")
  assert _expr_code(forced_res) == "helper(x, rngs=r)"

  # 3. Non-neural op WITHOUT magic args is untouched
  untouched = _parse_call("helper(x, axis=1)")
  assert handle_post_processing(rewriter, untouched, {}, "math_id") is untouched

  # 4. Non-Call result on a neural op returns unchanged
  name_node = cst.Name("x")
  assert handle_post_processing(rewriter, name_node, {}, "neural_id") is name_node

  # 5. `auto_strip_magic_args=True` strips `known_magic_args` excluding injected target keys
  auto_traits = StructuralTraits(
    inject_magic_args=[("is_training", "True")],
    strip_magic_args=["training"],
    auto_strip_magic_args=True,
  )
  auto_rewriter = _PostRewriterStub(context=ctx, semantics=sem, traits=auto_traits)
  auto_res = handle_post_processing(
    auto_rewriter,
    _parse_call("Block(x, training=True, extra_magic=1)"),
    {},
    "neural_id",
  )
  assert _expr_code(auto_res) == "Block(x, is_training=is_training)"
