"""Post-processing Phase for Call Rewriting.

Handles output adaptation and state threading.
Updated to prevent imports from legacy modules and removed deprecated output adapter hooks.
"""

import logging
import typing

import libcst as cst
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.rewriter.calls.transformers import apply_index_select
from ml_switcheroo.core.rewriter.calls.utils import inject_kwarg, strip_kwarg
from ml_switcheroo.enums import RuntimeTensorOrValue

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost
  from ml_switcheroo.semantics.schema import StructuralTraits


@typing.runtime_checkable
class _FailureReporterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_report_failure`."""

  def _report_failure(self, msg: str) -> None:
    """Report a failure message.

    Args:
        msg: Failure message.
    """


@typing.runtime_checkable
class _TargetTraitsGetterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_get_target_traits`."""

  def _get_target_traits(self) -> "StructuralTraits":
    """Return target framework `StructuralTraits`.

    Returns:
        Resolved `StructuralTraits`.
    """


@typing.runtime_checkable
class _PostSigFrameProtocol(typing.Protocol):
  """Protocol for signature stack entries tracked during post-processing.

  Attributes:
      is_init: True if the enclosing function is ``__init__``.
      is_module_method: True if the enclosing function belongs to a module class.
  """

  is_init: bool
  is_module_method: bool


@typing.runtime_checkable
class _PostContextProtocol(typing.Protocol):
  """Protocol for rewriter contexts exposing `signature_stack`.

  Attributes:
      signature_stack: Stack of active signature frames.
  """

  signature_stack: list[_PostSigFrameProtocol]


@typing.runtime_checkable
class _PostOriginsProtocol(typing.Protocol):
  """Protocol for semantics objects exposing `_key_origins`.

  Attributes:
      _key_origins: Mapping from abstract operation ID to semantic tier string.
  """

  _key_origins: dict[str, str]


@typing.runtime_checkable
class _PostMagicArgsProtocol(typing.Protocol):
  """Protocol for semantics objects exposing `known_magic_args`.

  Attributes:
      known_magic_args: Collection of magic argument names across frameworks.
  """

  known_magic_args: list[str] | set[str] | tuple[str, ...]


class _BareContextProtocol(typing.Protocol):
  """Fallback structural protocol for partial context stubs lacking `signature_stack`."""


@typing.runtime_checkable
class _PostStateRewriterProtocol(typing.Protocol):
  """Protocol for rewriters exposing `context` and `semantics` during constructor threading.

  Attributes:
      context: Rewriter context candidate.
      semantics: Semantics manager containing tier origin metadata.
  """

  context: _PostContextProtocol | _BareContextProtocol | None
  semantics: _PostOriginsProtocol


class PostRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by handle_post_processing."""

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Create a dotted name expression from string.

    Args:
        name_str: Dotted name string.

    Returns:
        Created CST expression.
    """


def _apply_constructor_state_threading(
  rewriter: "_PostStateRewriterProtocol",
  traits_owner: "_TargetTraitsGetterProtocol",
  result_node: cst.Call,
  abstract_id: str,
) -> cst.Call:
  """Inject and strip constructor magic arguments (e.g. ``rngs``) when inside a module ``__init__``.

  Args:
      rewriter: Rewriter exposing ``semantics``.
      traits_owner: Rewriter exposing ``_get_target_traits``.
      result_node: Rewritten constructor ``cst.Call`` node.
      abstract_id: Canonical operation identifier.

  Returns:
      Updated ``cst.Call`` node with magic arguments injected/stripped as configured.
  """
  sem_obj = rewriter.semantics
  tier = sem_obj._key_origins.get(abstract_id)
  traits = traits_owner._get_target_traits()
  is_neural = tier in (SemanticTier.NEURAL.value, "neural_ops")
  known_magic = sem_obj.known_magic_args if isinstance(sem_obj, _PostMagicArgsProtocol) else None

  magic = set(traits.strip_magic_args)
  if traits.auto_strip_magic_args and isinstance(known_magic, (set, list, tuple)):
    magic.update(known_magic)

  force = any(arg.keyword and arg.keyword.value in magic for arg in result_node.args)
  if not (is_neural or force):
    return result_node

  updated_call = result_node
  for arg_name, _ in traits.inject_magic_args:
    updated_call = inject_kwarg(updated_call, arg_name, arg_name)

  args_to_strip = set(traits.strip_magic_args)
  if traits.auto_strip_magic_args and isinstance(known_magic, (set, list, tuple)):
    args_to_strip.update(known_magic)
    args_to_strip -= {a[0] for a in traits.inject_magic_args}

  for arg_name in args_to_strip:
    updated_call = strip_kwarg(updated_call, arg_name)
  return updated_call


def handle_post_processing(
  rewriter: "RewriterHost | PostRewriterProtocol",
  node: cst.BaseExpression,
  mapping: typing.Mapping[str, RuntimeTensorOrValue],
  abstract_id: str,
) -> cst.BaseExpression:
  """Apply post-rewrite modifications to the result node, such as type casting or state threading.

  Args:
      rewriter: The CST rewriter instance containing context.
      node: The node being processed.
      mapping: The configuration mapping for the rewrite.
      abstract_id: The abstract ID of the operation.

  Returns:
      cst.BaseExpression: The modified result node.
  """
  result_node = node

  raw_select = mapping.get("output_select_index")
  if raw_select is not None:
    try:
      result_node = apply_index_select(result_node, int(str(raw_select)))
    except (ValueError, TypeError, IndexError, RuntimeError) as e:
      if isinstance(rewriter, _FailureReporterProtocol):
        rewriter._report_failure(f"Output indexing failed: {e}")
      return result_node

  raw_cast = mapping.get("output_cast")
  if isinstance(raw_cast, str) and raw_cast:
    try:
      type_node = rewriter._create_dotted_name(raw_cast)
      result_node = cst.Call(
        func=cst.Attribute(value=result_node, attr=cst.Name("astype")),
        args=[cst.Arg(value=type_node)],
      )
    except (cst.CSTValidationError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
      logging.getLogger(__name__).debug("Output cast skipped for '%s': %s", raw_cast, exc)

  if (
    isinstance(result_node, cst.Call)
    and isinstance(rewriter, _PostStateRewriterProtocol)
    and isinstance(rewriter, _TargetTraitsGetterProtocol)
    and isinstance(rewriter.context, _PostContextProtocol)
    and len(rewriter.context.signature_stack) > 0
    and rewriter.context.signature_stack[-1].is_init
    and rewriter.context.signature_stack[-1].is_module_method
  ):
    return _apply_constructor_state_threading(rewriter, rewriter, result_node, abstract_id)

  return result_node
