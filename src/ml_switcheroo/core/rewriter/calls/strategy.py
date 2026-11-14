"""Core Transformation Strategies for Call Rewriting."""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.dispatch import evaluate_dispatch_rules
from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.calls.transformers import rewrite_as_infix, rewrite_as_inline_lambda, rewrite_as_macro
from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_permute_call,
  is_neural_on_math_backend,
)
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.exceptions import MLSwitcherooError, PluginExecutionError

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.hooks import HookContext
  from ml_switcheroo.core.rewriter.interface import RewriterHost
  from ml_switcheroo.semantics.manager import SemanticsManager


@typing.runtime_checkable
class _HookContextOwnerProtocol(typing.Protocol):
  """Protocol for rewriter contexts exposing `hook_context`.

  Attributes:
      hook_context: Active hook execution context.
  """

  hook_context: "HookContext"


@typing.runtime_checkable
class _StrategyContextOwnerProtocol(typing.Protocol):
  """Protocol for rewriters exposing `context`.

  Attributes:
      context: Active rewriter context.
  """

  context: _HookContextOwnerProtocol


@typing.runtime_checkable
class _VariantImportsHandlerProtocol(typing.Protocol):
  """Protocol for rewriters providing `_handle_variant_imports`."""

  def _handle_variant_imports(self, mapping: dict[str, RuntimeTensorOrValue]) -> None:
    """Handle variant required imports.

    Args:
        mapping: Target variant dictionary.
    """
    ...


@typing.runtime_checkable
class _ModuleAliasProtocol(typing.Protocol):
  """Protocol for rewriters providing `_is_module_alias`."""

  def _is_module_alias(self, node: cst.BaseExpression) -> bool:
    """Check if `node` is a module alias.

    Args:
        node: CST expression node.

    Returns:
        True if module alias.
    """
    ...


@typing.runtime_checkable
class _SemanticsOwnerProtocol(typing.Protocol):
  """Protocol for rewriters exposing `semantics`."""

  @property
  def semantics(self) -> "SemanticsManager":
    """Return `SemanticsManager`.

    Returns:
        `SemanticsManager` instance.
    """
    ...


@typing.runtime_checkable
class _StrictModeOwnerProtocol(typing.Protocol):
  """Protocol for rewriters exposing `strict_mode`."""

  @property
  def strict_mode(self) -> bool:
    """Return `strict_mode` flag.

    Returns:
        Boolean flag.
    """
    ...


class StrategyRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by execute_strategy."""

  @property
  def source_framework(self) -> str:
    """Return source framework key.

    Returns:
        Source framework key.
    """
    ...

  @property
  def target_framework(self) -> str:
    """Return target framework key.

    Returns:
        Target framework key.
    """
    ...

  def _report_failure(self, msg: str) -> None:
    """Report a rewrite failure.

    Args:
        msg: Failure message.
    """
    ...

  def _create_name_node(self, api_str: str) -> cst.BaseExpression:
    """Create a CST Name or Attribute node for api_str.

    Args:
        api_str: API path string.

    Returns:
        CST Name or Attribute node.
    """
    ...


def _extract_std_arg_names(details: typing.Mapping[str, RuntimeTensorOrValue]) -> list[str]:
  """Extract ordered standard parameter names from ``details['std_args']``.

  Args:
      details: Operation semantic details mapping.

  Returns:
      Ordered list of standard argument name strings.
  """
  raw_std_args = details.get("std_args", [])
  std_arg_seq = raw_std_args if isinstance(raw_std_args, list) else []
  names: list[str] = []
  for item in std_arg_seq:
    if isinstance(item, dict):
      names.append(str(item.get("name", "")))
    elif isinstance(item, (list, tuple)) and item:
      names.append(str(item[0]))
    else:
      names.append(str(item))
  return names


def _execute_infix_or_lambda(
  rewriter: "RewriterHost | StrategyRewriterProtocol",
  original: cst.Call,
  updated: cst.Call,
  mutable_mapping: dict[str, RuntimeTensorOrValue],
  details: typing.Mapping[str, RuntimeTensorOrValue],
  is_mod_alias: typing.Callable[[cst.BaseExpression], bool],
  *,
  is_infix: bool,
) -> cst.BaseExpression:
  """Execute an ``infix`` or ``inline_lambda`` call rewrite strategy.

  Args:
      rewriter: Active rewriter instance.
      original: Original source ``cst.Call`` node.
      updated: Pre-processed ``cst.Call`` node.
      mutable_mapping: Resolved variant mapping dictionary.
      details: Standard operation details dictionary.
      is_mod_alias: Module-alias predicate function.
      is_infix: ``True`` for ``infix`` transformation, ``False`` for ``inline_lambda``.

  Returns:
      Rewritten CST expression or ``updated`` on failure.
  """
  if is_infix:
    try:
      norm_args = normalize_arguments(
        original, updated, details, mutable_mapping, rewriter.source_framework, is_mod_alias
      )
      raw_op = mutable_mapping.get("operator")
      op_str = raw_op if isinstance(raw_op, str) else (str(raw_op) if raw_op is not None else "")
      return rewrite_as_infix(original, norm_args, op_str, _extract_std_arg_names(details))
    except (ValueError, IndexError) as e:
      rewriter._report_failure(f"Infix/Prefix transformation failed: {e}")
      return updated

  try:
    norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_framework, is_mod_alias)
    raw_api = mutable_mapping.get("api")
    api_str = raw_api if isinstance(raw_api, str) else (str(raw_api) if raw_api is not None else "")
    return rewrite_as_inline_lambda(api_str, norm_args)
  except (cst.ParserSyntaxError, ValueError, TypeError, KeyError, IndexError, RuntimeError, MLSwitcherooError) as e:
    rewriter._report_failure(f"Inline lambda transformation failed: {e}")
    return updated


def _execute_plugin_strategy(
  rewriter: "RewriterHost | StrategyRewriterProtocol",
  updated: cst.Call,
  mutable_mapping: dict[str, RuntimeTensorOrValue],
  hook_ctx: "HookContext | None",
) -> cst.BaseExpression:
  """Execute a ``requires_plugin`` hook strategy.

  Args:
      rewriter: Active rewriter instance.
      updated: Pre-processed ``cst.Call`` node.
      mutable_mapping: Resolved variant mapping dictionary.
      hook_ctx: Active ``HookContext`` if bound on ``rewriter``.

  Returns:
      Hook-transformed ``cst.BaseExpression`` or ``updated`` on fallback/missing hook.

  Raises:
      PluginExecutionError: If the hook exists but ``hook_ctx`` is ``None``.
  """
  raw_plugin = mutable_mapping["requires_plugin"]
  plugin_name = raw_plugin if isinstance(raw_plugin, str) else (str(raw_plugin) if raw_plugin is not None else "")
  hook = get_hook(plugin_name)
  if not hook:
    rewriter._report_failure(f"Missing required plugin: '{plugin_name}'")
    return updated
  if hook_ctx is None:
    raise PluginExecutionError(f"Plugin '{plugin_name}' requires a valid HookContext.")
  res = hook(updated, hook_ctx)
  return res if isinstance(res, cst.BaseExpression) else updated


def _report_missing_target_api(
  rewriter: "RewriterHost | StrategyRewriterProtocol",
  mutable_mapping: dict[str, RuntimeTensorOrValue],
  abstract_id: str,
) -> None:
  """Report a structured failure diagnostic when ``mutable_mapping`` defines no target ``api``.

  Args:
      rewriter: Active rewriter instance.
      mutable_mapping: Resolved variant mapping dictionary.
      abstract_id: Canonical abstract operation identifier.
  """
  sem_obj = rewriter.semantics if isinstance(rewriter, _SemanticsOwnerProtocol) else None
  try:
    origins = sem_obj._key_origins if sem_obj is not None else {}
  except AttributeError:
    origins = {}
  tier = origins.get(abstract_id) if origins else None

  if is_neural_on_math_backend(tier, rewriter.target_framework):
    msg = (
      f"Cannot map neural network abstraction '{abstract_id}' directly to pure math backend "
      f"'{rewriter.target_framework}'. Use a stateful neural framework."
    )
  else:
    raw_missing = mutable_mapping.get("missing_message")
    msg = (
      str(raw_missing)
      if isinstance(raw_missing, str) and raw_missing
      else f"No mapping available for '{abstract_id}' -> '{rewriter.target_framework}'"
    )
  rewriter._report_failure(msg)


def _execute_standard_call(
  rewriter: "RewriterHost | StrategyRewriterProtocol",
  original: cst.Call,
  updated: cst.Call,
  mutable_mapping: dict[str, RuntimeTensorOrValue],
  details: typing.Mapping[str, RuntimeTensorOrValue],
  abstract_id: str,
  is_mod_alias: typing.Callable[[cst.BaseExpression], bool],
) -> cst.BaseExpression:
  """Execute standard target API call rewriting, strict rank guards, and layout permutation.

  Args:
      rewriter: Active rewriter instance.
      original: Original source ``cst.Call`` node.
      updated: Pre-processed ``cst.Call`` node.
      mutable_mapping: Resolved variant mapping dictionary.
      details: Standard operation details dictionary.
      abstract_id: Canonical abstract operation identifier.
      is_mod_alias: Module-alias predicate function.

  Returns:
      Rewritten CST expression or ``updated`` on failure.
  """
  raw_target_api = mutable_mapping.get("api")
  target_api = str(raw_target_api) if isinstance(raw_target_api, str) and raw_target_api else None
  if not target_api:
    _report_missing_target_api(rewriter, mutable_mapping, abstract_id)
    return updated

  try:
    norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_framework, is_mod_alias)
    if isinstance(rewriter, _StrictModeOwnerProtocol) and bool(rewriter.strict_mode):
      norm_args = apply_strict_guards(rewriter, norm_args, details, mutable_mapping)

    rewritten_call = updated.with_changes(func=rewriter._create_name_node(target_api), args=norm_args)
    raw_layout_map = mutable_mapping.get("layout_map")
    if isinstance(raw_layout_map, typing.Mapping) and raw_layout_map:
      return _apply_layout_permutation(rewritten_call, mutable_mapping, details, rewriter)
    return rewritten_call
  except ValueError:
    rewriter._report_failure("Argument normalization failed")
    return updated


def execute_strategy(
  rewriter: "RewriterHost | StrategyRewriterProtocol",
  original: cst.Call,
  updated: cst.Call,
  mapping: typing.Mapping[str, RuntimeTensorOrValue],
  details: typing.Mapping[str, RuntimeTensorOrValue],
  abstract_id: str,
) -> cst.BaseExpression:
  """Apply the appropriate transformation strategy for rewriting a function call.

  This function determines and executes the correct transformation mechanism
  (e.g., infix operators, inline lambdas, macro expansions, plugins, or standard
  API calls) based on the provided mapping rules and API details. It also handles
  dispatch rules, argument normalization, strict mode guards, and layout permutations.

  Args:
      rewriter: The active rewriter instance providing context, semantics, and target framework details.
      original: The original `cst.Call` node from the source tree before any modifications.
      updated: The currently updated `cst.Call` node, which may have already undergone pre-processing.
      mapping: The target framework-specific mapping dictionary detailing how to transform the call.
      details: The general API details dictionary for the abstract operation being mapped.
      abstract_id: The unique identifier for the abstract operation (e.g., 'math.add').

  Returns:
      cst.BaseExpression: The transformed CST node representing the rewritten call or expression.

  Raises:
      PluginExecutionError: If a registered plugin hook is invoked without an active ``HookContext``.
  """
  mutable_mapping: dict[str, RuntimeTensorOrValue] = dict(mapping)
  hook_ctx: HookContext | None = None
  if isinstance(rewriter, _StrategyContextOwnerProtocol) and isinstance(rewriter.context, _HookContextOwnerProtocol):
    hook_ctx = rewriter.context.hook_context
    hook_ctx.current_op_id = abstract_id

  if isinstance(rewriter, _VariantImportsHandlerProtocol):
    rewriter._handle_variant_imports(mutable_mapping)

  raw_rules = mutable_mapping.get("dispatch_rules")
  if isinstance(raw_rules, list) and raw_rules:
    dispatched_api = evaluate_dispatch_rules(rewriter, original, raw_rules, details)
    if dispatched_api:
      mutable_mapping = mutable_mapping.copy()
      mutable_mapping["api"] = dispatched_api

  trans_type = mutable_mapping.get("transformation_type")
  is_mod_alias: typing.Callable[[cst.BaseExpression], bool] = (
    rewriter._is_module_alias if isinstance(rewriter, _ModuleAliasProtocol) else (lambda _n: False)
  )

  if trans_type in ("infix", "inline_lambda"):
    return _execute_infix_or_lambda(
      rewriter, original, updated, mutable_mapping, details, is_mod_alias, is_infix=(trans_type == "infix")
    )
  if "requires_plugin" in mutable_mapping:
    return _execute_plugin_strategy(rewriter, updated, mutable_mapping, hook_ctx)

  raw_macro = mutable_mapping.get("macro_template")
  if isinstance(raw_macro, str) and bool(raw_macro):
    try:
      norm_args = normalize_arguments(
        original, updated, details, mutable_mapping, rewriter.source_framework, is_mod_alias
      )
      return rewrite_as_macro(raw_macro, norm_args, _extract_std_arg_names(details))
    except (cst.ParserSyntaxError, ValueError, TypeError, KeyError, IndexError, RuntimeError, MLSwitcherooError) as e:
      rewriter._report_failure(f"Macro expansion failed: {e}")
      return updated

  return _execute_standard_call(rewriter, original, updated, mutable_mapping, details, abstract_id, is_mod_alias)


def _apply_layout_permutation(
  node: cst.Call,
  mapping: typing.Mapping[str, RuntimeTensorOrValue],
  details: typing.Mapping[str, RuntimeTensorOrValue],
  rewriter: "RewriterHost | StrategyRewriterProtocol",
) -> cst.BaseExpression:
  """Apply layout permutation to the arguments or return value of a call.

  This function modifies the arguments of a function call, or wraps the entire call,
  with permutation logic (e.g., transposing dimensions) based on the `layout_map`
  provided in the transformation mapping. This is commonly used when transforming
  between frameworks with different data layout conventions (like NCHW vs. NHWC).

  Args:
      node: The call node whose arguments or return value need layout permutation.
      mapping: The transformation mapping dictionary containing the 'layout_map' rules.
      details: The general API details dictionary specifying standard argument names.
      rewriter: The active rewriter instance providing semantics and target framework details.

  Returns:
      cst.BaseExpression: The modified call node with permuted arguments or a permutation wrapped around the return value.

  """
  raw_layout_map = mapping.get("layout_map")
  layout_map: dict[str, str] = (
    {str(k): str(v) for k, v in raw_layout_map.items()} if isinstance(raw_layout_map, dict) else {}
  )
  raw_std_args = details.get("std_args", [])
  std_args_raw = raw_std_args if isinstance(raw_std_args, list) else []
  modified_args = list(node.args)
  if not isinstance(rewriter, _SemanticsOwnerProtocol):
    return node
  sem = rewriter.semantics

  for idx, item in enumerate(std_args_raw):
    arg_name = (
      str(item.get("name", ""))
      if isinstance(item, dict)
      else (str(item[0]) if isinstance(item, (list, tuple)) and item else str(item))
    )
    if arg_name and arg_name in layout_map:
      rule = layout_map[arg_name]
      if "->" in rule:
        src_l, tgt_l = rule.split("->")
        perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
        if perm_indices and idx < len(modified_args):
          original_arg = modified_args[idx]
          wrapped_val = inject_permute_call(
            original_arg.value,
            perm_indices,
            sem,
            rewriter.target_framework,
          )
          modified_args[idx] = original_arg.with_changes(value=wrapped_val)

  res_node: cst.BaseExpression = node.with_changes(args=modified_args)

  if "return" in layout_map:
    rule = layout_map["return"]
    if "->" in rule:
      src_l, tgt_l = rule.split("->")
      perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
      if perm_indices:
        res_node = inject_permute_call(res_node, perm_indices, sem, rewriter.target_framework)

  return res_node
