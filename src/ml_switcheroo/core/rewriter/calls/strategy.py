"""Core Transformation Strategies for Call Rewriting."""

import typing

import libcst as cst
from pydantic import JsonValue

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

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


class StrategyRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by execute_strategy."""

  @property
  def source_fw(self) -> str:
    """Return source framework key.

    Returns:
        Source framework key.
    """
    ...

  @property
  def target_fw(self) -> str:
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


def execute_strategy(
  rewriter: typing.Union["RewriterHost", StrategyRewriterProtocol],
  original: cst.Call,
  updated: cst.Call,
  mapping: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]],
  details: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]],
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

  """
  mutable_mapping: typing.Dict[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]] = dict(mapping)
  ctx_obj = getattr(rewriter, "context", None)
  if ctx_obj is not None and hasattr(ctx_obj, "hook_context"):
    ctx_obj.hook_context.current_op_id = abstract_id

  # Handle imports from ApiPass helper
  if hasattr(rewriter, "_handle_variant_imports"):
    getattr(rewriter, "_handle_variant_imports")(mutable_mapping)

  # 1. Dispatch Rules
  raw_rules = mutable_mapping.get("dispatch_rules")
  if isinstance(raw_rules, list) and raw_rules:
    dispatched_api = evaluate_dispatch_rules(rewriter, original, raw_rules, details)
    if dispatched_api:
      mutable_mapping = mutable_mapping.copy()
      mutable_mapping["api"] = dispatched_api

  trans_type = mutable_mapping.get("transformation_type")
  is_mod_alias: typing.Callable[[cst.BaseExpression], bool] = getattr(rewriter, "_is_module_alias", lambda _n: False)

  # 2. Infix
  if trans_type == "infix":
    try:
      norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_fw, is_mod_alias)
      op_str = str(mutable_mapping.get("operator") or "")
      raw_std_args = details.get("std_args", [])
      std_args_list: typing.List[str] = (
        [
          str(x.get("name", "")) if isinstance(x, dict) else (str(x[0]) if isinstance(x, (list, tuple)) else str(x))
          for x in raw_std_args
        ]
        if isinstance(raw_std_args, list)
        else []
      )
      return rewrite_as_infix(
        original,
        norm_args,
        op_str,
        std_args_list,
      )
    except (ValueError, IndexError) as e:
      rewriter._report_failure(f"Infix/Prefix transformation failed: {e}")
      return updated

  # 3. Inline Lambda
  elif trans_type == "inline_lambda":
    try:
      norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_fw, is_mod_alias)
      return rewrite_as_inline_lambda(str(mutable_mapping.get("api") or ""), norm_args)
    except Exception as e:
      rewriter._report_failure(f"Inline lambda transformation failed: {e}")
      return updated

  # 4. Plugin
  elif "requires_plugin" in mutable_mapping:
    plugin_name = str(mutable_mapping["requires_plugin"] or "")
    hook = get_hook(plugin_name)
    if hook:
      res = hook(updated, getattr(ctx_obj, "hook_context"))
      if isinstance(res, cst.BaseExpression):
        return res
      return updated
    else:
      rewriter._report_failure(f"Missing required plugin: '{plugin_name}'")
      return updated

  # 5. Macro
  elif mutable_mapping.get("macro_template"):
    try:
      norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_fw, is_mod_alias)
      std_arg_names: typing.List[str] = []
      for item in getattr(details, "get")("std_args", []):
        if isinstance(item, (list, tuple)) and item:
          std_arg_names.append(str(item[0]))
        elif isinstance(item, dict):
          std_arg_names.append(str(item.get("name", "")))
        else:
          std_arg_names.append(str(item))
      return rewrite_as_macro(str(mutable_mapping["macro_template"]), norm_args, std_arg_names)
    except Exception as e:
      rewriter._report_failure(f"Macro expansion failed: {e}")
      return updated

  # 6. Standard
  else:
    raw_target_api = mutable_mapping.get("api")
    target_api = str(raw_target_api) if isinstance(raw_target_api, str) and raw_target_api else None
    if not target_api:
      sem_obj = getattr(rewriter, "semantics", None)
      origins = getattr(sem_obj, "_key_origins", {}) if sem_obj is not None else {}
      tier = origins.get(abstract_id)

      if is_neural_on_math_backend(tier, rewriter.target_fw):
        msg = f"Cannot map neural network abstraction '{abstract_id}' directly to pure math backend '{rewriter.target_fw}'. Use a framework like Flax or Keras."
      else:
        raw_missing = mutable_mapping.get("missing_message")
        msg = (
          str(raw_missing)
          if isinstance(raw_missing, str) and raw_missing
          else f"No mapping available for '{abstract_id}' -> '{rewriter.target_fw}'"
        )
      rewriter._report_failure(msg)
      return updated

    try:
      norm_args = normalize_arguments(original, updated, details, mutable_mapping, rewriter.source_fw, is_mod_alias)

      # Apply Strict Guards (Rank Checking)
      if getattr(rewriter, "strict_mode", False):
        norm_args = apply_strict_guards(rewriter, norm_args, details, mutable_mapping)  # type: ignore[arg-type]

      new_func = rewriter._create_name_node(target_api)
      res_node: cst.BaseExpression = updated.with_changes(func=new_func, args=norm_args)

      # Layout Permutation Logic
      if "layout_map" in mutable_mapping and mutable_mapping["layout_map"]:
        res_node = _apply_layout_permutation(
          updated.with_changes(func=new_func, args=norm_args), mutable_mapping, details, rewriter
        )

      return res_node

    except ValueError:
      rewriter._report_failure("Argument normalization failed")
      return updated


def _apply_layout_permutation(
  node: cst.Call,
  mapping: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]],
  details: typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]],
  rewriter: typing.Union["RewriterHost", StrategyRewriterProtocol],
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
  layout_map: typing.Dict[str, str] = (
    {str(k): str(v) for k, v in raw_layout_map.items()} if isinstance(raw_layout_map, dict) else {}
  )
  raw_std_args = details.get("std_args", [])
  std_args_raw = raw_std_args if isinstance(raw_std_args, list) else []
  idx = 0
  modified_args = list(node.args)

  for item in std_args_raw:
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
            getattr(rewriter, "semantics"),
            rewriter.target_fw,
          )
          modified_args[idx] = original_arg.with_changes(value=wrapped_val)
    idx += 1

  res_node: cst.BaseExpression = node.with_changes(args=modified_args)

  if "return" in layout_map:
    rule = layout_map["return"]
    if "->" in rule:
      src_l, tgt_l = rule.split("->")
      perm_indices = compute_permutation(src_l.strip(), tgt_l.strip())
      if perm_indices:
        res_node = inject_permute_call(res_node, perm_indices, getattr(rewriter, "semantics"), rewriter.target_fw)

  return res_node
