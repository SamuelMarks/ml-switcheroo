"""Logic for Conditional API Dispatch."""

from ml_switcheroo.enums import RuntimeTensorOrValue
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.dsl import Rule
from ml_switcheroo.enums import LogicOp

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


class DispatchRuleProtocol(typing.Protocol):
  """Protocol for conditional dispatch rules."""

  @property
  def op(self) -> LogicOp | str | None:
    """Return logical operation.

    Returns:
        Logical operation enum or string.
    """
    ...

  @property
  def is_val(self) -> RuntimeTensorOrValue:
    """Return comparison target value.

    Returns:
        Comparison target value.
    """
    ...


RuleInput: typing.TypeAlias = DispatchRuleProtocol | Rule | typing.Mapping[str, RuntimeTensorOrValue] | JsonValue


class DispatchRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by evaluate_dispatch_rules."""


def evaluate_dispatch_rules(
  rewriter: "RewriterHost | DispatchRewriterProtocol",
  node: cst.Call,
  rules: typing.Sequence[RuleInput],
  details: typing.Mapping[str, RuntimeTensorOrValue],
) -> str | None:
  """Evaluate conditional dispatch rules against the current call arguments.

  Args:
      rewriter: The calling transformer object.
      node: The Call node from libcst being evaluated.
      rules: A list of conditional rules to evaluate.
      details: A dictionary containing details for rule evaluation, including variants.

  Returns:
      The string identifier of the API to dispatch to if a rule matches, otherwise None.

  """
  raw_variants = details.get("variants")
  src_fw = str(getattr(rewriter, "source_framework", ""))
  source_variant = raw_variants.get(src_fw, {}) if isinstance(raw_variants, dict) else {}
  raw_src_args = source_variant.get("args", {}) if isinstance(source_variant, dict) else {}
  source_arg_map: dict[str, str] = (
    {str(k): str(v) for k, v in raw_src_args.items() if v is not None} if isinstance(raw_src_args, dict) else {}
  )

  raw_std_args = details.get("std_args", [])
  std_args_raw = raw_std_args if isinstance(raw_std_args, list) else []
  std_args_order: list[str] = []
  for item in std_args_raw:
    if isinstance(item, (list, tuple)) and item:
      std_args_order.append(str(item[0]))
    elif isinstance(item, dict):
      name = item.get("name")
      if isinstance(name, str) and name:
        std_args_order.append(name)
    else:
      std_args_order.append(str(item))

  for rule in rules:
    if_arg = str(getattr(rule, "if_arg", ""))
    use_api = str(getattr(rule, "use_api", ""))
    src_arg_name = source_arg_map.get(if_arg, if_arg)
    arg_node = _extract_argument_node(rewriter, node, src_arg_name, if_arg, std_args_order)

    if arg_node is None:
      continue

    if _check_rule_condition(arg_node, rule):
      return use_api

  return None


def _extract_argument_node(
  rewriter: "RewriterHost | DispatchRewriterProtocol",
  node: cst.Call,
  src_name: str,
  std_name: str,
  std_order: list[str],
) -> cst.BaseExpression | None:
  """Extract the argument node for a given parameter name.

  Args:
      rewriter: The calling transformer object.
      node: The Call node from libcst.
      src_name: The source argument name to look for as a keyword.
      std_name: The standard argument name to find by positional order.
      std_order: The list of standard arguments defining their positional order.

  Returns:
      The corresponding CSTNode if found, otherwise None.

  """
  for arg in node.args:
    if arg.keyword and arg.keyword.value == src_name:
      return arg.value

  try:
    idx = std_order.index(std_name)
    is_module = False
    if hasattr(rewriter, "_is_module_alias"):
      is_module = rewriter._is_module_alias(node.func.value) if isinstance(node.func, cst.Attribute) else False

    is_method = isinstance(node.func, cst.Attribute) and not is_module
    call_idx = idx
    if is_method and std_order and std_order[0] == "x":
      call_idx = idx - 1

    if call_idx >= 0 and call_idx < len(node.args):
      arg = node.args[call_idx]
      if not arg.keyword:
        return arg.value
  except ValueError:
    pass

  return None


def _node_to_literal(node: cst.CSTNode) -> str | int | float | bool | None:
  """Convert a CST node into a basic Python literal type if possible.

  Args:
      node: The CSTNode to attempt converting to a literal.

  Returns:
      The parsed Python literal value (int, float, str, bool, or None) if successfully converted, otherwise None.

  """
  if isinstance(node, cst.Integer):
    try:
      return int(node.value)
    except ValueError:
      return None
  if isinstance(node, cst.Float):
    try:
      return float(node.value)
    except ValueError:
      return None
  if isinstance(node, cst.SimpleString):
    return node.value.strip("'").strip('"')
  if isinstance(node, cst.Name):
    name_map = {"True": True, "False": False, "None": None}
    return name_map.get(node.value, None)
  return None


def _check_rule_condition(node: cst.CSTNode, rule: RuleInput) -> bool:
  """Check if a given CSTNode meets the condition specified by a rule.

  Args:
      node: The CSTNode extracted from a call argument.
      rule: The rule object containing the logical operation and target value.

  Returns:
      True if the condition is met, False otherwise.

  """
  op = getattr(rule, "op", None)
  target = getattr(rule, "is_val", None)

  if op == LogicOp.IS_TYPE:
    expected_type = str(target).lower()
    if isinstance(node, cst.Integer):
      return expected_type == "int"
    if isinstance(node, cst.Float):
      return expected_type == "float"
    if isinstance(node, cst.SimpleString):
      return expected_type == "str"
    if isinstance(node, (cst.List, cst.Tuple)):
      return expected_type in ["list", "tuple", "sequence"]
    if isinstance(node, cst.Dict):
      return expected_type == "dict"
    if isinstance(node, cst.Name) and node.value in ["True", "False"]:
      return expected_type == "bool"
    return False

  val = _node_to_literal(node)
  if val is None:
    return False

  if op == LogicOp.EQ:
    return bool(val == target)
  if op == LogicOp.NEQ:
    return bool(val != target)
  if op == LogicOp.GT:
    return (
      bool(val > target)
      if isinstance(val, (int, float))
      and isinstance(target, (int, float))
      and not isinstance(val, bool)
      and not isinstance(target, bool)
      else False
    )
  if op == LogicOp.LT:
    return (
      bool(val < target)
      if isinstance(val, (int, float))
      and isinstance(target, (int, float))
      and not isinstance(val, bool)
      and not isinstance(target, bool)
      else False
    )
  if op == LogicOp.GTE:
    return (
      bool(val >= target)
      if isinstance(val, (int, float))
      and isinstance(target, (int, float))
      and not isinstance(val, bool)
      and not isinstance(target, bool)
      else False
    )
  if op == LogicOp.LTE:
    return (
      bool(val <= target)
      if isinstance(val, (int, float))
      and isinstance(target, (int, float))
      and not isinstance(val, bool)
      and not isinstance(target, bool)
      else False
    )
  if op == LogicOp.IN:
    return bool(val in target) if isinstance(target, (list, tuple, set, str, dict)) else False
  if op == LogicOp.NOT_IN:
    return bool(val not in target) if isinstance(target, (list, tuple, set, str, dict)) else False

  return False
