"""Initialization construction strategies."""

import typing

import libcst as cst
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.schema import StructuralTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


def _resolve_traits(framework: str, traits: StructuralTraits | object | None = None) -> StructuralTraits:
  """Retrieve StructuralTraits for ``framework``, falling back to an empty instance.

  Args:
      framework: Target framework identifier string.
      traits: Optional pre-resolved StructuralTraits instance.

  Returns:
      Resolved StructuralTraits configuration.
  """
  if isinstance(traits, StructuralTraits):
    return traits
  adapter = get_adapter(framework)
  base = adapter.structural_traits.model_copy() if adapter is not None else StructuralTraits()
  if traits is not None:
    for field_name in StructuralTraits.model_fields:
      if hasattr(traits, field_name):
        setattr(base, field_name, getattr(traits, field_name))
  return base


def format_args_from_metadata(metadata: typing.Mapping[str, AttributeValue] | None) -> str:
  """Format a LogicalNode metadata dictionary into a comma-separated Python argument string.

  Args:
      metadata: Dictionary of positional (`arg_0`, `arg_1`) and keyword argument values.

  Returns:
      Comma-separated argument string suitable for a Python call expression.
  """
  if not metadata:
    return ""
  args_list = []
  for key in sorted(metadata.keys()):
    val = str(metadata[key])
    if key.startswith("arg_"):
      args_list.append(val)
    else:
      args_list.append(f"{key}={val}")
  return ", ".join(args_list)


def is_stateful_layer(node: LogicalNode) -> bool:
  """Determine whether a LogicalNode represents a stateful neural network layer.

  Args:
      node: The LogicalNode to inspect.

  Returns:
      True if the node represents a stateful layer requiring initialization, False otherwise.
  """
  op_type: str = str(node.op_type)
  if op_type in ["Input", "Output"]:
    return False
  if "." in op_type and not op_type.startswith("nn."):
    return False
  return True


def generate_layer_init(
  node: LogicalNode,
  framework: str,
  semantics: "SemanticsManager | None",
  traits: StructuralTraits | None = None,
) -> cst.SimpleStatementLine:
  """Generate the LibCST initialization assignment statement for a stateful layer node.

  Args:
      node: The LogicalNode representing the layer to initialize.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for looking up variant constructors.
      traits: Optional StructuralTraits for the target framework.

  Returns:
      LibCST SimpleStatementLine representing the layer attribute initialization.
  """
  resolved_traits = _resolve_traits(framework, traits)
  kind: str = str(node.op_type)
  if semantics and hasattr(semantics, "resolve_variant"):
    definition = semantics.resolve_variant(kind, framework)
    if definition and "api" in definition:
      resolved_api = str(definition["api"])
      skip_token = resolved_traits.functional_namespace_skip
      if not (skip_token and skip_token in resolved_api):
        kind = resolved_api

  alias_strip = resolved_traits.module_alias_strip
  if alias_strip and kind.startswith(alias_strip[0]):
    kind = alias_strip[1] + kind[len(alias_strip[0]) :]

  if "." not in kind:
    rewritten_leaf = resolved_traits.layer_name_rewrites.get(kind, kind)
    prefix = resolved_traits.layer_namespace_prefix
    if prefix:
      kind = f"{prefix}.{rewritten_leaf}"
    else:
      kind = rewritten_leaf

  args_str = format_args_from_metadata(node.attributes)
  if resolved_traits.requires_rng_in_init:
    if "rngs" not in args_str:
      suffix = ", rngs=rngs" if args_str else "rngs=rngs"
      args_str += suffix

  reg_method = resolved_traits.child_registration_method
  if reg_method:
    code = f"self.{reg_method}('{node.id}', {kind}({args_str}))"
  else:
    code = f"self.{node.id} = {kind}({args_str})"
  stmt = cst.parse_statement(code)
  assert isinstance(stmt, cst.SimpleStatementLine)
  return stmt


def build_init(
  nodes: list[LogicalNode],
  framework: str,
  semantics: "SemanticsManager | None",
  traits: "StructuralTraits",
) -> cst.FunctionDef:
  """Build the `__init__` or `setup` CST FunctionDef node for a synthesized model class.

  Args:
      nodes: Ordered list of LogicalNodes in the model graph.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for layer constructor resolution.
      traits: StructuralTraits for the target framework.

  Returns:
      Constructed LibCST FunctionDef node for model initialization.
  """
  resolved_traits = _resolve_traits(framework, traits)
  stmts: list[cst.BaseStatement] = []
  if resolved_traits.requires_super_init:
    stmts.append(cst.parse_statement("super().__init__()"))

  for node in nodes:
    if is_stateful_layer(node):
      stmts.append(generate_layer_init(node, framework, semantics, resolved_traits))

  if not stmts:
    stmts.append(cst.SimpleStatementLine(body=[cst.Pass()]))

  params = [cst.Param(name=cst.Name("self"))]
  if resolved_traits.requires_rng_in_init:
    params.append(
      cst.Param(
        name=cst.Name("rngs"),
        annotation=cst.Annotation(cst.parse_expression("nnx.Rngs")),
      )
    )

  init_name = resolved_traits.init_method_name if resolved_traits.init_method_name else "__init__"
  return cst.FunctionDef(
    name=cst.Name(init_name),
    params=cst.Parameters(params=params),
    body=cst.IndentedBlock(body=stmts),
  )
