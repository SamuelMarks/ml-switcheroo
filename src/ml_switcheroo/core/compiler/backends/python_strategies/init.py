"""Initialization construction strategies."""

import inspect
import typing

import libcst as cst
import pydantic
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.runtime_models import RuntimeTraceReport
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.schema import PluginTraits, StructuralTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


def _resolve_traits(
  framework: str,
  traits: StructuralTraits | PluginTraits | pydantic.BaseModel | None = None,
) -> StructuralTraits:
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
    merged = base.model_dump()
    for field_name, val in inspect.getmembers(traits):
      if field_name in StructuralTraits.model_fields and not callable(val):
        merged[field_name] = val
    return StructuralTraits.model_validate(merged)
  return base


_INTERNAL_TRIVIA_KEYS = frozenset(
  {
    "leading_trivia",
    "trailing_trivia",
    "runtime_comment",
    "num_elements",
    "loop_trip_count",
    "row_stride_bytes",
    "slice_stride_bytes",
    "input_height",
    "input_width",
  }
)


def _normalize_py_comment(raw: str) -> str:
  """Normalize an arbitrary comment string (``#``, ``//``, or ``;``) to Python ``# ...`` syntax.

  Args:
      raw: Raw comment text.

  Returns:
      Comment text starting with ``#``.
  """
  stripped = raw.strip()
  if stripped.startswith("#"):
    return stripped
  if stripped.startswith("//"):
    return "#" + stripped[2:]
  if stripped.startswith(";"):
    return "#" + stripped[1:]
  return f"# {stripped}"


def apply_node_trivia_to_stmt(
  stmt: cst.SimpleStatementLine,
  node: LogicalNode,
  include_runtime_comment: bool = False,
) -> cst.SimpleStatementLine:
  """Attach ``leading_trivia``, ``trailing_trivia``, and optional ``runtime_comment`` onto a CST statement.

  Args:
      stmt: The ``cst.SimpleStatementLine`` to decorate.
      node: The ``LogicalNode`` carrying trivia attributes.
      include_runtime_comment: Whether to append ``node.attributes.get('runtime_comment')`` as trailing comment.

  Returns:
      Decorated ``cst.SimpleStatementLine``.
  """
  attrs = node.attributes or {}
  raw_leading = attrs.get("leading_trivia")
  leading_items: list[str] = (
    [s.strip() for s in raw_leading.splitlines() if s.strip()]
    if isinstance(raw_leading, str)
    else ([str(i).strip() for i in raw_leading if str(i).strip()] if isinstance(raw_leading, list) else [])
  )
  leading_lines: list[cst.EmptyLine] = [
    *stmt.leading_lines,
    *(
      cst.EmptyLine(comment=cst.Comment(_normalize_py_comment(item_str)), newline=cst.Newline())
      for item_str in leading_items
    ),
  ]

  trailing_parts: list[str] = []
  if stmt.trailing_whitespace.comment is not None:
    trailing_parts.append(stmt.trailing_whitespace.comment.value)
  raw_trailing = attrs.get("trailing_trivia")
  if isinstance(raw_trailing, str) and raw_trailing.strip():
    trailing_parts.append(_normalize_py_comment(raw_trailing))
  if include_runtime_comment:
    rt_cmt = attrs.get("runtime_comment")
    if isinstance(rt_cmt, str) and rt_cmt.strip():
      trailing_parts.append(_normalize_py_comment(rt_cmt))

  updated = stmt
  if leading_lines != list(stmt.leading_lines):
    updated = updated.with_changes(leading_lines=leading_lines)
  if trailing_parts:
    combined_trailing = "  ".join(trailing_parts)
    updated = updated.with_changes(
      trailing_whitespace=cst.TrailingWhitespace(
        whitespace=cst.SimpleWhitespace("  "),
        comment=cst.Comment(combined_trailing),
        newline=updated.trailing_whitespace.newline,
      )
    )
  return updated


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
    if key in _INTERNAL_TRIVIA_KEYS:
      continue
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
  return not ("." in op_type and not op_type.startswith("nn."))


def _extract_inferred_dim(
  node: LogicalNode,
  runtime_trace: RuntimeTraceReport | None,
) -> int | None:
  """Extract runtime-inferred input dimension from `runtime_trace` or `node.shape_metadata`.

  Args:
      node: LogicalNode representing a stateful layer.
      runtime_trace: Optional on-device `RuntimeTraceReport`.

  Returns:
      Inferred integer input feature/channel dimension, or None.
  """
  if runtime_trace is not None and node.id in runtime_trace.resolved_constructors:
    return runtime_trace.resolved_constructors[node.id].inferred_in_features
  if isinstance(node.shape_metadata, tuple) and node.shape_metadata:
    last_dim = node.shape_metadata[-1]
    if isinstance(last_dim, int):
      return last_dim
  return None


def generate_layer_init(
  node: LogicalNode,
  framework: str,
  semantics: "SemanticsManager | None",
  traits: StructuralTraits | None = None,
  runtime_trace: RuntimeTraceReport | None = None,
) -> cst.SimpleStatementLine:
  """Generate the LibCST initialization assignment statement for a stateful layer node.

  Args:
      node: The LogicalNode representing the layer to initialize.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for looking up variant constructors.
      traits: Optional StructuralTraits for the target framework.
      runtime_trace: Optional RuntimeTraceReport for concretizing lazy layer dimensions.

  Returns:
      LibCST SimpleStatementLine representing the layer attribute initialization.
  """
  resolved_traits = _resolve_traits(framework, traits)
  kind: str = str(node.op_type)
  leaf_op = kind.split(".")[-1]
  if leaf_op.startswith("Lazy"):
    leaf_op = leaf_op.removeprefix("Lazy")
    kind = leaf_op

  if semantics is not None:
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
    kind = f"{prefix}.{rewritten_leaf}" if prefix else rewritten_leaf

  effective_attrs: dict[str, AttributeValue] = dict(node.attributes) if node.attributes else {}
  dim_kw = resolved_traits.lazy_layer_dim_arg_map.get(leaf_op)
  if dim_kw:
    inferred_dim = _extract_inferred_dim(node, runtime_trace)
    if inferred_dim is not None:
      existing_val = effective_attrs.get(dim_kw)
      if existing_val is None or str(existing_val) == "?":
        effective_attrs[dim_kw] = str(inferred_dim)

  args_str = format_args_from_metadata(effective_attrs)
  if resolved_traits.requires_rng_in_init and "rngs" not in args_str:
    suffix = ", rngs=rngs" if args_str else "rngs=rngs"
    args_str += suffix

  reg_method = resolved_traits.child_registration_method
  code = f"self.{reg_method}('{node.id}', {kind}({args_str}))" if reg_method else f"self.{node.id} = {kind}({args_str})"
  stmt = cst.ensure_type(cst.parse_statement(code), cst.SimpleStatementLine)
  return apply_node_trivia_to_stmt(stmt, node, include_runtime_comment=False)


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

  stmts.extend(
    generate_layer_init(node, framework, semantics, resolved_traits) for node in nodes if is_stateful_layer(node)
  )

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
