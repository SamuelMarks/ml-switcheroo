"""Initialization construction strategies."""

from typing import Any, Dict, List, Set, Tuple, cast

import libcst as cst

from ml_switcheroo.core.compiler.ir import LogicalNode

_LAYER_PREFIX_MAP: Dict[str, str] = {
  "torch": "nn",
  "jax": "nnx",
  "flax": "nnx",
  "flax_nnx": "nnx",
  "keras": "keras.layers",
  "tensorflow": "tf.keras.layers",
  "mlx": "nn",
  "paxml": "pl",
}

_MODULE_ALIAS_STRIP: Dict[str, Tuple[str, str]] = {
  "torch": ("torch.nn.", "nn."),
  "mlx": ("mlx.nn.", "nn."),
}

_FUNCTIONAL_NAMESPACE_SKIP: Dict[str, str] = {
  "torch": "functional",
  "mlx": "mlx.core.",
}

_RNGS_FRAMEWORKS: Set[str] = {"jax", "flax", "flax_nnx"}


def format_args_from_metadata(metadata: Any) -> str:
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
  op_type: str = str(getattr(node, "op_type", None) or getattr(node, "kind", ""))
  if op_type in ["Input", "Output"]:
    return False
  if "." in op_type and not op_type.startswith("nn."):
    return False
  return True


def generate_layer_init(node: LogicalNode, framework: str, semantics: Any) -> cst.SimpleStatementLine:
  """Generate the LibCST initialization assignment statement for a stateful layer node.

  Args:
      node: The LogicalNode representing the layer to initialize.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for looking up variant constructors.

  Returns:
      LibCST SimpleStatementLine representing the layer attribute initialization.
  """
  kind: str = str(getattr(node, "op_type", None) or getattr(node, "kind", ""))
  if semantics and hasattr(semantics, "resolve_variant"):
    definition = semantics.resolve_variant(kind, framework)
    if definition and "api" in definition:
      resolved_api = definition["api"]
      skip_token = _FUNCTIONAL_NAMESPACE_SKIP.get(framework)
      if not (skip_token and skip_token in resolved_api):
        kind = resolved_api

  alias_strip = _MODULE_ALIAS_STRIP.get(framework)
  if alias_strip and kind.startswith(alias_strip[0]):
    kind = alias_strip[1] + kind[len(alias_strip[0]) :]

  if "." not in kind:
    prefix = _LAYER_PREFIX_MAP.get(framework)
    if prefix:
      kind = f"{prefix}.{kind}"

    if alias_strip and alias_strip[0] == "mlx.nn.":
      if kind.endswith("SwiGLU"):
        kind = kind.replace("SwiGLU", "silu")
      if kind.endswith("RoPE"):
        kind = "nn.RoPE"
      if kind.endswith("VisionPatchEmbedding"):
        kind = "nn.Conv2d"

  node_attrs = getattr(node, "attributes", {})
  args_str = format_args_from_metadata(node_attrs)
  if framework in _RNGS_FRAMEWORKS:
    if "rngs" not in args_str:
      suffix = ", rngs=rngs" if args_str else "rngs=rngs"
      args_str += suffix

  if _LAYER_PREFIX_MAP.get(framework) == "pl":
    code = f"self.create_child('{node.id}', {kind}({args_str}))"
  else:
    code = f"self.{node.id} = {kind}({args_str})"
  return cast(cst.SimpleStatementLine, cst.parse_statement(code))


def build_init(nodes: List[LogicalNode], framework: str, semantics: Any, traits: Any) -> cst.FunctionDef:
  """Build the `__init__` or `setup` CST FunctionDef node for a synthesized model class.

  Args:
      nodes: Ordered list of LogicalNodes in the model graph.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for layer constructor resolution.
      traits: StructuralTraits for the target framework.

  Returns:
      Constructed LibCST FunctionDef node for model initialization.
  """
  stmts: List[cst.BaseStatement] = []
  if traits.requires_super_init:
    stmts.append(cst.parse_statement("super().__init__()"))

  for node in nodes:
    if is_stateful_layer(node):
      assignment = generate_layer_init(node, framework, semantics)
      stmts.append(assignment)

  params = [cst.Param(name=cst.Name("self"))]
  if framework in _RNGS_FRAMEWORKS:
    params.append(
      cst.Param(
        name=cst.Name("rngs"),
        annotation=cst.Annotation(cst.parse_expression("nnx.Rngs")),
      )
    )

  if not stmts:
    stmts.append(cst.parse_statement("pass"))

  init_name = "setup" if _LAYER_PREFIX_MAP.get(framework) == "pl" else "__init__"
  return cst.FunctionDef(
    name=cst.Name(init_name),
    params=cst.Parameters(params=params),
    body=cst.IndentedBlock(body=stmts),
  )
