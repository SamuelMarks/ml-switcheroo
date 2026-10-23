"""Forward method construction strategies."""

import typing

import libcst as cst
from libcst import matchers as m

from ml_switcheroo.core.compiler.backends.python_strategies.init import format_args_from_metadata
from ml_switcheroo.core.compiler.backends.python_strategies.sharding import (
  format_partition_spec,
  format_partition_spec_tf,
  format_partition_spec_torch,
)
from ml_switcheroo.core.compiler.ir import LogicalNode, PartitionSpec

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.compiler.backends.python import PythonBackend
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.semantics.schema import StructuralTraits


def _emit_jax_sharding(current_var: str, sharding: PartitionSpec) -> str:
  """Format a JAX sharding constraint statement.

  Args:
      current_var: Current tensor variable name.
      sharding: Sharding specification object.

  Returns:
      Formatted Python statement string.
  """
  spec_str = format_partition_spec(sharding)
  return f"{current_var} = jax.lax.with_sharding_constraint({current_var}, {spec_str})"


def _emit_torch_sharding(current_var: str, sharding: PartitionSpec) -> str:
  """Format a PyTorch DTensor distribute_tensor statement.

  Args:
      current_var: Current tensor variable name.
      sharding: Sharding specification object.

  Returns:
      Formatted Python statement string.
  """
  spec_str = format_partition_spec_torch(sharding)
  return f"{current_var} = distribute_tensor({current_var}, self.mesh, [{spec_str}])"


def _emit_tf_sharding(current_var: str, sharding: PartitionSpec) -> str:
  """Format a Keras/TensorFlow distribution layout comment statement.

  Args:
      current_var: Current tensor variable name.
      sharding: Sharding specification object.

  Returns:
      Formatted Python statement string.
  """
  spec_str = format_partition_spec_tf(sharding)
  return f"pass  # {current_var} = keras.distribution.layout({spec_str})"


def _emit_mlx_sharding(current_var: str, _sharding: PartitionSpec) -> str:
  """Format an MLX distributed sharding comment statement.

  Args:
      current_var: Current tensor variable name.
      _sharding: Sharding specification object.

  Returns:
      Formatted Python statement string.
  """
  return f"pass  # MLX: mx.distributed.shard({current_var})"


_SHARDING_STYLE_EMITTERS: dict[str, typing.Callable[[str, PartitionSpec], str]] = {
  "with_sharding_constraint": _emit_jax_sharding,
  "distribute_tensor": _emit_torch_sharding,
  "distribution_layout": _emit_tf_sharding,
  "distributed_shard": _emit_mlx_sharding,
}


def build_forward(
  nodes: list[LogicalNode],
  framework: str,
  semantics: "SemanticsManager | None",
  traits: "StructuralTraits",
  backend: "PythonBackend | None" = None,
) -> cst.FunctionDef:
  """Construct the forward/inference method CST node for a synthesized Python class.

  Args:
      nodes: Ordered list of LogicalNodes in the model graph.
      framework: Target framework identifier string.
      semantics: Optional SemanticsManager for operation variant resolution.
      traits: StructuralTraits for the target framework.
      backend: Optional PythonBackend instance providing layer statefulness checks.

  Returns:
      Constructed LibCST FunctionDef node for the model's inference method.
  """
  stmts: list[cst.BaseStatement] = []
  input_nodes = [n for n in nodes if n.op_type == "Input"]
  input_arg_name = "x"
  if input_nodes:
    inp_attrs = input_nodes[0].attributes
    input_arg_name = str(inp_attrs.get("name", input_nodes[0].id))

  current_var = input_arg_name
  var_map = {input_nodes[0].id: input_arg_name} if input_nodes else {}

  for node in nodes:
    op_type: str = str(node.op_type)
    if op_type == "Input":
      continue
    if op_type == "Output":
      stmts.append(cst.parse_statement(f"return {current_var}"))
      continue

    if backend is not None and hasattr(backend, "_is_stateful_layer") and backend._is_stateful_layer(node):
      line = f"{current_var} = self.{node.id}({current_var})"
      stmts.append(cst.parse_statement(line))
    else:
      func_api: str = op_type
      if semantics and hasattr(semantics, "resolve_variant"):
        definition = semantics.resolve_variant(func_api, framework)
        if not definition and hasattr(semantics, "get_definition"):
          rev = semantics.get_definition(func_api)
          if rev:
            abstract_id, _ = rev
            definition = semantics.resolve_variant(abstract_id, framework)

        if definition and "api" in definition:
          func_api = str(definition["api"])

      args_str = current_var
      extra_args = format_args_from_metadata(node.attributes)
      if extra_args:
        args_str += f", {extra_args}"
      line = f"{current_var} = {func_api}({args_str})"
      stmts.append(cst.parse_statement(line))

    sharding_val = node.sharding
    sharding_style = getattr(traits, "sharding_style", None)
    sharding_emitter = _SHARDING_STYLE_EMITTERS.get(sharding_style or "")
    if sharding_val is not None and sharding_emitter is not None:
      sharding_line = sharding_emitter(current_var, sharding_val)
      stmts.append(cst.parse_statement(sharding_line))

    var_map[node.id] = current_var

  if not stmts or not m.matches(stmts[-1], m.SimpleStatementLine(body=[m.Return()])):
    stmts.append(cst.parse_statement(f"return {current_var}"))

  func_name = traits.forward_method if traits.forward_method else "forward"

  return cst.FunctionDef(
    name=cst.Name(func_name),
    params=cst.Parameters(
      params=[
        cst.Param(name=cst.Name("self")),
        cst.Param(name=cst.Name(input_arg_name)),
      ]
    ),
    body=cst.IndentedBlock(body=stmts),
  )
