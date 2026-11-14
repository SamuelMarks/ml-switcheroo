"""Live ``ProxyTensor`` Tracing and Shape-Inference Hook Utilities for ``ir_bridge``.

Provides ``_resolve_ir_dtype``, ``_run_shape_inference_hooks``, ``_eval_proxy_ast_expr``,
and ``trace_with_zero_proxy_tensors``.
"""

from __future__ import annotations

import ast
import importlib
import logging
import operator as _op
import typing

from ml_switcheroo_ir import (
  DType,
  LogicalGraph,
  LogicalNode,
  propagate_shapes_and_constants,
)

from ml_switcheroo.core.compiler.compiler_passes import ensure_compiler_and_zero_on_path
from ml_switcheroo.exceptions import GraphExtractionError, MLSwitcherooError


@typing.runtime_checkable
class _ShapeInferenceModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.transforms.passes.shape_inference`` module."""

  def shape_inference_pass(self, graph: LogicalGraph) -> LogicalGraph | None:
    """Run shape inference over ``graph``.

    Args:
        graph: Target ``LogicalGraph``.

    Returns:
        Optional updated ``LogicalGraph`` or None.
    """


@typing.runtime_checkable
class _LearnedShapesModuleProtocol(typing.Protocol):
  """Protocol for ``shape_inference`` modules exposing ``annotate_learned_shapes``."""

  def annotate_learned_shapes(
    self,
    graph: LogicalGraph,
    observed_shapes: typing.Mapping[str, list[int]],
  ) -> None:
    """Annotate observed runtime shapes onto ``graph``.

    Args:
        graph: Target ``LogicalGraph``.
        observed_shapes: Mapping from node identifier to concrete shape list.
    """


class _TracingStateProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.tracing.global_tracing_state``."""

  def start_tracing(self) -> None:
    """Start proxy tensor tracing."""
    ...

  def add_node(self, node: LogicalNode) -> None:
    """Add a node to the active graph.

    Args:
        node: ``LogicalNode`` to add.
    """
    ...

  def stop_tracing(self) -> LogicalGraph:
    """Stop proxy tensor tracing and return the traced ``LogicalGraph``.

    Returns:
        Traced ``LogicalGraph``.
    """
    ...


class _ProxyTensorProtocol(typing.Protocol):
  """Protocol for ``ProxyTensor`` instances.

  Attributes:
      id: The id (``str``).
      shape: The shape (``list[int]``).
      dtype: The dtype (``str``).
  """

  id: str
  shape: list[int]
  dtype: str


class _ProxyTensorCtorProtocol(typing.Protocol):
  """Protocol for ``ProxyTensor`` constructor."""

  def __call__(self, *, id: str, shape: list[int], dtype: str) -> _ProxyTensorProtocol:
    """Construct a ``ProxyTensor`` instance.

    Args:
        id: Node identifier.
        shape: Tensor shape list.
        dtype: Data type string.

    Returns:
        ``ProxyTensor`` instance.
    """
    ...


@typing.runtime_checkable
class _TracingModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.tracing``.

  Attributes:
      global_tracing_state: The global tracing state (``_TracingStateProtocol | None``).
      ProxyTensor: The proxytensor (``_ProxyTensorCtorProtocol | None``).
  """

  global_tracing_state: _TracingStateProtocol | None
  ProxyTensor: _ProxyTensorCtorProtocol | None


def _resolve_ir_dtype(dtype_val: str | DType | None) -> DType:
  """Resolve a string or ``DType`` value into an ``ml_switcheroo_ir.DType`` enum.

  Args:
      dtype_val: String dtype identifier or ``DType``.

  Returns:
      Resolved ``DType`` enum member (defaults to ``DType.float32``).
  """
  if isinstance(dtype_val, DType):
    return dtype_val
  clean = str(dtype_val or "float32").strip().lower()
  for member in DType:
    if member.value == clean or member.name == clean:
      return member
  return DType.float32


def _run_shape_inference_hooks(
  graph: LogicalGraph,
  observed_shapes: typing.Mapping[str, list[int]],
) -> None:
  """Propagate shapes and invoke optional ``ml-switcheroo-compiler`` shape-inference hooks.

  Args:
      graph: Target ``LogicalGraph`` to annotate.
      observed_shapes: Concrete runtime tensor shapes keyed by symbol/node name.
  """
  try:
    propagate_shapes_and_constants(graph)
    shape_inf_mod = importlib.import_module("ml_switcheroo_compiler.transforms.passes.shape_inference")
    if not isinstance(shape_inf_mod, _ShapeInferenceModuleProtocol) or not isinstance(
      shape_inf_mod, _LearnedShapesModuleProtocol
    ):
      raise ImportError("shape_inference module missing required hooks")
    shape_inf_mod.shape_inference_pass(graph)
    shape_inf_mod.annotate_learned_shapes(graph, observed_shapes)
  except (
    ImportError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    RuntimeError,
    MLSwitcherooError,
  ) as exc:
    logging.getLogger(__name__).debug("Shape inference hook skipped: %s", exc)


def _eval_proxy_ast_expr(
  node: ast.AST,
  env: dict[str, _ProxyTensorProtocol | int | float | str | bool | None],
) -> _ProxyTensorProtocol | int | float | str | bool | None:
  """Evaluate a Python AST expression on ``ProxyTensor`` inputs without ``compile``/``exec``.

  Args:
      node: Python ``ast.AST`` expression node.
      env: Local variable mapping.

  Returns:
      Evaluated ``ProxyTensor`` or scalar literal.

  Raises:
      GraphExtractionError: If ``node`` is not a supported ``ProxyTensor`` AST expression.
  """
  if isinstance(node, ast.Name):
    return env[node.id]
  if not isinstance(node, ast.BinOp):
    raise GraphExtractionError(f"Unsupported proxy AST expression: {type(node).__name__}")
  lhs = _eval_proxy_ast_expr(node.left, env)
  rhs = _eval_proxy_ast_expr(node.right, env)
  res_add: _ProxyTensorProtocol | int | float | str | bool | None = _op.add(lhs, rhs)
  return res_add


def _execute_first_public_function(
  code: str,
  proxy_inputs: list[_ProxyTensorProtocol],
) -> None:
  """Execute the first top-level public function in ``code`` on ``proxy_inputs``.

  Args:
      code: Source Python code string.
      proxy_inputs: Initialized ``ProxyTensor`` argument list.

  Raises:
      RuntimeError: If a function body statement is not a non-empty ``return`` statement.
  """
  tree = ast.parse(code, filename="<zero_proxy_trace>")
  for stmt in tree.body:
    if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
      arg_names = [a.arg for a in stmt.args.args]
      trace_ns: dict[str, _ProxyTensorProtocol | int | float | str | bool | None] = {
        arg_names[i]: proxy_inputs[i] for i in range(min(len(arg_names), len(proxy_inputs)))
      }
      for body_stmt in stmt.body:
        if isinstance(body_stmt, ast.Return) and body_stmt.value is not None:
          _eval_proxy_ast_expr(body_stmt.value, trace_ns)
        else:
          raise RuntimeError("Unsupported or raising statement in trace")
      break


def trace_with_zero_proxy_tensors(
  code: str,
  input_shapes: list[tuple[int, ...]],
  dtype: str = "float32",
  seed: int = 42,
) -> LogicalGraph | None:
  """Capture a live ``LogicalGraph`` using ``ml_switcheroo_compiler.tracing.state.global_tracing_state``.

  Args:
      code: Source Python code string.
      input_shapes: Inferred or configured input tensor shapes.
      dtype: Input tensor dtype string.
      seed: Deterministic random seed.

  Returns:
      Captured ``LogicalGraph`` if tracing succeeded and recorded nodes, otherwise None.
  """
  ensure_compiler_and_zero_on_path()
  try:
    tracing_mod = importlib.import_module("ml_switcheroo_compiler.tracing")
  except ImportError:
    return None

  if not isinstance(tracing_mod, _TracingModuleProtocol):
    return None
  global_state = tracing_mod.global_tracing_state
  proxy_cls = tracing_mod.ProxyTensor
  if global_state is None or proxy_cls is None:
    return None

  global_state.start_tracing()
  ir_dt = _resolve_ir_dtype(dtype)
  try:
    proxy_inputs: list[_ProxyTensorProtocol] = []
    for idx, shape in enumerate(input_shapes):
      node_id = f"input_{idx}" if idx > 0 else "x"
      global_state.add_node(LogicalNode(id=node_id, op_type="Input", shape_metadata=tuple(shape), dtype=ir_dt))
      proxy_inputs.append(proxy_cls(id=node_id, shape=list(shape), dtype=dtype))
    _execute_first_public_function(code, proxy_inputs)
  except (
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    IndexError,
    ZeroDivisionError,
    RuntimeError,
    MLSwitcherooError,
  ) as exc:
    logging.getLogger(__name__).debug("Zero proxy trace ended early: %s", exc)
  finally:
    traced_graph: LogicalGraph = global_state.stop_tracing()

  return traced_graph if traced_graph.nodes else None


__all__ = [
  "_LearnedShapesModuleProtocol",
  "_ProxyTensorCtorProtocol",
  "_ProxyTensorProtocol",
  "_ShapeInferenceModuleProtocol",
  "_TracingModuleProtocol",
  "_TracingStateProtocol",
  "_eval_proxy_ast_expr",
  "_resolve_ir_dtype",
  "_run_shape_inference_hooks",
  "trace_with_zero_proxy_tensors",
]
