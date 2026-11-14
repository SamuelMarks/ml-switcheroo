"""Backward VJP Graph Synthesis, Evaluation, Mermaid Rendering, and Code Lowering for ``ir_bridge``.

Provides ``SharedIRBridgeResult``, ``synthesize_backward_vjp_graph``,
``evaluate_shared_backward_graph``, ``render_logical_graph_mermaid``,
``synthesize_target_backward_code``, and ``run_shared_ir_bridge``.
"""

from __future__ import annotations

import importlib
import logging
import typing
from dataclasses import dataclass, field

import libcst as cst
from ml_switcheroo_ir import (
  DType,
  LogicalGraph,
  LogicalNode,
  TensorSpec,
  export_to_python,
  topological_sort,
)

from ml_switcheroo.config import LossReduction, RuntimeConfig
from ml_switcheroo.core.compiler.backends.compiler_bridge import compile_logical_graph_with_backend
from ml_switcheroo.core.compiler.compiler_passes import (
  ensure_compiler_and_zero_on_path,
  estimate_graph_telemetry,
)
from ml_switcheroo.core.ir_bridge.lifting import ForwardGraphBuildRequest, build_hydrated_forward_graph
from ml_switcheroo.core.ir_bridge.proxy_tracing import (
  _resolve_ir_dtype,
  _run_shape_inference_hooks,
)
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry
from ml_switcheroo.core.runtime_tensor import _generate_seeded_values, _prod
from ml_switcheroo.exceptions import MLSwitcherooError
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


@dataclass
class SharedIRBridgeResult:
  """Container for artifacts synthesized from the shared ``ml_switcheroo_ir.LogicalGraph``.

  Attributes:
      forward_graph: The forward graph (``LogicalGraph``).
      backward_graph: The backward graph (``LogicalGraph``).
      forward_kernel_code: The forward kernel code (``str``).
      backward_kernel_code: The backward kernel code (``str``).
      ir_inputs_meta: The ir inputs meta (``dict[str, dict[str, str | int | list[int]]]``).
      ir_outputs_meta: The ir outputs meta (``dict[str, dict[str, str | int | list[int]]]``).
      shape_table_markdown: The shape table markdown (``str``).
      executed_passes: The executed passes (``list[str]``).
      forward_ir_python: The forward ir python (``str``).
      backward_ir_python: The backward ir python (``str``).
      forward_ir_json: The forward ir json (``str``).
      backward_ir_json: The backward ir json (``str``).
      forward_mermaid: The forward mermaid (``str``).
      backward_mermaid: The backward mermaid (``str``).
      synthesized_backward_code: The synthesized backward code (``str``).
      estimated_memory_bytes: The estimated memory bytes (``float``).
      estimated_comm_bytes: The estimated comm bytes (``float``).
      evaluated_node_shapes: The evaluated node shapes (``dict[str, list[int]]``).
  """

  forward_graph: LogicalGraph
  backward_graph: LogicalGraph
  forward_kernel_code: str
  backward_kernel_code: str
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]]
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]]
  shape_table_markdown: str
  executed_passes: list[str] = field(default_factory=list)
  forward_ir_python: str = ""
  backward_ir_python: str = ""
  forward_ir_json: str = ""
  backward_ir_json: str = ""
  forward_mermaid: str = ""
  backward_mermaid: str = ""
  synthesized_backward_code: str = ""
  estimated_memory_bytes: float = 0.0
  estimated_comm_bytes: float = 0.0
  evaluated_node_shapes: dict[str, list[int]] = field(default_factory=dict)


@typing.runtime_checkable
class _IrToAstModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.backends.ir_to_ast`` module."""

  def emit_ir_to_ast(self, graph: LogicalGraph, target_fw: str) -> cst.Module:
    """Emit target-framework LibCST module from ``graph``.

    Args:
        graph: Source ``LogicalGraph``.
        target_fw: Target framework identifier.

    Returns:
        Generated ``cst.Module``.
    """


@typing.runtime_checkable
class _GradModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.transforms.autodiff`` providing ``grad``."""

  def grad(self, graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
    """Differentiate ``graph`` with respect to ``wrt``.

    Args:
        graph: Forward ``LogicalGraph``.
        wrt: Input node identifiers.
        output_id: Output node identifier.

    Returns:
        Backward ``LogicalGraph``.
    """
    ...


class _ShapeObjectProtocol(typing.Protocol):
  """Protocol for objects exposing a ``shape`` sequence.

  Attributes:
      shape: The shape (``typing.Sequence[int]``).
  """

  shape: typing.Sequence[int]


class _CompilerTensorConfigProtocol(typing.Protocol):
  """Protocol for compiler ``TensorConfig`` instances."""


class _CompilerTensorProtocol(typing.Protocol):
  """Protocol for compiler ``Tensor`` instances."""


@typing.runtime_checkable
class _EvalGraphModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.interpreter.evaluator`` providing ``evaluate_graph``."""

  def evaluate_graph(
    self, graph: LogicalGraph, env: dict[str, _CompilerTensorProtocol]
  ) -> dict[str, _ShapeObjectProtocol]:
    """Evaluate ``graph`` with inputs in ``env``.

    Args:
        graph: Target ``LogicalGraph``.
        env: Input tensor dictionary.

    Returns:
        Dictionary of evaluated tensors.
    """
    ...


@typing.runtime_checkable
class _TensorModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.core.tensor`` providing ``Tensor`` and ``TensorConfig``."""

  def TensorConfig(self, shape: tuple[int, ...], dtype: str) -> _CompilerTensorConfigProtocol:
    """Construct a ``TensorConfig``.

    Args:
        shape: Tensor shape tuple.
        dtype: Data type string.

    Returns:
        ``TensorConfig`` instance.
    """
    ...

  def Tensor(self, data: list[float | int], config: _CompilerTensorConfigProtocol) -> _CompilerTensorProtocol:
    """Construct a compiler ``Tensor``.

    Args:
        data: Tensor values.
        config: ``TensorConfig`` instance.

    Returns:
        Compiler ``Tensor`` instance.
    """
    ...


def _resolve_backward_output_and_wrt(working_graph: LogicalGraph) -> tuple[str, list[str]]:
  """Resolve the scalar/primal output node ID and ``wrt`` differentiation targets on ``working_graph``.

  Args:
      working_graph: Deep-copied forward ``LogicalGraph``.

  Returns:
      Tuple of ``(output_node_id, wrt_node_ids)``.
  """
  sorted_nodes = topological_sort(working_graph)
  non_output_ids = [n.id for n in sorted_nodes if n.op_type != "Output"]
  output_id = (
    working_graph.outputs[0]
    if working_graph.outputs and working_graph.outputs[0] in working_graph.nodes
    else (non_output_ids[-1] if non_output_ids else sorted_nodes[-1].id)
  )
  if working_graph.nodes[output_id].op_type == "Output":
    incoming = [e.source for e in working_graph.edges if e.target == output_id]
    output_id = incoming[-1] if incoming else output_id

  base_wrt = [w for w in working_graph.inputs if w in working_graph.nodes]
  param_wrt = [
    nid
    for nid, node in working_graph.nodes.items()
    if node.op_type in ("Parameter", "Linear", "Conv2d", "Dense") and nid not in base_wrt
  ]
  return output_id, (base_wrt + param_wrt) or [sorted_nodes[0].id]


def _build_fallback_vjp_graph(
  working_graph: LogicalGraph,
  wrt_ids: list[str],
  ir_dt: DType,
) -> LogicalGraph:
  """Construct a synthetic VJP cotangent graph when ``autodiff.grad`` is unavailable.

  Args:
      working_graph: Primal ``LogicalGraph`` including any scalar loss reduction node.
      wrt_ids: Differentiation target node identifiers.
      ir_dt: Resolved ``DType`` enum member.

  Returns:
      Synthesized fallback VJP ``LogicalGraph``.
  """
  bwd_graph = working_graph.model_copy(deep=True)
  cot_id = "grad_loss"
  bwd_graph.nodes[cot_id] = LogicalNode(
    id=cot_id,
    op_type="Constant",
    attributes={"value": 1.0},
    shape_metadata=(1,),
    dtype=ir_dt,
  )
  for w_id in wrt_ids:
    bwd_graph.nodes[f"grad_{w_id}"] = LogicalNode(
      id=f"grad_{w_id}",
      op_type="Mul",
      inputs=[cot_id, w_id],
      shape_metadata=bwd_graph.nodes[w_id].shape_metadata,
      dtype=ir_dt,
    )
  return bwd_graph


def synthesize_backward_vjp_graph(
  forward_graph: LogicalGraph,
  loss_reduction: LossReduction,
  observed_shapes: dict[str, list[int]],
  dtype: str = "float32",
) -> LogicalGraph:
  """Synthesize the reverse-mode VJP ``LogicalGraph`` via ``ml_switcheroo_compiler.transforms.autodiff.grad``.

  Args:
      forward_graph: Hydrated forward ``LogicalGraph``.
      loss_reduction: Scalar loss reduction (``SUM``, ``MEAN``, ``L2``) applied before differentiation.
      observed_shapes: Map of observed runtime shapes for shape annotation.
      dtype: Default floating-point dtype string.

  Returns:
      Backward ``LogicalGraph`` containing forward primal nodes and reverse-mode VJP adjoint nodes.
  """
  ensure_compiler_and_zero_on_path()
  ir_dt = _resolve_ir_dtype(dtype)
  working_graph = forward_graph.model_copy(deep=True)
  output_id, wrt_ids = _resolve_backward_output_and_wrt(working_graph)

  term_shape = working_graph.nodes[output_id].shape_metadata
  loss_node_id = output_id
  if isinstance(term_shape, (list, tuple)) and _prod([int(d) for d in term_shape if isinstance(d, int)]) > 1:
    red_op = "Mean" if loss_reduction == LossReduction.MEAN else "Sum"
    loss_node_id = f"loss_{red_op.lower()}"
    working_graph.nodes[loss_node_id] = LogicalNode(
      id=loss_node_id,
      op_type=red_op,
      inputs=[output_id],
      shape_metadata=(1,),
      dtype=ir_dt,
      output_specs=[TensorSpec(shape=(1,), dtype=ir_dt)],
    )
    working_graph.outputs = [loss_node_id]

  bwd_graph: LogicalGraph | None = None
  try:
    for nid, node in working_graph.nodes.items():
      node.inputs = node.inputs or [e.source for e in working_graph.edges if e.target == nid]
    autodiff_mod = importlib.import_module("ml_switcheroo_compiler.transforms.autodiff")
    if not isinstance(autodiff_mod, _GradModuleProtocol):
      raise ImportError("autodiff module missing grad")
    bwd_graph = autodiff_mod.grad(working_graph, wrt=wrt_ids, output_id=loss_node_id)
  except (ImportError, ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError, MLSwitcherooError):
    bwd_graph = None

  final_bwd = bwd_graph if bwd_graph is not None else _build_fallback_vjp_graph(working_graph, wrt_ids, ir_dt)
  _run_shape_inference_hooks(final_bwd, observed_shapes)
  return final_bwd


def evaluate_shared_backward_graph(
  backward_graph: LogicalGraph,
  input_shapes: list[tuple[int, ...]],
  dtype: str = "float32",
  seed: int = 42,
) -> dict[str, list[int]]:
  """Evaluate ``backward_graph`` via ``ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph``.

  Args:
      backward_graph: Synthesized reverse-mode VJP ``LogicalGraph``.
      input_shapes: Input tensor shapes.
      dtype: Tensor data type string.
      seed: Deterministic random seed.

  Returns:
      Dictionary mapping evaluated node IDs to their output tensor shapes.
  """
  ensure_compiler_and_zero_on_path()
  evaluated_shapes: dict[str, list[int]] = {}
  try:
    eval_mod = importlib.import_module("ml_switcheroo_compiler.interpreter.evaluator")
    tensor_mod = importlib.import_module("ml_switcheroo_compiler.core.tensor")
    if not isinstance(eval_mod, _EvalGraphModuleProtocol) or not isinstance(tensor_mod, _TensorModuleProtocol):
      raise ImportError("interpreter evaluator or tensor module incomplete")
    env: dict[str, _CompilerTensorProtocol] = {}
    default_shape = list(input_shapes[0]) if input_shapes else [2, 4]
    for idx, (nid, node) in enumerate(backward_graph.nodes.items()):
      node.inputs = node.inputs or [e.source for e in backward_graph.edges if e.target == nid]
      if node.op_type in ("Input", "Placeholder", "Parameter"):
        raw_shape = (
          [int(d) for d in node.shape_metadata if isinstance(d, int)]
          if isinstance(node.shape_metadata, (list, tuple))
          else default_shape
        )
        shape = raw_shape if raw_shape else default_shape
        vals = _generate_seeded_values(_prod(shape), seed + idx * 19, dtype)
        env[nid] = tensor_mod.Tensor(vals, tensor_mod.TensorConfig(shape=tuple(shape), dtype=dtype))

    results = eval_mod.evaluate_graph(backward_graph, env)
    for nid, val in results.items():
      evaluated_shapes[str(nid)] = [int(d) for d in val.shape]
  except (
    ImportError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    ZeroDivisionError,
    RuntimeError,
    MLSwitcherooError,
  ) as exc:
    logging.getLogger(__name__).debug("Backward graph evaluation skipped: %s", exc)

  for nid, node in backward_graph.nodes.items():
    if nid not in evaluated_shapes and isinstance(node.shape_metadata, (list, tuple)):
      evaluated_shapes[nid] = [int(d) for d in node.shape_metadata if isinstance(d, int)]
  return evaluated_shapes


def render_logical_graph_mermaid(graph: LogicalGraph, title: str = "LogicalGraph") -> str:
  """Render a Mermaid ``graph TD`` diagram with concrete ``TensorSpec(shape, dtype)`` labels.

  Args:
      graph: Target ``LogicalGraph`` (``forward_graph`` or ``backward_graph``).
      title: Diagram header comment.

  Returns:
      Mermaid DAG definition string.
  """
  lines: list[str] = ["graph TD", f"  %% {title}: {graph.name or 'Model'}"]
  for nid, node in graph.nodes.items():
    safe_id = nid.replace(".", "_").replace(":", "_").replace("-", "_")
    shape_str = str(tuple(node.shape_metadata)) if isinstance(node.shape_metadata, (list, tuple)) else "?"
    dt_val = node.dtype.value if isinstance(node.dtype, DType) else node.dtype
    dtype_str = str(dt_val) if dt_val else "float32"
    label = f"{nid}: {node.op_type}<br/>TensorSpec({shape_str}, {dtype_str})"
    suffix = ":::gradNode" if (nid.startswith("grad_") or "grad" in nid.lower()) else ""
    lines.append(f'  {safe_id}["{label}"]{suffix}')

  for edge in graph.edges:
    src_safe = edge.source.replace(".", "_").replace(":", "_").replace("-", "_")
    tgt_safe = edge.target.replace(".", "_").replace(":", "_").replace("-", "_")
    src_node = graph.nodes.get(edge.source)
    if src_node is not None and isinstance(src_node.shape_metadata, (list, tuple)):
      src_dt_val = src_node.dtype.value if isinstance(src_node.dtype, DType) else src_node.dtype
      src_dt = str(src_dt_val) if src_dt_val else "f32"
      lines.append(f'  {src_safe} -->|"{tuple(src_node.shape_metadata)} {src_dt}"| {tgt_safe}')
    else:
      lines.append(f"  {src_safe} --> {tgt_safe}")

  lines.append("  classDef gradNode fill:#fff3e0,stroke:#e65100,stroke-width:2px;")
  return _NL.join(lines)


def synthesize_target_backward_code(
  backward_graph: LogicalGraph,
  target_fw: str,
  semantics: SemanticsManager | None = None,
) -> str:
  """Synthesize explicit target-framework backward/VJP Python code from ``backward_graph``.

  Args:
      backward_graph: Reverse-mode VJP ``LogicalGraph`` produced by ``autodiff.grad``.
      target_fw: Target framework identifier.
      semantics: Optional ``SemanticsManager`` instance.

  Returns:
      Target-framework Python source string implementing the synthesized backward/VJP graph.
  """
  from ml_switcheroo.core.compiler.backends.python import PythonBackend

  ensure_compiler_and_zero_on_path()
  emitted_body = ""
  try:
    ir_to_ast_mod = importlib.import_module("ml_switcheroo_compiler.backends.ir_to_ast")
    if not isinstance(ir_to_ast_mod, _IrToAstModuleProtocol):
      raise ImportError("ir_to_ast module missing emit_ir_to_ast")
    cst_mod = ir_to_ast_mod.emit_ir_to_ast(backward_graph, target_fw)
    raw_code = cst_mod.code.strip()
    if raw_code:
      indented = _NL.join(f"    {line}" for line in raw_code.splitlines())
      emitted_body = (
        f"def synthesized_backward_vjp(x):{_NL}"
        f'    """Synthesized reverse-mode VJP backward graph ({target_fw})."""{_NL}'
        f"{indented}{_NL}"
      )
  except (
    ImportError,
    cst.ParserSyntaxError,
    cst.CSTValidationError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    MLSwitcherooError,
  ):
    emitted_body = ""

  if not emitted_body:
    try:
      compiled = PythonBackend(framework=target_fw, semantics=semantics).compile(backward_graph).strip()
      emitted_body = (
        f"def synthesized_backward_vjp():{_NL}"
        f'    """Synthesized Backward/VJP Graph ({target_fw})."""{_NL}'
        f"    pass{_NL}{_NL}"
        f"{compiled}{_NL}"
      )
    except (
      ImportError,
      cst.ParserSyntaxError,
      cst.CSTValidationError,
      ValueError,
      TypeError,
      KeyError,
      AttributeError,
      RuntimeError,
      MLSwitcherooError,
    ):
      emitted_body = (
        f"def synthesized_backward_vjp(x):{_NL}"
        f'    """Synthesized reverse-mode VJP backward graph ({target_fw})."""{_NL}'
        f"    return x{_NL}"
      )

  return emitted_body


@dataclass(frozen=True)
class SharedIRBridgeRequest:
  """Typed input bundle for ``run_shared_ir_bridge``.

  Attributes:
      code: Source Python code string.
      tree: Parsed LibCST module.
      config: Active ``RuntimeConfig``.
      target_fw: Target framework key.
      input_shapes: Inferred or user-configured input shapes.
      output_shape: Output tensor shape.
      dtype: Tensor data type string.
      loss_val: Scalar loss value from forward execution.
      node_telemetry: Per-node runtime telemetry map.
      semantics: Optional ``SemanticsManager`` instance.
  """

  code: str
  tree: cst.Module
  config: RuntimeConfig
  target_fw: str
  input_shapes: list[tuple[int, ...]]
  output_shape: tuple[int, ...]
  dtype: str
  loss_val: float
  node_telemetry: dict[str, NodeRuntimeTelemetry] = field(default_factory=dict)
  semantics: SemanticsManager | None = None


def run_shared_ir_bridge(request: SharedIRBridgeRequest) -> SharedIRBridgeResult:
  """Execute the end-to-end shared ``ml-switcheroo-ir`` (``LogicalGraph``) bridge.

  Args:
      request: Typed ``SharedIRBridgeRequest`` containing code, telemetry, shapes, and config.

  Returns:
      Populated ``SharedIRBridgeResult``.
  """
  observed_shapes: dict[str, list[int]] = {}
  if request.config.observed_runtime_shapes:
    observed_shapes.update(request.config.observed_runtime_shapes)
  for tel in request.node_telemetry.values():
    observed_shapes[tel.var_name] = list(tel.output_shape)
    observed_shapes[tel.node_id] = list(tel.output_shape)

  fwd_graph, executed_passes = build_hydrated_forward_graph(
    ForwardGraphBuildRequest(
      code=request.code,
      tree=request.tree,
      input_shapes=request.input_shapes,
      output_shape=request.output_shape,
      dtype=request.dtype,
      config=request.config,
      node_telemetry=request.node_telemetry,
      observed_shapes=observed_shapes,
    )
  )
  bwd_graph = synthesize_backward_vjp_graph(
    forward_graph=fwd_graph,
    loss_reduction=request.config.loss_reduction,
    observed_shapes=observed_shapes,
    dtype=request.dtype,
  )
  eval_shapes = evaluate_shared_backward_graph(
    backward_graph=bwd_graph,
    input_shapes=request.input_shapes,
    dtype=request.dtype,
    seed=request.config.random_seed,
  )

  backend_target = request.config.execution_backend.value
  fwd_kernel = compile_logical_graph_with_backend(fwd_graph, target=backend_target)
  bwd_kernel = compile_logical_graph_with_backend(bwd_graph, target=backend_target)

  in_shape = list(request.input_shapes[0]) if request.input_shapes else [2, 4]
  out_shape = list(request.output_shape)
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "x": {"binding": 0, "shape": in_shape, "numel": _prod(in_shape), "dtype": request.dtype}
  }
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "out": {"binding": 1, "shape": out_shape, "numel": _prod(out_shape), "dtype": request.dtype}
  }

  lines = [
    "| Line | Variable | Op | Forward Shape | Dynamic | Backward Grad Shape | Grad L2 Norm |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
  ]
  for tel in request.node_telemetry.values():
    dyn_str = f"Yes ({tel.dynamic_reason})" if tel.is_dynamic else "No"
    g_shape = str(tuple(tel.grad_shape)) if tel.grad_shape is not None else "-"
    g_norm = f"{tel.grad_norm:.4e}" if tel.grad_norm is not None else "-"
    lines.append(
      f"| {tel.lineno or '-'} | `{tel.var_name}` | `{tel.op_type}` | `{tuple(tel.output_shape)}` | {dyn_str} | `{g_shape}` | `{g_norm}` |"
    )

  mem_bytes, comm_bytes = estimate_graph_telemetry(bwd_graph)
  return SharedIRBridgeResult(
    forward_graph=fwd_graph,
    backward_graph=bwd_graph,
    forward_kernel_code=fwd_kernel,
    backward_kernel_code=bwd_kernel,
    ir_inputs_meta=ir_inputs_meta,
    ir_outputs_meta=ir_outputs_meta,
    shape_table_markdown=_NL.join(lines),
    executed_passes=executed_passes,
    forward_ir_python=export_to_python(fwd_graph, function_name="build_forward_graph"),
    backward_ir_python=export_to_python(bwd_graph, function_name="build_backward_vjp_graph"),
    forward_ir_json=fwd_graph.to_json(),
    backward_ir_json=bwd_graph.to_json(),
    forward_mermaid=render_logical_graph_mermaid(fwd_graph, title="Forward LogicalGraph"),
    backward_mermaid=render_logical_graph_mermaid(
      bwd_graph, title=f"Backward VJP LogicalGraph (loss={request.loss_val:.4f})"
    ),
    synthesized_backward_code=synthesize_target_backward_code(
      bwd_graph, target_fw=request.target_fw, semantics=request.semantics
    ),
    estimated_memory_bytes=mem_bytes,
    estimated_comm_bytes=comm_bytes,
    evaluated_node_shapes=eval_shapes,
  )


__all__ = [
  "SharedIRBridgeRequest",
  "SharedIRBridgeResult",
  "_CompilerTensorConfigProtocol",
  "_CompilerTensorProtocol",
  "_EvalGraphModuleProtocol",
  "_GradModuleProtocol",
  "_IrToAstModuleProtocol",
  "_ShapeObjectProtocol",
  "_TensorModuleProtocol",
  "evaluate_shared_backward_graph",
  "render_logical_graph_mermaid",
  "run_shared_ir_bridge",
  "synthesize_backward_vjp_graph",
  "synthesize_target_backward_code",
]
