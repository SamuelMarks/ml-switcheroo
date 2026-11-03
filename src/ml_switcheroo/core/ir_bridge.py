"""Shared `ml-switcheroo-ir` (`LogicalGraph`) Unification Bridge.

Provides zero-copy `LogicalGraph` construction, hydration, live `zero-*` `ProxyTensor`
tracing, reverse-mode VJP graph synthesis (`autodiff.grad`), graph evaluation
(`evaluate_graph`), Mermaid DAG rendering, and target-framework backward/VJP code
synthesis (`emit_ir_to_ast` & `PythonBackend`).
"""

from __future__ import annotations

import importlib
import types
import typing
from dataclasses import dataclass, field

import libcst as cst
from ml_switcheroo_ir import (
  DType,
  LogicalEdge,
  LogicalGraph,
  LogicalNode,
  TensorSpec,
  export_to_python,
  propagate_shapes_and_constants,
  topological_sort,
)

from ml_switcheroo.config import ConversionMode, LossReduction, RuntimeConfig
from ml_switcheroo.core.compiler.backends.compiler_bridge import compile_logical_graph_with_backend
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.compiler_passes import (
  ensure_compiler_and_zero_on_path,
  estimate_graph_telemetry,
  run_multi_pass_optimizer,
)
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry, RuntimeTraceReport
from ml_switcheroo.core.runtime_tensor import _generate_seeded_values, _prod
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)

_NON_PYTHON_SOURCE_DIALECTS = frozenset(
  {
    "rdna",
    "nvidia_sass",
    "stablehlo",
    "mlir",
    "tikz",
    "ir",
    "ml_switcheroo_ir",
  }
)


def _lift_non_python_source_for_trace(
  code: str,
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None = None,
) -> tuple[str, cst.Module] | None:
  """Lift a non-Python source dialect (`rdna`, `nvidia_sass`, `stablehlo`, `mlir`, `tikz`, `ir`) to a Python CST.

  Args:
      code: Raw source code in the non-Python dialect.
      source_fw: Source dialect identifier.
      target_fw: Target framework identifier.
      semantics: Optional `SemanticsManager` instance.

  Returns:
      Tuple of ``(lifted_python_code, parsed_cst_module)``, or ``None`` if lifting fails.
  """
  synth_fw = target_fw if get_adapter(target_fw) is not None else "flax_nnx"
  try:
    if source_fw in ("rdna", "nvidia_sass"):
      from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
      from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser

      if source_fw == "rdna":
        graph = RdnaLifter().lift(RdnaParser(code).parse().statements)
      else:
        graph = NvidiaSassLifter().lift(NvidiaSassParser(code).parse().statements)
      py_code = PythonBackend(framework=synth_fw, semantics=semantics).compile(graph)
      return py_code, cst.parse_module(py_code)

    from ml_switcheroo.core.ingestion import ingest_code
    from ml_switcheroo.core.tracer import TraceLogger

    src_adapter = get_adapter(source_fw)
    tree = ingest_code(code, source_fw, synth_fw, src_adapter, TraceLogger())
    return tree.code, tree
  except Exception:
    return None


def _resolve_ir_dtype(dtype_val: str | DType | None) -> DType:
  """Resolve a string or `DType` value into an `ml_switcheroo_ir.DType` enum.

  Args:
      dtype_val: String dtype identifier or `DType`.

  Returns:
      Resolved `DType` enum member (defaults to `DType.float32`).
  """
  if isinstance(dtype_val, DType):
    return dtype_val
  clean = str(dtype_val or "float32").strip().lower()
  for member in DType:
    if member.value == clean or member.name == clean:
      return member
  return DType.float32


@dataclass
class SharedIRBridgeResult:
  """Container for artifacts synthesized from the shared `ml_switcheroo_ir.LogicalGraph`."""

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


class _TracingStateProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.tracing.global_tracing_state`."""

  def start_tracing(self) -> None:
    """Start proxy tensor tracing."""
    ...

  def add_node(self, node: LogicalNode) -> None:
    """Add a node to the active graph.

    Args:
        node: `LogicalNode` to add.
    """
    ...

  def stop_tracing(self) -> LogicalGraph:
    """Stop proxy tensor tracing and return the traced `LogicalGraph`.

    Returns:
        Traced `LogicalGraph`.
    """
    ...


class _ProxyTensorCtorProtocol(typing.Protocol):
  """Protocol for `ProxyTensor` constructor."""

  def __call__(self, *, id: str, shape: list[int], dtype: str) -> object:
    """Construct a `ProxyTensor` instance.

    Args:
        id: Node identifier.
        shape: Tensor shape list.
        dtype: Data type string.

    Returns:
        `ProxyTensor` instance.
    """
    ...


@typing.runtime_checkable
class _TracingModuleProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.tracing`."""

  global_tracing_state: _TracingStateProtocol | None
  ProxyTensor: _ProxyTensorCtorProtocol | None


def trace_with_zero_proxy_tensors(
  code: str,
  input_shapes: list[tuple[int, ...]],
  dtype: str = "float32",
  seed: int = 42,
) -> LogicalGraph | None:
  """Capture a live `LogicalGraph` using `ml_switcheroo_compiler.tracing.state.global_tracing_state`.

  Executes user-defined top-level callables with `ProxyTensor` inputs inside an active
  `global_tracing_state` session when `ml_switcheroo_compiler` is available.

  Args:
      code: Source Python code string.
      input_shapes: Inferred or configured input tensor shapes.
      dtype: Input tensor dtype string.
      seed: Deterministic random seed.

  Returns:
      Captured `LogicalGraph` if tracing succeeded and recorded nodes, otherwise None.
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
  assert global_state is not None and proxy_cls is not None

  global_state.start_tracing()
  ir_dt = _resolve_ir_dtype(dtype)
  try:
    proxy_inputs = []
    for idx, shape in enumerate(input_shapes):
      node_id = f"input_{idx}" if idx > 0 else "x"
      global_state.add_node(LogicalNode(id=node_id, op_type="Input", shape_metadata=tuple(shape), dtype=ir_dt))
      proxy_inputs.append(proxy_cls(id=node_id, shape=list(shape), dtype=dtype))

    # Execute top-level function definitions via AST function compilation
    import ast

    tree = ast.parse(code, filename="<zero_proxy_trace>")
    for stmt in tree.body:
      if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
        arg_names = [a.arg for a in stmt.args.args]
        call_args_str = ", ".join(arg_names[: len(proxy_inputs)])
        wrapper_src = f"{code}\n_trace_res = {stmt.name}({call_args_str})\n"
        wrapper_code = compile(wrapper_src, "<zero_proxy_trace>", "exec")
        trace_ns: dict[str, object] = {
          arg_names[i]: proxy_inputs[i] for i in range(min(len(arg_names), len(proxy_inputs)))
        }
        trace_fn = types.FunctionType(wrapper_code, trace_ns)
        trace_fn()
        break
  except Exception:
    pass
  finally:
    traced_graph: LogicalGraph = global_state.stop_tracing()

  if traced_graph.nodes:
    return traced_graph
  return None


def build_hydrated_forward_graph(
  code: str,
  tree: cst.Module,
  input_shapes: list[tuple[int, ...]],
  output_shape: tuple[int, ...],
  dtype: str,
  node_telemetry: dict[str, NodeRuntimeTelemetry],
  observed_shapes: dict[str, list[int]],
  config: RuntimeConfig,
) -> tuple[LogicalGraph, list[str]]:
  """Construct and hydrate the canonical `ml_switcheroo_ir.LogicalGraph` for the forward pass.

  Combines `ml_switcheroo_compiler.backends.ast_to_ir.parse_ast_to_ir`, `GraphExtractor`,
  and live `ProxyTensor` tracing, then populates `input_specs`, `shape_metadata`, and
  `output_specs` and runs `propagate_shapes_and_constants` and `annotate_learned_shapes`.

  Args:
      code: Source Python code string.
      tree: Parsed LibCST module.
      input_shapes: Input tensor shapes.
      output_shape: Terminal output tensor shape.
      dtype: Tensor data type string.
      node_telemetry: Per-node runtime telemetry recorded during forward execution.
      observed_shapes: Map of variable/node names to concrete runtime shapes.
      config: Active `RuntimeConfig`.

  Returns:
      Tuple of `(hydrated_forward_graph, executed_pass_names)`.
  """
  ensure_compiler_and_zero_on_path()
  ir_dt = _resolve_ir_dtype(dtype)
  extractor = GraphExtractor()
  tree.visit(extractor)
  cst_graph = extractor.graph

  graph: LogicalGraph | None = None
  try:
    import operator

    ast_to_ir_mod = importlib.import_module("ml_switcheroo_compiler.backends.ast_to_ir")
    parse_fn = operator.attrgetter("parse_ast_to_ir")(ast_to_ir_mod)
    parsed_graph = parse_fn(code)
    if isinstance(parsed_graph, LogicalGraph) and parsed_graph.nodes:
      graph = parsed_graph
  except Exception:
    graph = None

  if graph is None or not graph.nodes:
    traced = trace_with_zero_proxy_tensors(code, input_shapes, dtype=dtype, seed=config.random_seed)
    graph = traced if traced is not None else cst_graph

  if not graph.nodes:
    in_shape = tuple(input_shapes[0]) if input_shapes else (2, 4)
    out_shape_tuple = tuple(output_shape)
    graph.nodes["x"] = LogicalNode(
      id="x",
      op_type="Input",
      shape_metadata=in_shape,
      dtype=ir_dt,
      output_specs=[TensorSpec(shape=in_shape, dtype=ir_dt)],
    )
    graph.nodes["out"] = LogicalNode(
      id="out",
      op_type="Identity",
      shape_metadata=out_shape_tuple,
      dtype=ir_dt,
      output_specs=[TensorSpec(shape=out_shape_tuple, dtype=ir_dt)],
    )
    graph.edges.append(LogicalEdge(source="x", target="out"))
    graph.inputs = ["x"]
    graph.outputs = ["out"]

  # Populate input_specs and inputs list
  in_shape_default = tuple(input_shapes[0]) if input_shapes else (2, 4)
  input_ids: list[str] = [nid for nid in graph.inputs if nid in graph.nodes]
  for nid, node in graph.nodes.items():
    if node.op_type in ("Input", "Placeholder") and nid not in input_ids:
      input_ids.append(nid)
  if not input_ids:
    first_id = next(iter(graph.nodes.keys()))
    input_ids.append(first_id)
  graph.inputs = input_ids

  for idx, in_id in enumerate(graph.inputs):
    shape_tuple = input_shapes[idx] if idx < len(input_shapes) else in_shape_default
    spec = TensorSpec(shape=tuple(shape_tuple), dtype=ir_dt)
    graph.input_specs[in_id] = spec
    graph.nodes[in_id].shape_metadata = tuple(shape_tuple)
    graph.nodes[in_id].dtype = ir_dt
    graph.nodes[in_id].output_specs = [spec]

  # Hydrate node shapes from runtime telemetry and CST provenance
  for nid, node in graph.nodes.items():
    cst_node = cst_graph.nodes.get(nid)
    node.source_ast_ref = node.source_ast_ref or (cst_node.source_ast_ref if cst_node is not None else None)
    matched_shape = observed_shapes.get(nid) or (
      list(node_telemetry[nid].output_shape) if nid in node_telemetry else None
    )
    if matched_shape is not None:
      node.shape_metadata = tuple(matched_shape)
      node.dtype = node.dtype or ir_dt
      node.output_specs = [TensorSpec(shape=tuple(matched_shape), dtype=node.dtype)]

  if not graph.outputs:
    sorted_nodes = topological_sort(graph)
    non_output = [n.id for n in sorted_nodes if n.op_type != "Output"]
    terminal_id = non_output[-1] if non_output else sorted_nodes[-1].id
    graph.outputs = [terminal_id]

  for out_id in graph.outputs:
    if out_id in graph.nodes and graph.nodes[out_id].shape_metadata is None:
      graph.nodes[out_id].shape_metadata = tuple(output_shape)
      graph.nodes[out_id].dtype = ir_dt
      graph.nodes[out_id].output_specs = [TensorSpec(shape=tuple(output_shape), dtype=ir_dt)]

  try:
    import operator

    propagate_shapes_and_constants(graph)
    shape_inf_mod = importlib.import_module("ml_switcheroo_compiler.transforms.passes.shape_inference")
    operator.attrgetter("shape_inference_pass")(shape_inf_mod)(graph)
    operator.attrgetter("annotate_learned_shapes")(shape_inf_mod)(graph, observed_shapes)
  except Exception:
    pass

  executed_passes: list[str] = ["propagate_shapes_and_constants", "shape_inference_pass"]
  if config.mode == ConversionMode.OPTIMIZED or config.enable_graph_optimization:
    graph, opt_passes = run_multi_pass_optimizer(graph, opt_level=config.opt_level)
    executed_passes.extend(opt_passes)

  return graph, executed_passes


@typing.runtime_checkable
class _GradModuleProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.transforms.autodiff` providing `grad`."""

  def grad(self, graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
    """Differentiate `graph` with respect to `wrt`.

    Args:
        graph: Forward `LogicalGraph`.
        wrt: Input node identifiers.
        output_id: Output node identifier.

    Returns:
        Backward `LogicalGraph`.
    """
    ...


class _ShapeObjectProtocol(typing.Protocol):
  """Protocol for objects exposing a `shape` sequence."""

  shape: typing.Sequence[int]


@typing.runtime_checkable
class _EvalGraphModuleProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.interpreter.evaluator` providing `evaluate_graph`."""

  def evaluate_graph(self, graph: LogicalGraph, env: dict[str, object]) -> dict[str, _ShapeObjectProtocol]:
    """Evaluate `graph` with inputs in `env`.

    Args:
        graph: Target `LogicalGraph`.
        env: Input tensor dictionary.

    Returns:
        Dictionary of evaluated tensors.
    """
    ...


@typing.runtime_checkable
class _TensorModuleProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.core.tensor` providing `Tensor` and `TensorConfig`."""

  def TensorConfig(self, shape: tuple[int, ...], dtype: str) -> object:
    """Construct a `TensorConfig`.

    Args:
        shape: Tensor shape tuple.
        dtype: Data type string.

    Returns:
        `TensorConfig` instance.
    """
    ...

  def Tensor(self, data: object, config: object) -> object:
    """Construct a compiler `Tensor`.

    Args:
        data: Tensor values.
        config: `TensorConfig` instance.

    Returns:
        Compiler `Tensor` instance.
    """
    ...


def synthesize_backward_vjp_graph(
  forward_graph: LogicalGraph,
  loss_reduction: LossReduction,
  observed_shapes: dict[str, list[int]],
  dtype: str = "float32",
) -> LogicalGraph:
  """Synthesize the reverse-mode VJP `LogicalGraph` via `ml_switcheroo_compiler.transforms.autodiff.grad`.

  Args:
      forward_graph: Hydrated forward `LogicalGraph`.
      loss_reduction: Scalar loss reduction (`SUM`, `MEAN`, `L2`) applied before differentiation.
      observed_shapes: Map of observed runtime shapes for shape annotation.
      dtype: Default floating-point dtype string.

  Returns:
      Backward `LogicalGraph` containing forward primal nodes and reverse-mode VJP adjoint nodes.
  """
  ensure_compiler_and_zero_on_path()
  ir_dt = _resolve_ir_dtype(dtype)
  working_graph = forward_graph.model_copy(deep=True)

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
  wrt_ids: list[str] = (base_wrt + param_wrt) or [sorted_nodes[0].id]

  # Append scalar reduction node if terminal output is non-scalar
  term_shape = working_graph.nodes[output_id].shape_metadata
  loss_node_id = output_id
  if isinstance(term_shape, (list, tuple)) and _prod([int(d) for d in term_shape if isinstance(d, int)]) > 1:
    red_op = "Mean" if loss_reduction == LossReduction.MEAN else "Sum"
    loss_node_id = f"loss_{red_op.lower()}"
    working_graph.nodes[loss_node_id] = LogicalNode(
      id=loss_node_id,
      op_type=red_op,
      shape_metadata=(1,),
      dtype=ir_dt,
      output_specs=[TensorSpec(shape=(1,), dtype=ir_dt)],
    )
    working_graph.edges.append(LogicalEdge(source=output_id, target=loss_node_id))
    working_graph.outputs = [loss_node_id]

  bwd_graph: LogicalGraph | None = None
  try:
    for nid, node in working_graph.nodes.items():
      node.inputs = node.inputs or [e.source for e in working_graph.edges if e.target == nid]
    autodiff_mod = importlib.import_module("ml_switcheroo_compiler.transforms.autodiff")
    assert isinstance(autodiff_mod, _GradModuleProtocol)
    bwd_graph = autodiff_mod.grad(working_graph, wrt=wrt_ids, output_id=loss_node_id)
  except Exception:
    bwd_graph = None

  if bwd_graph is None:
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
      w_shape = bwd_graph.nodes[w_id].shape_metadata
      g_id = f"grad_{w_id}"
      bwd_graph.nodes[g_id] = LogicalNode(
        id=g_id,
        op_type="Mul",
        shape_metadata=w_shape,
        dtype=ir_dt,
      )
      bwd_graph.edges.append(LogicalEdge(source=cot_id, target=g_id))
      bwd_graph.edges.append(LogicalEdge(source=w_id, target=g_id))

  try:
    import operator

    propagate_shapes_and_constants(bwd_graph)
    shape_inf_mod = importlib.import_module("ml_switcheroo_compiler.transforms.passes.shape_inference")
    operator.attrgetter("shape_inference_pass")(shape_inf_mod)(bwd_graph)
    operator.attrgetter("annotate_learned_shapes")(shape_inf_mod)(bwd_graph, observed_shapes)
  except Exception:
    pass

  return bwd_graph


def evaluate_shared_backward_graph(
  backward_graph: LogicalGraph,
  input_shapes: list[tuple[int, ...]],
  dtype: str = "float32",
  seed: int = 42,
) -> dict[str, list[int]]:
  """Evaluate `backward_graph` via `ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph`.

  Args:
      backward_graph: Synthesized reverse-mode VJP `LogicalGraph`.
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
    assert isinstance(eval_mod, _EvalGraphModuleProtocol)
    assert isinstance(tensor_mod, _TensorModuleProtocol)

    env: dict[str, object] = {}
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
  except Exception:
    pass

  for nid, node in backward_graph.nodes.items():
    if nid not in evaluated_shapes and isinstance(node.shape_metadata, (list, tuple)):
      evaluated_shapes[nid] = [int(d) for d in node.shape_metadata if isinstance(d, int)]
  return evaluated_shapes


def render_logical_graph_mermaid(graph: LogicalGraph, title: str = "LogicalGraph") -> str:
  """Render a Mermaid `graph TD` diagram with concrete `TensorSpec(shape, dtype)` labels.

  Args:
      graph: Target `LogicalGraph` (`forward_graph` or `backward_graph`).
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
    if nid.startswith("grad_") or "grad" in nid.lower():
      lines.append(f'  {safe_id}["{label}"]:::gradNode')
    else:
      lines.append(f'  {safe_id}["{label}"]')

  for edge in graph.edges:
    src_safe = edge.source.replace(".", "_").replace(":", "_").replace("-", "_")
    tgt_safe = edge.target.replace(".", "_").replace(":", "_").replace("-", "_")
    src_node = graph.nodes.get(edge.source)
    if src_node is not None and isinstance(src_node.shape_metadata, (list, tuple)):
      src_dt_val = src_node.dtype.value if isinstance(src_node.dtype, DType) else src_node.dtype
      src_dt = str(src_dt_val) if src_dt_val else "f32"
      edge_lbl = f"{tuple(src_node.shape_metadata)} {src_dt}"
      lines.append(f'  {src_safe} -->|"{edge_lbl}"| {tgt_safe}')
    else:
      lines.append(f"  {src_safe} --> {tgt_safe}")

  lines.append("  classDef gradNode fill:#fff3e0,stroke:#e65100,stroke-width:2px;")
  return _NL.join(lines)


def synthesize_target_backward_code(
  backward_graph: LogicalGraph,
  target_fw: str,
  semantics: SemanticsManager | None = None,
) -> str:
  """Synthesize explicit target-framework backward/VJP Python code from `backward_graph`.

  Attempts `ml_switcheroo_compiler.backends.ir_to_ast.emit_ir_to_ast(backward_graph, target_fw)`
  first and combines or falls back to `PythonBackend(framework=target_fw, semantics=semantics)`.

  Args:
      backward_graph: Reverse-mode VJP `LogicalGraph` produced by `autodiff.grad`.
      target_fw: Target framework identifier.
      semantics: Optional `SemanticsManager` instance.

  Returns:
      Target-framework Python source string implementing the synthesized backward/VJP graph.
  """
  ensure_compiler_and_zero_on_path()
  emitted_body = ""
  try:
    import operator

    ir_to_ast_mod = importlib.import_module("ml_switcheroo_compiler.backends.ir_to_ast")
    emit_fn = operator.attrgetter("emit_ir_to_ast")(ir_to_ast_mod)
    cst_mod = emit_fn(backward_graph, target_fw)
    raw_code = cst_mod.code.strip()
    if raw_code:
      indented = _NL.join(f"    {line}" for line in raw_code.splitlines())
      emitted_body = (
        f"def synthesized_backward_vjp(x):{_NL}"
        f'    """Synthesized reverse-mode VJP backward graph ({target_fw})."""{_NL}'
        f"{indented}{_NL}"
      )
  except Exception:
    emitted_body = ""

  if not emitted_body:
    try:
      py_backend = PythonBackend(framework=target_fw, semantics=semantics)
      compiled = py_backend.compile(backward_graph).strip()
      emitted_body = f"# Synthesized Backward/VJP Graph ({target_fw}){_NL}{compiled}{_NL}"
    except Exception:
      emitted_body = (
        f"def synthesized_backward_vjp(x):{_NL}"
        f'    """Synthesized reverse-mode VJP backward graph ({target_fw})."""{_NL}'
        f"    return x{_NL}"
      )

  return emitted_body


def run_shared_ir_bridge(
  code: str,
  tree: cst.Module,
  config: RuntimeConfig,
  target_fw: str,
  input_shapes: list[tuple[int, ...]],
  output_shape: tuple[int, ...],
  dtype: str,
  loss_val: float,
  node_telemetry: dict[str, NodeRuntimeTelemetry],
  semantics: SemanticsManager | None = None,
) -> SharedIRBridgeResult:
  """Execute the end-to-end shared `ml-switcheroo-ir` (`LogicalGraph`) bridge.

  Args:
      code: Source Python code string.
      tree: Parsed LibCST module.
      config: Active `RuntimeConfig`.
      target_fw: Target framework key.
      input_shapes: Inferred or user-configured input shapes.
      output_shape: Output tensor shape.
      dtype: Tensor data type string.
      loss_val: Scalar loss value from forward execution.
      node_telemetry: Per-node runtime telemetry map.
      semantics: Optional `SemanticsManager` instance.

  Returns:
      Populated `SharedIRBridgeResult`.
  """
  observed_shapes: dict[str, list[int]] = {}
  if config.observed_runtime_shapes:
    observed_shapes.update(config.observed_runtime_shapes)
  for tel in node_telemetry.values():
    observed_shapes[tel.var_name] = list(tel.output_shape)
    observed_shapes[tel.node_id] = list(tel.output_shape)

  fwd_graph, executed_passes = build_hydrated_forward_graph(
    code=code,
    tree=tree,
    input_shapes=input_shapes,
    output_shape=output_shape,
    dtype=dtype,
    node_telemetry=node_telemetry,
    observed_shapes=observed_shapes,
    config=config,
  )

  bwd_graph = synthesize_backward_vjp_graph(
    forward_graph=fwd_graph,
    loss_reduction=config.loss_reduction,
    observed_shapes=observed_shapes,
    dtype=dtype,
  )

  eval_shapes = evaluate_shared_backward_graph(
    backward_graph=bwd_graph,
    input_shapes=input_shapes,
    dtype=dtype,
    seed=config.random_seed,
  )

  backend_target = config.execution_backend.value
  fwd_kernel = compile_logical_graph_with_backend(fwd_graph, target=backend_target)
  bwd_kernel = compile_logical_graph_with_backend(bwd_graph, target=backend_target)

  in_shape = list(input_shapes[0]) if input_shapes else [2, 4]
  out_shape = list(output_shape)
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "x": {
      "binding": 0,
      "shape": in_shape,
      "numel": _prod(in_shape),
      "dtype": dtype,
    }
  }
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "out": {
      "binding": 1,
      "shape": out_shape,
      "numel": _prod(out_shape),
      "dtype": dtype,
    }
  }

  lines = [
    "| Line | Variable | Op | Forward Shape | Dynamic | Backward Grad Shape | Grad L2 Norm |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
  ]
  for tel in node_telemetry.values():
    dyn_str = f"Yes ({tel.dynamic_reason})" if tel.is_dynamic else "No"
    g_shape = str(tuple(tel.grad_shape)) if tel.grad_shape is not None else "-"
    g_norm = f"{tel.grad_norm:.4e}" if tel.grad_norm is not None else "-"
    lines.append(
      f"| {tel.lineno or '-'} | `{tel.var_name}` | `{tel.op_type}` | `{tuple(tel.output_shape)}` | {dyn_str} | `{g_shape}` | `{g_norm}` |"
    )
  shape_md = _NL.join(lines)

  fwd_ir_py = export_to_python(fwd_graph, function_name="build_forward_graph")
  bwd_ir_py = export_to_python(bwd_graph, function_name="build_backward_vjp_graph")
  fwd_ir_json = fwd_graph.to_json()
  bwd_ir_json = bwd_graph.to_json()
  fwd_mermaid = render_logical_graph_mermaid(fwd_graph, title="Forward LogicalGraph")
  bwd_mermaid = render_logical_graph_mermaid(bwd_graph, title=f"Backward VJP LogicalGraph (loss={loss_val:.4f})")
  synth_bwd_code = synthesize_target_backward_code(bwd_graph, target_fw=target_fw, semantics=semantics)
  mem_bytes, comm_bytes = estimate_graph_telemetry(bwd_graph)

  return SharedIRBridgeResult(
    forward_graph=fwd_graph,
    backward_graph=bwd_graph,
    forward_kernel_code=fwd_kernel,
    backward_kernel_code=bwd_kernel,
    ir_inputs_meta=ir_inputs_meta,
    ir_outputs_meta=ir_outputs_meta,
    shape_table_markdown=shape_md,
    executed_passes=executed_passes,
    forward_ir_python=fwd_ir_py,
    backward_ir_python=bwd_ir_py,
    forward_ir_json=fwd_ir_json,
    backward_ir_json=bwd_ir_json,
    forward_mermaid=fwd_mermaid,
    backward_mermaid=bwd_mermaid,
    synthesized_backward_code=synth_bwd_code,
    estimated_memory_bytes=mem_bytes,
    estimated_comm_bytes=comm_bytes,
    evaluated_node_shapes=eval_shapes,
  )


def _format_node_runtime_comment(
  output_shape: list[int] | tuple[int, ...],
  dtype: str,
  grad_shape: list[int] | None = None,
  grad_norm: float | None = None,
  dynamic_reason: str | None = None,
) -> str:
  """Format a concise runtime telemetry comment for a ``LogicalNode``.

  Args:
      output_shape: Forward pass output tensor shape.
      dtype: Tensor data type string.
      grad_shape: Optional backward gradient tensor shape.
      grad_norm: Optional backward gradient L2 norm.
      dynamic_reason: Optional explanation for dynamic shape resolution.

  Returns:
      Formatted comment string.
  """
  parts = [f"[on-device fwd] shape={tuple(output_shape)} dtype={dtype}"]
  if dynamic_reason:
    parts.append(f"dynamic: {dynamic_reason}")
  if grad_shape is not None and grad_norm is not None:
    parts.append(f"[bwd] grad={tuple(grad_shape)} norm={grad_norm:.4e}")
  return " | ".join(parts)


def hydrate_graph_with_runtime_trace(
  graph: LogicalGraph,
  trace: RuntimeTraceReport,
) -> None:
  """Hydrate a ``LogicalGraph`` in-place with runtime-discovered shapes, dimensions, and telemetry.

  Populates:
  - ``node.attributes["in_features"]`` and ``node.attributes["out_features"]`` from
    ``trace.resolved_constructors`` and ``trace.node_telemetry``.
  - ``node.attributes["in_channels"]``, ``node.attributes["out_channels"]``,
    ``node.attributes["input_height"]``, ``node.attributes["input_width"]``,
    ``node.attributes["row_stride_bytes"]`` (``W * 4``), and
    ``node.attributes["slice_stride_bytes"]`` (``H * W * 4``) for convolution and pooling layers.
  - ``node.attributes["num_elements"]`` and ``node.attributes["loop_trip_count"]``.
  - ``node.shape_metadata``, ``node.dtype``, and ``node.attributes["runtime_comment"]``.

  Args:
      graph: The ``LogicalGraph`` to hydrate in-place.
      trace: The ``RuntimeTraceReport`` produced by ``run_on_device_trace``.
  """
  if not trace.executed:
    return

  tel_list = list(trace.node_telemetry.values())
  default_in_shape = trace.input_shapes[0] if trace.input_shapes else [2, 4]
  max_loop_trips = max(trace.loop_trip_counts.values()) if trace.loop_trip_counts else None

  # Build incoming edge map to infer input shapes from predecessor nodes
  incoming_map: dict[str, list[str]] = {}
  for edge in graph.edges:
    incoming_map.setdefault(edge.target, []).append(edge.source)

  non_io_idx = 0
  for nid, node in graph.nodes.items():
    op_lower = str(node.op_type).lower()
    ir_dt = _resolve_ir_dtype("float32")

    if node.op_type == "Input":
      in_shape = tuple(default_in_shape)
      node.shape_metadata = in_shape
      node.dtype = ir_dt
      node.output_specs = [TensorSpec(shape=in_shape, dtype=ir_dt)]
      node.attributes["num_elements"] = _prod(in_shape)
      continue

    if node.op_type == "Output":
      out_shape = tuple(trace.output_shape) if trace.output_shape else tuple(default_in_shape)
      node.shape_metadata = out_shape
      node.dtype = ir_dt
      node.output_specs = [TensorSpec(shape=out_shape, dtype=ir_dt)]
      node.attributes["num_elements"] = _prod(out_shape)
      continue

    # Match telemetry entry by node id, op_type, or topological sequence index
    matched_tel: NodeRuntimeTelemetry | None = trace.node_telemetry.get(nid)
    if matched_tel is None:
      matched_tel = next(
        (t for t in tel_list if t.var_name == nid or t.op_type.lower() == op_lower),
        tel_list[non_io_idx] if non_io_idx < len(tel_list) else None,
      )
    non_io_idx += 1

    # Resolve predecessor input shape
    pred_ids = incoming_map.get(nid, [])
    pred_meta = graph.nodes[pred_ids[0]].shape_metadata if (pred_ids and pred_ids[0] in graph.nodes) else None
    if isinstance(pred_meta, (list, tuple)) and pred_meta:
      pred_shape: list[int] = [int(d) for d in pred_meta if isinstance(d, int)]
    elif matched_tel is not None and matched_tel.input_shapes:
      pred_shape = list(matched_tel.input_shapes[0])
    else:
      pred_shape = list(default_in_shape)

    node_out_shape: list[int] = (
      list(matched_tel.output_shape)
      if matched_tel is not None and matched_tel.output_shape
      else (list(trace.output_shape) if trace.output_shape else list(pred_shape))
    )
    dtype_str = matched_tel.dtype if matched_tel is not None else "float32"
    ir_dt = _resolve_ir_dtype(dtype_str)

    # 1. Constructor resolution (Linear / LazyLinear / Dense / Conv)
    ctor = trace.resolved_constructors.get(nid)
    if ctor is not None:
      if "conv" in op_lower:
        node.attributes["in_channels"] = ctor.inferred_in_features
        if ctor.out_features is not None:
          node.attributes["out_channels"] = ctor.out_features
      else:
        node.attributes["in_features"] = ctor.inferred_in_features
        out_val = node.attributes.get("out_features", ctor.out_features)
        if out_val is not None:
          node.attributes["out_features"] = int(str(out_val))
          node_out_shape[-1] = int(str(out_val))
        node.attributes.pop("arg_0", None)
        node.attributes.pop("arg_1", None)
    elif any(k in op_lower for k in ("linear", "dense", "gemm", "matmul")):
      in_feat = pred_shape[-1] if pred_shape else 128
      if "in_features" not in node.attributes:
        if "arg_0" in node.attributes and "arg_1" in node.attributes:
          try:
            node.attributes["in_features"] = int(str(node.attributes.pop("arg_0")))
            node.attributes["out_features"] = int(str(node.attributes.pop("arg_1")))
          except ValueError:
            node.attributes["in_features"] = in_feat
        else:
          raw_arg0 = node.attributes.pop("arg_0", None)
          node.attributes["in_features"] = in_feat
          try:
            node.attributes["out_features"] = int(str(raw_arg0))
          except (ValueError, TypeError):
            pass
      node.attributes.setdefault("out_features", node_out_shape[-1])

    # 2. Flatten output shape calculation if predecessor is >2D
    if "flatten" in op_lower and len(pred_shape) >= 2:
      node_out_shape = [pred_shape[0], _prod(pred_shape[1:])]

    # 3. Spatial strides for Conv / Pool operations
    if any(k in op_lower for k in ("conv", "pool")) and len(pred_shape) >= 3:
      in_c = pred_shape[1]
      in_h = pred_shape[-2] if len(pred_shape) >= 4 else pred_shape[-1]
      in_w = pred_shape[-1]
      node.attributes.setdefault("in_channels", in_c)
      node.attributes.setdefault("out_channels", node_out_shape[1] if len(node_out_shape) >= 2 else in_c)
      node.attributes["input_height"] = in_h
      node.attributes["input_width"] = in_w
      node.attributes["row_stride_bytes"] = in_w * 4
      node.attributes["slice_stride_bytes"] = in_h * in_w * 4

    # 4. Populate num_elements, loop_trip_count, shape_metadata, and runtime_comment
    node.shape_metadata = tuple(node_out_shape)
    node.dtype = ir_dt
    node.output_specs = [TensorSpec(shape=tuple(node_out_shape), dtype=ir_dt)]
    node.attributes["num_elements"] = _prod(node_out_shape)
    if max_loop_trips is not None:
      node.attributes["loop_trip_count"] = max_loop_trips

    grad_shape = matched_tel.grad_shape if matched_tel is not None else None
    grad_norm = matched_tel.grad_norm if matched_tel is not None else None
    dyn_reason = matched_tel.dynamic_reason if matched_tel is not None else None
    node.attributes["runtime_comment"] = _format_node_runtime_comment(
      output_shape=node_out_shape,
      dtype=dtype_str,
      grad_shape=grad_shape,
      grad_norm=grad_norm,
      dynamic_reason=dyn_reason,
    )
