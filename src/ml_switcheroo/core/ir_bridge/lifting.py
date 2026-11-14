"""Forward ``LogicalGraph`` Lifting and Dialect Ingestion for ``ir_bridge``.

Provides ``_lift_non_python_source_for_trace`` and ``build_hydrated_forward_graph``.
"""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass, field

import libcst as cst
from ml_switcheroo_ir import (
  DType,
  LogicalGraph,
  LogicalNode,
  TensorSpec,
  topological_sort,
)

from ml_switcheroo.config import ConversionMode, RuntimeConfig
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.compiler_passes import (
  ensure_compiler_and_zero_on_path,
  run_multi_pass_optimizer,
)
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.ir_bridge.proxy_tracing import (
  _resolve_ir_dtype,
  _run_shape_inference_hooks,
  trace_with_zero_proxy_tensors,
)
from ml_switcheroo.core.runtime_models import NodeRuntimeTelemetry
from ml_switcheroo.exceptions import MLSwitcherooError
from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager

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


@typing.runtime_checkable
class _AstToIrModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.backends.ast_to_ir`` module."""

  def parse_ast_to_ir(self, code: str) -> LogicalGraph:
    """Parse Python source code into a ``LogicalGraph``.

    Args:
        code: Source code string.

    Returns:
        Parsed ``LogicalGraph``.
    """


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
      semantics: Optional ``SemanticsManager`` instance.

  Returns:
      Tuple of ``(lifted_python_code, parsed_cst_module)``, or ``None`` if lifting fails.
  """
  synth_fw = target_fw if get_adapter(target_fw) is not None else "flax_nnx"
  try:
    if source_fw in ("rdna", "nvidia_sass"):
      from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassLifter, NvidiaSassParser
      from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser

      graph = (
        RdnaLifter().lift(RdnaParser(code).parse().statements)
        if source_fw == "rdna"
        else NvidiaSassLifter().lift(NvidiaSassParser(code).parse().statements)
      )
      py_code = PythonBackend(framework=synth_fw, semantics=semantics).compile(graph)
      return py_code, cst.parse_module(py_code)

    from ml_switcheroo.core.ingestion import ingest_code
    from ml_switcheroo.core.tracer import TraceLogger

    src_adapter = get_adapter(source_fw)
    tree = ingest_code(code, source_fw, synth_fw, src_adapter, TraceLogger())
    return tree.code, tree
  except (
    ImportError,
    OSError,
    SyntaxError,
    cst.ParserSyntaxError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    MLSwitcherooError,
  ):
    return None


def _extract_initial_forward_graph(
  code: str,
  cst_graph: LogicalGraph,
  input_shapes: list[tuple[int, ...]],
  dtype: str,
  seed: int,
) -> LogicalGraph:
  """Parse ``code`` via ``ast_to_ir``, falling back to ``ProxyTensor`` tracing or ``cst_graph``.

  Args:
      code: Source Python code string.
      cst_graph: Graph extracted statically by ``GraphExtractor``.
      input_shapes: Input tensor shapes.
      dtype: Input tensor dtype string.
      seed: Deterministic random seed.

  Returns:
      Non-null ``LogicalGraph`` candidate.
  """
  graph: LogicalGraph | None = None
  try:
    ast_to_ir_mod = importlib.import_module("ml_switcheroo_compiler.backends.ast_to_ir")
    if not isinstance(ast_to_ir_mod, _AstToIrModuleProtocol):
      raise ImportError("ast_to_ir module missing parse_ast_to_ir")
    parsed_graph = ast_to_ir_mod.parse_ast_to_ir(code)
    if isinstance(parsed_graph, LogicalGraph) and parsed_graph.nodes:
      graph = parsed_graph
  except (ImportError, SyntaxError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, MLSwitcherooError):
    graph = None

  if graph is None or not graph.nodes:
    traced = trace_with_zero_proxy_tensors(code, input_shapes, dtype=dtype, seed=seed)
    return traced if traced is not None else cst_graph
  return graph


def _seed_identity_fallback_if_empty(
  graph: LogicalGraph,
  input_shapes: list[tuple[int, ...]],
  output_shape: tuple[int, ...],
  ir_dt: DType,
) -> None:
  """Populate minimal ``x -> Identity -> out`` nodes if ``graph.nodes`` is empty.

  Args:
      graph: Target ``LogicalGraph`` to populate in place.
      input_shapes: Input tensor shapes.
      output_shape: Terminal output tensor shape.
      ir_dt: Resolved ``DType`` enum member.
  """
  if graph.nodes:
    return
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
    inputs=["x"],
    shape_metadata=out_shape_tuple,
    dtype=ir_dt,
    output_specs=[TensorSpec(shape=out_shape_tuple, dtype=ir_dt)],
  )
  graph.inputs = ["x"]
  graph.outputs = ["out"]


def _bind_graph_io_and_telemetry(
  graph: LogicalGraph,
  cst_graph: LogicalGraph,
  input_shapes: list[tuple[int, ...]],
  output_shape: tuple[int, ...],
  ir_dt: DType,
  node_telemetry: dict[str, NodeRuntimeTelemetry],
  observed_shapes: dict[str, list[int]],
) -> None:
  """Populate ``inputs``, ``input_specs``, ``outputs``, and node ``shape_metadata`` on ``graph``.

  Args:
      graph: Target ``LogicalGraph`` mutated in place.
      cst_graph: Static ``GraphExtractor`` graph carrying AST provenance.
      input_shapes: Input tensor shapes.
      output_shape: Terminal output tensor shape.
      ir_dt: Resolved ``DType`` enum member.
      node_telemetry: Per-node runtime telemetry map.
      observed_shapes: Observed concrete tensor shape dictionary.
  """
  in_shape_default = tuple(input_shapes[0]) if input_shapes else (2, 4)
  input_ids: list[str] = [nid for nid in graph.inputs if nid in graph.nodes]
  for nid, node in graph.nodes.items():
    if node.op_type in ("Input", "Placeholder") and nid not in input_ids:
      input_ids.append(nid)
  if not input_ids:
    input_ids.append(next(iter(graph.nodes.keys())))
  graph.inputs = input_ids

  for idx, in_id in enumerate(graph.inputs):
    shape_tuple = input_shapes[idx] if idx < len(input_shapes) else in_shape_default
    spec = TensorSpec(shape=tuple(shape_tuple), dtype=ir_dt)
    graph.input_specs[in_id] = spec
    graph.nodes[in_id].shape_metadata = tuple(shape_tuple)
    graph.nodes[in_id].dtype = ir_dt
    graph.nodes[in_id].output_specs = [spec]

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
    graph.outputs = [non_output[-1] if non_output else sorted_nodes[-1].id]

  for out_id in graph.outputs:
    if out_id in graph.nodes and graph.nodes[out_id].shape_metadata is None:
      graph.nodes[out_id].shape_metadata = tuple(output_shape)
      graph.nodes[out_id].dtype = ir_dt
      graph.nodes[out_id].output_specs = [TensorSpec(shape=tuple(output_shape), dtype=ir_dt)]


@dataclass(frozen=True)
class ForwardGraphBuildRequest:
  """Typed input bundle for ``build_hydrated_forward_graph``.

  Attributes:
      code: Source Python code string.
      tree: Parsed LibCST module.
      input_shapes: Input tensor shapes.
      output_shape: Terminal output tensor shape.
      dtype: Tensor data type string.
      config: Active ``RuntimeConfig``.
      node_telemetry: Per-node runtime telemetry recorded during forward execution.
      observed_shapes: Map of variable/node names to concrete runtime shapes.
  """

  code: str
  tree: cst.Module
  input_shapes: list[tuple[int, ...]]
  output_shape: tuple[int, ...]
  dtype: str
  config: RuntimeConfig = field(default_factory=RuntimeConfig)
  node_telemetry: dict[str, NodeRuntimeTelemetry] = field(default_factory=dict)
  observed_shapes: dict[str, list[int]] = field(default_factory=dict)


def build_hydrated_forward_graph(
  request: ForwardGraphBuildRequest,
) -> tuple[LogicalGraph, list[str]]:
  """Construct and hydrate the canonical ``ml_switcheroo_ir.LogicalGraph`` for the forward pass.

  Args:
      request: Typed ``ForwardGraphBuildRequest`` bundle with code, shapes, telemetry, and config.

  Returns:
      Tuple of ``(hydrated_forward_graph, executed_pass_names)``.
  """
  ensure_compiler_and_zero_on_path()
  ir_dt = _resolve_ir_dtype(request.dtype)
  extractor = GraphExtractor()
  request.tree.visit(extractor)
  cst_graph = extractor.graph

  graph = _extract_initial_forward_graph(
    request.code, cst_graph, request.input_shapes, request.dtype, request.config.random_seed
  )
  _seed_identity_fallback_if_empty(graph, request.input_shapes, request.output_shape, ir_dt)
  _bind_graph_io_and_telemetry(
    graph,
    cst_graph,
    request.input_shapes,
    request.output_shape,
    ir_dt,
    request.node_telemetry,
    request.observed_shapes,
  )
  _run_shape_inference_hooks(graph, request.observed_shapes)

  executed_passes: list[str] = ["propagate_shapes_and_constants", "shape_inference_pass"]
  if request.config.mode == ConversionMode.OPTIMIZED or request.config.enable_graph_optimization:
    graph, opt_passes = run_multi_pass_optimizer(graph, opt_level=request.config.opt_level)
    executed_passes.extend(opt_passes)

  return graph, executed_passes


__all__ = [
  "_NON_PYTHON_SOURCE_DIALECTS",
  "ForwardGraphBuildRequest",
  "_AstToIrModuleProtocol",
  "_lift_non_python_source_for_trace",
  "build_hydrated_forward_graph",
]
