"""Multi-Pass Compiler Transformation Bridge for `ml-switcheroo-ir` (`LogicalGraph`).

This module bridges `ml-switcheroo-compiler`'s middle-end `PassManager` and
`LogicalGraph` transformation passes (`shape_inference_pass`, `dtype_inference_pass`,
`constant_folding_pass`, `common_subexpression_elimination_pass`,
`dead_code_elimination_pass`, `batch_norm_folding_pass`, `operator_fusion_pass`,
`rematerialization_pass`, `spmd_partitioning_pass`, `inject_spmd_communication_pass`,
`lift_state_pass`, `state_lowering_pass`, `loop_tiling_pass`, `loop_unrolling_pass`,
`vectorization_pass`, `parallel_scan_pass`, `graph_scheduling_pass`, and
`buffer_allocation_pass`) alongside `ml_switcheroo_ir` graph passes.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import types
import typing

import libcst as cst
from ml_switcheroo_ir import (
  LogicalGraph,
  eliminate_common_subexpressions,
  eliminate_dead_nodes,
  propagate_shapes_and_constants,
)

from ml_switcheroo.config import OptimizationLevel
from ml_switcheroo.core.bootstrap import ensure_compiler_and_zero_on_path as ensure_compiler_and_zero_on_path
from ml_switcheroo.core.compiler.sharding import ShardingInferencePass, estimate_tensor_memory
from ml_switcheroo.core.tracer import TraceLogger


def _load_compiler_passes_module() -> types.ModuleType | None:
  """Import `ml_switcheroo_compiler.transforms.passes` if available.

  Returns:
      Imported passes module or None if unavailable.
  """
  ensure_compiler_and_zero_on_path()
  try:
    return importlib.import_module("ml_switcheroo_compiler.transforms.passes")
  except ImportError:
    return None


def _load_pass_manager_module() -> types.ModuleType | None:
  """Import `ml_switcheroo_compiler.transforms.pass_manager` if available.

  Returns:
      Imported pass_manager module or None if unavailable.
  """
  ensure_compiler_and_zero_on_path()
  try:
    return importlib.import_module("ml_switcheroo_compiler.transforms.pass_manager")
  except ImportError:
    return None


@typing.runtime_checkable
class _GraphPassCallableProtocol(typing.Protocol):
  """Protocol for single-argument `LogicalGraph` compiler pass callables."""

  def __call__(self, graph: LogicalGraph) -> LogicalGraph | None:
    """Apply pass transformation to ``graph``.

    Args:
        graph: Target `LogicalGraph`.

    Returns:
        Optional updated `LogicalGraph` or None when mutated in-place.
    """


@typing.runtime_checkable
class _PassManagerInstanceProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.transforms.pass_manager.PassManager` instances."""

  def add_pass(self, pass_fn: _GraphPassCallableProtocol, name: str = "") -> None:
    """Register a pass callable with the pass manager.

    Args:
        pass_fn: Pass callable.
        name: Human-readable pass identifier.
    """

  def run_until_converged(self, graph: LogicalGraph, max_iterations: int = 5) -> LogicalGraph:
    """Execute registered passes iteratively until convergence.

    Args:
        graph: Input `LogicalGraph`.
        max_iterations: Maximum fixpoint iterations.

    Returns:
        Converged `LogicalGraph`.
    """


@typing.runtime_checkable
class _PassManagerModuleProtocol(typing.Protocol):
  """Protocol for `ml_switcheroo_compiler.transforms.pass_manager` module.

  Attributes:
      PassManager: Pass manager constructor.
  """

  PassManager: typing.Callable[[], _PassManagerInstanceProtocol]


def _lookup_graph_pass(passes_mod: types.ModuleType, pass_name: str) -> _GraphPassCallableProtocol | None:
  """Look up a named graph pass callable from ``passes_mod`` without ``getattr_static``.

  Args:
      passes_mod: Loaded `ml_switcheroo_compiler.transforms.passes` module.
      pass_name: Function name of the pass to retrieve.

  Returns:
      Matching pass callable if present, otherwise None.
  """
  for name, member in inspect.getmembers(passes_mod):
    if name == pass_name and isinstance(member, _GraphPassCallableProtocol):
      return member
  return None


def _invoke_graph_pass(passes_mod: types.ModuleType, pass_name: str, graph: LogicalGraph) -> bool:
  """Invoke a single named pass on `graph` safely.

  Args:
      passes_mod: Loaded `ml_switcheroo_compiler.transforms.passes` module.
      pass_name: Function name of the pass to execute.
      graph: Target `LogicalGraph` to transform in-place.

  Returns:
      True if the pass executed without raising an exception.
  """
  pass_fn = _lookup_graph_pass(passes_mod, pass_name)
  if pass_fn is None:
    return False
  try:
    pass_fn(graph)
    return True
  except (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError):
    return False


def _run_named_passes(
  passes_mod: types.ModuleType | None,
  pass_names: tuple[str, ...],
  graph: LogicalGraph,
) -> list[str]:
  """Invoke a sequence of named passes on `graph` when `passes_mod` is available.

  Args:
      passes_mod: Loaded `ml_switcheroo_compiler.transforms.passes` module or None.
      pass_names: Tuple of pass function names to invoke.
      graph: Target `LogicalGraph` to transform in-place.

  Returns:
      Ordered list of successfully executed pass names.
  """
  if passes_mod is None:
    return []
  return [p_name for p_name in pass_names if _invoke_graph_pass(passes_mod, p_name, graph)]


def run_canonicalization_passes(graph: LogicalGraph) -> list[str]:
  """Run O0 canonicalization and type/shape inference passes on `graph`.

  Args:
      graph: Target `LogicalGraph` to canonicalize.

  Returns:
      Ordered list of executed pass names.
  """
  passes_mod = _load_compiler_passes_module()
  executed = _run_named_passes(
    passes_mod,
    (
      "polyfill_lowering_pass",
      "type_promotion_explicitizer_pass",
      "broadcast_explicitizer_pass",
      "shape_inference_pass",
      "dtype_inference_pass",
    ),
    graph,
  )
  propagate_shapes_and_constants(graph)
  executed.append("propagate_shapes_and_constants")
  return executed


def run_algebraic_simplification_passes(graph: LogicalGraph, include_cse_and_bn: bool = True) -> list[str]:
  """Run O1/O2 constant folding, dead code elimination, CSE, and batch-norm folding passes.

  Args:
      graph: Target `LogicalGraph` to simplify.
      include_cse_and_bn: If True, also runs CSE, batch-norm folding, and control-flow optimization.

  Returns:
      Ordered list of executed pass names.
  """
  passes_mod = _load_compiler_passes_module()
  executed = _run_named_passes(passes_mod, ("constant_folding_pass",), graph)
  propagate_shapes_and_constants(graph)
  executed.append("propagate_shapes_and_constants")

  if include_cse_and_bn:
    executed.extend(
      _run_named_passes(
        passes_mod,
        (
          "common_subexpression_elimination_pass",
          "batch_norm_folding_pass",
          "control_flow_optimization_pass",
        ),
        graph,
      )
    )
    eliminate_common_subexpressions(graph)
    executed.append("eliminate_common_subexpressions")

  if graph.outputs:
    executed.extend(_run_named_passes(passes_mod, ("dead_code_elimination_pass",), graph))
    eliminate_dead_nodes(graph)
    executed.append("eliminate_dead_nodes")
  return executed


def run_fusion_and_scheduling_passes(
  graph: LogicalGraph,
  include_aggressive: bool = False,
) -> list[str]:
  """Run O2/O3 operator fusion, scheduling, buffer allocation, and aggressive loop/SPMD passes.

  Args:
      graph: Target `LogicalGraph` to optimize.
      include_aggressive: If True (O3), runs mixed precision, loop tiling/unrolling,
          vectorization, parallel scan, rematerialization, SPMD, and state lifting/lowering.

  Returns:
      Ordered list of executed pass names.
  """
  passes_mod = _load_compiler_passes_module()
  executed = _run_named_passes(
    passes_mod,
    (
      "operator_fusion_pass",
      "graph_scheduling_pass",
      "buffer_allocation_pass",
    ),
    graph,
  )

  if include_aggressive:
    executed.extend(
      _run_named_passes(
        passes_mod,
        (
          "mixed_precision_pass",
          "loop_tiling_pass",
          "loop_unrolling_pass",
          "vectorization_pass",
          "parallel_scan_pass",
          "rematerialization_pass",
          "spmd_partitioning_pass",
          "inject_spmd_communication_pass",
          "lift_state_pass",
          "state_lowering_pass",
        ),
        graph,
      )
    )

  return executed


def run_multi_pass_optimizer(
  graph: LogicalGraph,
  opt_level: OptimizationLevel = OptimizationLevel.O2,
  tracer: TraceLogger | None = None,
) -> tuple[LogicalGraph, list[str]]:
  """Execute multi-pass `LogicalGraph` optimization according to `opt_level`.

  Coordinates `ml_switcheroo_compiler.transforms.PassManager` (`run_until_converged`)
  with `ml_switcheroo_ir` algebraic passes (`propagate_shapes_and_constants`,
  `eliminate_common_subexpressions`, `eliminate_dead_nodes`).

  Args:
      graph: Shared `ml_switcheroo_ir.LogicalGraph` to optimize.
      opt_level: Requested `OptimizationLevel` (`O0`, `O1`, `O2`, `O3`).
      tracer: Optional `TraceLogger` for recording pass execution telemetry.

  Returns:
      Tuple of `(optimized_graph, executed_pass_names)`.
  """
  executed_passes: list[str] = []

  if tracer is not None:
    tracer.start_phase("Multi-Pass Compiler Optimizer", f"Level {opt_level.value}")

  # 1. Canonicalization (O0+)
  executed_passes.extend(run_canonicalization_passes(graph))

  # 2. Algebraic Simplification (O1+)
  if opt_level in (OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3):
    include_cse = opt_level in (OptimizationLevel.O2, OptimizationLevel.O3)
    executed_passes.extend(run_algebraic_simplification_passes(graph, include_cse_and_bn=include_cse))

  # 3. Fusion, Scheduling & Aggressive Passes (O2+)
  if opt_level in (OptimizationLevel.O2, OptimizationLevel.O3):
    include_aggressive = opt_level == OptimizationLevel.O3
    executed_passes.extend(run_fusion_and_scheduling_passes(graph, include_aggressive=include_aggressive))

    pm_mod = _load_pass_manager_module()
    passes_mod = _load_compiler_passes_module()
    if isinstance(pm_mod, _PassManagerModuleProtocol) and passes_mod is not None and graph.outputs:
      try:
        pm = pm_mod.PassManager()
        for p_name in (
          "constant_folding_pass",
          "canonicalize_pass",
          "common_subexpression_elimination_pass",
          "operator_fusion_pass",
        ):
          p_fn = _lookup_graph_pass(passes_mod, p_name)
          if p_fn is not None:
            pm.add_pass(p_fn, name=p_name)
        graph = pm.run_until_converged(graph, max_iterations=5)
        executed_passes.append("PassManager.run_until_converged")
      except (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError) as exc:
        logging.getLogger(__name__).debug("PassManager.run_until_converged skipped: %s", exc)

  if tracer is not None:
    tracer.log_mutation(
      "Multi-Pass Optimizer",
      f"OptimizationLevel.{opt_level.value}",
      ", ".join(executed_passes),
    )
    tracer.end_phase()

  return graph, executed_passes


def estimate_graph_telemetry(graph: LogicalGraph) -> tuple[float, float]:
  """Estimate total memory footprint and collective communication volume for `graph`.

  Args:
      graph: Target `LogicalGraph` with populated shape metadata.

  Returns:
      Tuple of `(estimated_memory_bytes, estimated_comm_bytes)`.
  """
  sharding_pass = ShardingInferencePass()
  mem_bytes = float(sharding_pass.estimate_graph_memory_bytes(graph))
  if mem_bytes == 0.0:
    for node in graph.nodes.values():
      if isinstance(node.shape_metadata, (list, tuple)):
        mem_bytes += estimate_tensor_memory(
          shape=node.shape_metadata,
          dtype=str(node.dtype or "float32"),
        )
  comm_bytes = float(sharding_pass.estimate_graph_communication_bytes(graph))
  return mem_bytes, comm_bytes


class CSTConstantFolder(cst.CSTTransformer):
  """LibCST transformer that folds constant numeric expressions in `ConversionMode.OPTIMIZED`.

  Folds arithmetic expressions over integer and floating-point literals (e.g.,
  `2 * 3` -> `6`, `1 + 2 * 3` -> `7`, `(4 + 4) // 2` -> `4`) while leaving
  non-constant expressions untouched.
  """

  @staticmethod
  def _extract_numeric_literal(node: cst.BaseExpression) -> int | float | None:
    """Extract an `int` or `float` value from a CST literal or parenthesized literal.

    Args:
        node: LibCST expression node.

    Returns:
        Numeric value (`int` or `float`) or None if not a constant literal.
    """
    if isinstance(node, cst.Integer):
      return int(node.value, 0)
    if isinstance(node, cst.Float):
      return float(node.value)
    if (
      isinstance(node, cst.UnaryOperation)
      and isinstance(node.operator, cst.Minus)
      and isinstance(node.expression, (cst.Integer, cst.Float))
    ):
      inner_val = (
        int(node.expression.value, 0) if isinstance(node.expression, cst.Integer) else float(node.expression.value)
      )
      return -inner_val
    return None

  @staticmethod
  def _make_literal_node(val: int | float) -> cst.BaseExpression:
    """Construct a LibCST literal expression from a folded numeric value.

    Args:
        val: Folded `int` or `float` value.

    Returns:
        `cst.Integer`, `cst.Float`, or `cst.UnaryOperation` for negative numbers.
    """
    if isinstance(val, int) and not isinstance(val, bool):
      if val < 0:
        return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Integer(str(abs(val))))
      return cst.Integer(str(val))
    if val < 0.0:
      return cst.UnaryOperation(operator=cst.Minus(), expression=cst.Float(str(abs(val))))
    return cst.Float(str(val))

  def leave_BinaryOperation(
    self,
    original_node: cst.BinaryOperation,
    updated_node: cst.BinaryOperation,
  ) -> cst.BaseExpression:
    """Fold binary operations whose left and right operands are numeric literals.

    Args:
        original_node: Original CST BinaryOperation node.
        updated_node: Updated CST BinaryOperation node after child visits.

    Returns:
        Folded literal node or `updated_node` if not statically foldable.
    """
    left_val = self._extract_numeric_literal(updated_node.left)
    right_val = self._extract_numeric_literal(updated_node.right)
    if left_val is None or right_val is None:
      return updated_node

    result: int | float | None = None
    op = updated_node.operator
    if isinstance(op, cst.Add):
      result = left_val + right_val
    elif isinstance(op, cst.Subtract):
      result = left_val - right_val
    elif isinstance(op, cst.Multiply):
      result = left_val * right_val
    elif isinstance(op, cst.FloorDivide) and right_val != 0:
      result = left_val // right_val
    elif isinstance(op, cst.Divide) and right_val != 0:
      result = left_val / right_val

    if result is None:
      return updated_node
    return self._make_literal_node(result)
