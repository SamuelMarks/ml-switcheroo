"""End-to-End On-Device Trace Orchestrator for ``runtime_tracing``.

Provides ``run_on_device_trace`` and modular CST preparation and gradient
telemetry aggregation helpers.
"""

from __future__ import annotations

import math

import libcst as cst

from ml_switcheroo.config import ExecutionBackend, RuntimeConfig
from ml_switcheroo.core.ir_bridge import (
  _NON_PYTHON_SOURCE_DIALECTS,
  SharedIRBridgeRequest,
  _lift_non_python_source_for_trace,
  run_shared_ir_bridge,
)
from ml_switcheroo.core.runtime_models import (
  RuntimeTraceReport,
  infer_input_shapes_from_cst,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _check_all_finite,
  _compute_loss,
  _generate_seeded_values,
  _l2_norm,
  _prod,
)
from ml_switcheroo.core.runtime_tracing.autodiff import (
  _execute_autodiff_strategy,
  _run_compiler_graph_and_codegen,
)
from ml_switcheroo.core.runtime_tracing.sandbox import (
  _lookup_adapter,
  _resolve_traits,
  zero_framework_sandbox,
)
from ml_switcheroo.core.runtime_tracing.symbolic_executor import _ForwardSymbolicExecutor
from ml_switcheroo.semantics.manager import SemanticsManager


def _prepare_trace_cst(
  code: str,
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None,
) -> tuple[str, cst.Module | None, str | None]:
  """Lift non-Python dialects if needed and parse ``code`` into a ``cst.Module``.

  Args:
      code: Raw source code string.
      source_fw: Source framework or dialect identifier.
      target_fw: Target framework identifier.
      semantics: Optional ``SemanticsManager`` instance.

  Returns:
      Tuple of ``(trace_code, parsed_module_or_none, syntax_error_or_none)``.
  """
  trace_code = code
  tree: cst.Module | None = None
  if source_fw in _NON_PYTHON_SOURCE_DIALECTS:
    lifted = _lift_non_python_source_for_trace(code, source_fw, target_fw, semantics)
    if lifted is not None:
      trace_code, tree = lifted
  if tree is None:
    try:
      tree = cst.parse_module(trace_code)
    except cst.ParserSyntaxError as exc:
      return trace_code, None, f"Syntax error during on-device trace: {exc}"
  return trace_code, tree, None


def _build_seeded_inputs(
  inferred_shapes: list[tuple[int, ...]],
  dtype: str,
  random_seed: int,
) -> list[OnDeviceTraceTensor]:
  """Instantiate deterministic seeded ``OnDeviceTraceTensor`` inputs for ``inferred_shapes``.

  Args:
      inferred_shapes: List of concrete input tensor shapes.
      dtype: Input tensor dtype string.
      random_seed: Base PRNG seed.

  Returns:
      Ordered list of input ``OnDeviceTraceTensor`` instances.
  """
  requires_grad = "int" not in dtype and "bool" not in dtype
  return [
    OnDeviceTraceTensor(
      shape=shape,
      data=_generate_seeded_values(_prod(shape), random_seed + idx * 31, dtype),
      dtype=dtype,
      requires_grad=requires_grad,
    )
    for idx, shape in enumerate(inferred_shapes)
  ]


def _collect_gradient_telemetry(
  executor: _ForwardSymbolicExecutor,
  input_tensors: list[OnDeviceTraceTensor],
) -> tuple[dict[str, list[int]], dict[str, float], int, float]:
  """Populate activation gradient fields on ``executor`` and summarize parameter/input gradients.

  Args:
      executor: Completed ``_ForwardSymbolicExecutor`` after ``backward()``.
      input_tensors: Seeded forward input tensors.

  Returns:
      Tuple of ``(param_grads, param_grad_norms, param_count, total_grad_norm)``.
  """
  sq_grad_sum = 0.0
  for node_id, lineno, act_tensor in executor.traced_activations:
    if act_tensor.grad is not None:
      g_shape = list(act_tensor.grad.shape)
      g_norm = _l2_norm(act_tensor.grad.data)
      sq_grad_sum += sum(v * v for v in act_tensor.grad.data)
      executor.node_telemetry[node_id].grad_shape = g_shape
      executor.node_telemetry[node_id].grad_norm = g_norm
      executor.line_telemetry[lineno].grad_shape = g_shape
      executor.line_telemetry[lineno].grad_norm = g_norm

  param_grads: dict[str, list[int]] = {}
  param_grad_norms: dict[str, float] = {}
  param_count = 0
  for pname, ptensor in executor.parameters.items():
    param_count += len(ptensor.data)
    if ptensor.grad is not None:
      param_grads[pname] = list(ptensor.grad.shape)
      param_grad_norms[pname] = _l2_norm(ptensor.grad.data)
      sq_grad_sum += sum(v * v for v in ptensor.grad.data)

  for idx, inp in enumerate(input_tensors):
    if inp.grad is not None:
      in_name = "input" if idx == 0 else f"input_{idx}"
      param_grads[in_name] = list(inp.grad.shape)
      param_grad_norms[in_name] = _l2_norm(inp.grad.data)
      sq_grad_sum += sum(v * v for v in inp.grad.data)

  return param_grads, param_grad_norms, param_count, round(math.sqrt(sq_grad_sum), 6)


def run_on_device_trace(
  code: str,
  source_fw: str,
  target_fw: str,
  semantics: SemanticsManager | None = None,
  config: RuntimeConfig | None = None,
) -> RuntimeTraceReport:
  """Execute forward and backward passes on-device and return rich runtime telemetry.

  Args:
      code: Source Python code to trace.
      source_fw: Source framework identifier.
      target_fw: Target framework identifier.
      semantics: Optional ``SemanticsManager`` instance.
      config: Optional ``RuntimeConfig`` instance.

  Returns:
      Populated ``RuntimeTraceReport``.
  """
  active_cfg = config or RuntimeConfig(source_framework=source_fw, target_framework=target_fw)
  src_traits = _resolve_traits(source_fw, semantics)
  tgt_traits = _resolve_traits(target_fw, semantics)

  src_zero_pkg = src_traits.zero_runtime_package or "ml_switcheroo_compiler"
  tgt_zero_pkg = tgt_traits.zero_runtime_package or "ml_switcheroo_compiler"
  target_supported = _lookup_adapter(target_fw) is not None

  trace_code, tree, err_msg = _prepare_trace_cst(code, source_fw, target_fw, semantics)
  if tree is None:
    return RuntimeTraceReport(
      executed=False,
      backend_used=active_cfg.execution_backend.value,
      source_zero_pkg=src_zero_pkg,
      target_zero_pkg=tgt_zero_pkg,
      target_supported=target_supported,
      error=err_msg,
    )

  inferred_shapes, dtype, declared_layers = infer_input_shapes_from_cst(tree, active_cfg)

  with zero_framework_sandbox(source_fw, target_fw, semantics) as sandbox_mods:
    input_tensors = _build_seeded_inputs(inferred_shapes, dtype, active_cfg.random_seed)
    executor = _ForwardSymbolicExecutor(declared_layers, input_tensors, active_cfg.random_seed)
    out_tensor = executor.execute_module(tree)

    loss_tensor = _compute_loss(out_tensor, active_cfg.loss_reduction)
    loss_val = round(loss_tensor.item(), 6)

    _execute_autodiff_strategy(
      src_traits.autodiff_strategy or tgt_traits.autodiff_strategy,
      loss_tensor,
      sandbox_mods,
      autodiff_module_key=src_traits.autodiff_module_key or tgt_traits.autodiff_module_key,
    )

    param_grads, param_grad_norms, param_count, total_grad_norm = _collect_gradient_telemetry(executor, input_tensors)
    finite_ok = _check_all_finite([act for _, _, act in executor.traced_activations], executor.parameters, sandbox_mods)

    fwd_kernel, bwd_kernel, ir_in, ir_out, shape_md = _run_compiler_graph_and_codegen(
      code=trace_code,
      tree=tree,
      config=active_cfg,
      executor=executor,
      loss_val=loss_val,
    )
    bridge_res = run_shared_ir_bridge(
      SharedIRBridgeRequest(
        code=trace_code,
        tree=tree,
        config=active_cfg,
        target_fw=target_fw,
        input_shapes=inferred_shapes,
        output_shape=out_tensor.shape,
        dtype=dtype,
        loss_val=loss_val,
        node_telemetry=executor.node_telemetry,
        semantics=semantics,
      )
    )

    backend_used = (
      f"pyodide ({src_zero_pkg} -> {tgt_zero_pkg})"
      if active_cfg.execution_backend == ExecutionBackend.AUTO
      else active_cfg.execution_backend.value
    )

    return RuntimeTraceReport(
      executed=True,
      backend_used=backend_used,
      source_zero_pkg=src_zero_pkg,
      target_zero_pkg=tgt_zero_pkg,
      target_supported=target_supported,
      input_shapes=[list(s) for s in inferred_shapes],
      output_shape=list(out_tensor.shape),
      loss_value=loss_val,
      total_grad_norm=total_grad_norm,
      parameter_count=param_count,
      finite_check_passed=finite_ok,
      node_telemetry=executor.node_telemetry,
      line_telemetry=executor.line_telemetry,
      resolved_constructors=executor.resolved_constructors,
      resolved_branches=executor.resolved_branches,
      loop_trip_counts=executor.loop_trip_counts,
      parameter_gradients=param_grads,
      parameter_grad_norms=param_grad_norms,
      shape_table_markdown=shape_md,
      forward_kernel_code=fwd_kernel or bridge_res.forward_kernel_code,
      backward_kernel_code=bwd_kernel or bridge_res.backward_kernel_code,
      ir_inputs_meta=ir_in or bridge_res.ir_inputs_meta,
      ir_outputs_meta=ir_out or bridge_res.ir_outputs_meta,
      conversion_mode=active_cfg.mode.value,
      opt_level=active_cfg.opt_level.value,
      executed_passes=bridge_res.executed_passes,
      forward_ir_python=bridge_res.forward_ir_python,
      backward_ir_python=bridge_res.backward_ir_python,
      forward_ir_json=bridge_res.forward_ir_json,
      backward_ir_json=bridge_res.backward_ir_json,
      forward_mermaid=bridge_res.forward_mermaid,
      backward_mermaid=bridge_res.backward_mermaid,
      synthesized_backward_code=bridge_res.synthesized_backward_code,
      estimated_memory_bytes=bridge_res.estimated_memory_bytes,
      estimated_comm_bytes=bridge_res.estimated_comm_bytes,
    )


__all__ = ["run_on_device_trace"]
