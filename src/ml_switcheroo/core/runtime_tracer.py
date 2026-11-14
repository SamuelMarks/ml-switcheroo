"""On-Device Forward and Backward Pass Execution Engine Facade.

Re-exports the modular runtime tracing subsystems implemented in
``ml_switcheroo.core.runtime_tracing`` alongside compatibility helpers.
"""

from __future__ import annotations

import importlib

from ml_switcheroo.core.ir_bridge import (
  _NON_PYTHON_SOURCE_DIALECTS,
  _lift_non_python_source_for_trace,
  hydrate_graph_with_runtime_trace,
)
from ml_switcheroo.core.runtime_models import (
  BranchResolutionRecord,
  ConstructorDimResolution,
  DeclaredLayerSpec,
  InputShapeInferenceVisitor,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
  _extract_dims_from_annotation_text,
  _extract_int_from_expr,
  _get_call_leaf_name,
  infer_input_shapes_from_cst,
)
from ml_switcheroo.core.runtime_tensor import (
  OnDeviceTraceTensor,
  _check_all_finite,
  _compute_loss,
  _l2_norm,
)
from ml_switcheroo.core.runtime_tracing import (
  SandboxNamespace,
  _execute_autodiff_strategy,
  _ForwardSymbolicExecutor,
  _invoke_codegen,
  _is_codegen_cls,
  _resolve_traits,
  _run_compiler_graph_and_codegen,
  _TracedLayerInstance,
  run_on_device_trace,
  zero_framework_sandbox,
)
from ml_switcheroo.frameworks.base import get_adapter

__all__ = [
  "_NON_PYTHON_SOURCE_DIALECTS",
  "BranchResolutionRecord",
  "ConstructorDimResolution",
  "DeclaredLayerSpec",
  "InputShapeInferenceVisitor",
  "NodeRuntimeTelemetry",
  "OnDeviceTraceTensor",
  "RuntimeTraceReport",
  "SandboxNamespace",
  "_ForwardSymbolicExecutor",
  "_TracedLayerInstance",
  "_check_all_finite",
  "_compute_loss",
  "_execute_autodiff_strategy",
  "_extract_dims_from_annotation_text",
  "_extract_int_from_expr",
  "_get_call_leaf_name",
  "_invoke_codegen",
  "_is_codegen_cls",
  "_l2_norm",
  "_lift_non_python_source_for_trace",
  "_resolve_traits",
  "_run_compiler_graph_and_codegen",
  "get_adapter",
  "hydrate_graph_with_runtime_trace",
  "importlib",
  "infer_input_shapes_from_cst",
  "run_on_device_trace",
  "zero_framework_sandbox",
]
