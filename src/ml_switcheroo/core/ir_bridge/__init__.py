"""Shared ``ml-switcheroo-ir`` (``LogicalGraph``) Unification Bridge Package.

Decomposes forward lifting, live ``ProxyTensor`` tracing, backward VJP lowering,
and runtime trace hydration into dedicated submodules:
- ``ml_switcheroo.core.ir_bridge.proxy_tracing``
- ``ml_switcheroo.core.ir_bridge.lifting``
- ``ml_switcheroo.core.ir_bridge.lowering``
- ``ml_switcheroo.core.ir_bridge.hydration``
"""

from __future__ import annotations

import importlib

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.ir_bridge.hydration import (
  _format_node_runtime_comment,
  hydrate_graph_with_runtime_trace,
)
from ml_switcheroo.core.ir_bridge.lifting import (
  _NON_PYTHON_SOURCE_DIALECTS,
  ForwardGraphBuildRequest,
  _AstToIrModuleProtocol,
  _lift_non_python_source_for_trace,
  build_hydrated_forward_graph,
)
from ml_switcheroo.core.ir_bridge.lowering import (
  SharedIRBridgeRequest,
  SharedIRBridgeResult,
  _CompilerTensorConfigProtocol,
  _CompilerTensorProtocol,
  _EvalGraphModuleProtocol,
  _GradModuleProtocol,
  _IrToAstModuleProtocol,
  _ShapeObjectProtocol,
  _TensorModuleProtocol,
  evaluate_shared_backward_graph,
  render_logical_graph_mermaid,
  run_shared_ir_bridge,
  synthesize_backward_vjp_graph,
  synthesize_target_backward_code,
)
from ml_switcheroo.core.ir_bridge.proxy_tracing import (
  _eval_proxy_ast_expr,
  _LearnedShapesModuleProtocol,
  _ProxyTensorCtorProtocol,
  _ProxyTensorProtocol,
  _resolve_ir_dtype,
  _run_shape_inference_hooks,
  _ShapeInferenceModuleProtocol,
  _TracingModuleProtocol,
  _TracingStateProtocol,
  trace_with_zero_proxy_tensors,
)

__all__ = [
  "_NON_PYTHON_SOURCE_DIALECTS",
  "ForwardGraphBuildRequest",
  "PythonBackend",
  "SharedIRBridgeRequest",
  "SharedIRBridgeResult",
  "_AstToIrModuleProtocol",
  "_CompilerTensorConfigProtocol",
  "_CompilerTensorProtocol",
  "_EvalGraphModuleProtocol",
  "_GradModuleProtocol",
  "_IrToAstModuleProtocol",
  "_LearnedShapesModuleProtocol",
  "_ProxyTensorCtorProtocol",
  "_ProxyTensorProtocol",
  "_ShapeInferenceModuleProtocol",
  "_ShapeObjectProtocol",
  "_TensorModuleProtocol",
  "_TracingModuleProtocol",
  "_TracingStateProtocol",
  "_eval_proxy_ast_expr",
  "_format_node_runtime_comment",
  "_lift_non_python_source_for_trace",
  "_resolve_ir_dtype",
  "_run_shape_inference_hooks",
  "build_hydrated_forward_graph",
  "evaluate_shared_backward_graph",
  "hydrate_graph_with_runtime_trace",
  "importlib",
  "render_logical_graph_mermaid",
  "run_shared_ir_bridge",
  "synthesize_backward_vjp_graph",
  "synthesize_target_backward_code",
  "trace_with_zero_proxy_tensors",
]
