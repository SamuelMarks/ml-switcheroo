"""Modular On-Device Runtime Tracing Package.

Organizes forward symbolic CST execution, isolated ``zero-*`` sandbox namespaces,
reverse-mode autodiff & hardware kernel synthesis, and end-to-end orchestration:
- ``ml_switcheroo.core.runtime_tracing.sandbox``
- ``ml_switcheroo.core.runtime_tracing.symbolic_executor``
- ``ml_switcheroo.core.runtime_tracing.autodiff``
- ``ml_switcheroo.core.runtime_tracing.orchestrator``
"""

from __future__ import annotations

from ml_switcheroo.core.runtime_tracing.autodiff import (
  _execute_autodiff_strategy,
  _invoke_codegen,
  _is_codegen_cls,
  _run_compiler_graph_and_codegen,
)
from ml_switcheroo.core.runtime_tracing.orchestrator import run_on_device_trace
from ml_switcheroo.core.runtime_tracing.sandbox import (
  SandboxNamespace,
  _resolve_traits,
  _TracedLayerInstance,
  zero_framework_sandbox,
)
from ml_switcheroo.core.runtime_tracing.symbolic_executor import _ForwardSymbolicExecutor

__all__ = [
  "SandboxNamespace",
  "_ForwardSymbolicExecutor",
  "_TracedLayerInstance",
  "_execute_autodiff_strategy",
  "_invoke_codegen",
  "_is_codegen_cls",
  "_resolve_traits",
  "_run_compiler_graph_and_codegen",
  "run_on_device_trace",
  "zero_framework_sandbox",
]
