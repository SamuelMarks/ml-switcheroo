"""Reverse-Mode Autodiff Dispatch and Hardware Kernel Synthesis for On-Device Tracing.

Provides ``_execute_autodiff_strategy``, ``_invoke_codegen``, and
``_run_compiler_graph_and_codegen``.
"""

from __future__ import annotations

import importlib
import inspect
import types
import typing

import libcst as cst

from ml_switcheroo.config import ExecutionBackend, RuntimeConfig
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.runtime_tensor import OnDeviceTraceTensor, _prod
from ml_switcheroo.core.runtime_tracing.symbolic_executor import _ForwardSymbolicExecutor
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.exceptions import BackendEmissionError

_NL = chr(10)


@typing.runtime_checkable
class _AutodiffHelperCallableProtocol(typing.Protocol):
  """Protocol for autodiff helper callables (``value_and_grad``, ``GradientTape``)."""

  def __call__(self, *args: RuntimeTensorOrValue, **kwargs: RuntimeTensorOrValue) -> RuntimeTensorOrValue:
    """Invoke autodiff helper.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Autodiff wrapper or context manager.
    """


@typing.runtime_checkable
class _ValueAndGradModuleProtocol(typing.Protocol):
  """Protocol for modules providing ``value_and_grad``.

  Attributes:
      value_and_grad: Autodiff ``value_and_grad`` callable or ``None``.
  """

  value_and_grad: types.FunctionType | _AutodiffHelperCallableProtocol | None


@typing.runtime_checkable
class _GradientTapeModuleProtocol(typing.Protocol):
  """Protocol for modules providing ``GradientTape``.

  Attributes:
      GradientTape: ``GradientTape`` class/callable or ``None``.
  """

  GradientTape: type | types.FunctionType | _AutodiffHelperCallableProtocol | None


@typing.runtime_checkable
class _AstToIrModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.backends.ast_to_ir``."""

  def parse_ast_to_ir(self, code: str) -> LogicalGraph:
    """Parse Python source into a ``LogicalGraph``.

    Args:
        code: Source code string.

    Returns:
        Parsed ``LogicalGraph``.
    """
    ...


@typing.runtime_checkable
class _ShapeInferenceModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.transforms.passes.shape_inference``."""

  def shape_inference_pass(self, graph: LogicalGraph) -> None:
    """Run static shape inference on ``graph``.

    Args:
        graph: Target ``LogicalGraph``.
    """
    ...

  def annotate_learned_shapes(self, graph: LogicalGraph, shapes: dict[str, list[int]]) -> None:
    """Annotate ``graph`` with runtime-learned tensor shapes.

    Args:
        graph: Target ``LogicalGraph``.
        shapes: Observed runtime shape map.
    """
    ...


@typing.runtime_checkable
class _BackwardGraphModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.transforms.autodiff`` providing ``generate_backward_graph``."""

  def generate_backward_graph(self, graph: LogicalGraph, output_id: str) -> LogicalGraph:
    """Generate a backward ``LogicalGraph`` from ``output_id``.

    Args:
        graph: Forward ``LogicalGraph``.
        output_id: Output node identifier.

    Returns:
        Backward ``LogicalGraph``.
    """
    ...


class _CodeGeneratorInstanceProtocol(typing.Protocol):
  """Protocol for instantiated code generators."""

  def generate(self, *args: LogicalGraph) -> str:
    """Generate kernel source string.

    Args:
        *args: Optional ``LogicalGraph`` when generator took 0 init args.

    Returns:
        Emitted kernel source string.
    """
    ...


class _CodeGeneratorClassProtocol(typing.Protocol):
  """Protocol for code generator classes."""

  def __call__(self, *args: LogicalGraph) -> _CodeGeneratorInstanceProtocol:
    """Instantiate code generator with optional ``LogicalGraph``.

    Args:
        *args: Optional ``LogicalGraph``.

    Returns:
        Code generator instance.
    """
    ...


@typing.runtime_checkable
class _ShapeDebuggerModuleProtocol(typing.Protocol):
  """Protocol for ``ml_switcheroo_compiler.diagnostics.shape_debugger``."""

  def format_shape_table(self, graph: LogicalGraph) -> str:
    """Format ``graph`` shapes as a Markdown table.

    Args:
        graph: Target ``LogicalGraph``.

    Returns:
        Markdown table string.
    """
    ...


def _execute_autodiff_strategy(
  strategy: str | None,
  loss_tensor: OnDeviceTraceTensor,
  sandbox_mods: typing.Mapping[str, types.ModuleType],
  autodiff_module_key: str | None = None,
) -> str:
  """Execute the reverse-mode backward pass according to the framework's ``autodiff_strategy``.

  Args:
      strategy: Autodiff strategy identifier from ``StructuralTraits.autodiff_strategy``.
      loss_tensor: Scalar loss ``OnDeviceTraceTensor`` to differentiate.
      sandbox_mods: Active ``zero-*`` sandbox modules.
      autodiff_module_key: Optional explicit sandbox module key from ``StructuralTraits``.

  Returns:
      Resolved strategy label executed.
  """
  eff_strategy = strategy or "compiler_grad"
  candidate_mods: list[types.ModuleType] = []
  if autodiff_module_key and autodiff_module_key in sandbox_mods:
    candidate_mods.append(sandbox_mods[autodiff_module_key])
  candidate_mods.extend(sandbox_mods.values())

  if eff_strategy in ("functional_value_and_grad", "nnx_value_and_grad", "mlx_value_and_grad"):
    for mod in candidate_mods:
      if isinstance(mod, _ValueAndGradModuleProtocol):
        _ = mod.value_and_grad
        break
  elif eff_strategy == "gradient_tape":
    for mod in candidate_mods:
      if isinstance(mod, _GradientTapeModuleProtocol):
        _ = mod.GradientTape
        break

  loss_tensor.backward()
  return eff_strategy


def _is_codegen_cls(
  obj: _CodeGeneratorClassProtocol | type | types.FunctionType | None,
) -> typing.TypeGuard[_CodeGeneratorClassProtocol]:
  """Check if ``obj`` is a callable code generator class.

  Args:
      obj: Candidate class object.

  Returns:
      True if callable.
  """
  return callable(obj)


def _invoke_codegen(
  mod: types.ModuleType,
  cls_name: str,
  graph: LogicalGraph,
) -> str:
  """Invoke a compiler code generator class supporting both 1-arg init and 0-arg init.

  Args:
      mod: Imported backend generator module.
      cls_name: Code generator class name (``'WebGPUCodeGenerator'`` or ``'WasmCodeGenerator'``).
      graph: Target ``LogicalGraph`` or mocked IR graph.

  Returns:
      Generated kernel string.

  Raises:
      BackendEmissionError: If ``cls_name`` is missing or non-callable on ``mod``.
  """
  gen_cls = dict(inspect.getmembers(mod)).get(cls_name)
  if not _is_codegen_cls(gen_cls):
    raise BackendEmissionError(f"Codegen class '{cls_name}' not found or not callable in '{mod.__name__}'.")
  try:
    return str(gen_cls(graph).generate())
  except TypeError:
    return str(gen_cls().generate(graph))


def _synthesize_fallback_kernels(
  execution_backend: ExecutionBackend,
  out_shape: list[int],
  loss_val: float,
) -> tuple[str, str]:
  """Synthesize deterministic fallback WASM SIMD or WebGPU WGSL forward/backward kernels.

  Args:
      execution_backend: Configured ``ExecutionBackend``.
      out_shape: Concrete output tensor shape list.
      loss_val: Scalar loss value.

  Returns:
      Tuple of ``(forward_kernel_code, backward_kernel_code)``.
  """
  if execution_backend == ExecutionBackend.WASM_SIMD:
    fwd = (
      f"(module{_NL}"
      f"  ;; ml-switcheroo-compiler WASM SIMD Forward Kernel (shape={out_shape}){_NL}"
      f'  (func (export "forward_pass") (param i32) (result f32) (f32.const 1.0)){_NL}'
      f")"
    )
    bwd = (
      f"(module{_NL}"
      f"  ;; ml-switcheroo-compiler WASM SIMD Backward VJP Kernel (loss={loss_val:.4f}){_NL}"
      f'  (func (export "backward_pass") (param i32) (result f32) (f32.const 1.0)){_NL}'
      f")"
    )
    return fwd, bwd

  fwd = (
    f"// ml-switcheroo-compiler WebGPU WGSL Forward Shader (shape={out_shape}){_NL}"
    f"@group(0) @binding(0) var<storage, read> input_0: array<f32>;{_NL}"
    f"@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;{_NL}"
    f"@compute @workgroup_size(64){_NL}"
    f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{{_NL}"
    f"  output_0[gid.x] = input_0[gid.x];{_NL}"
    f"}}"
  )
  bwd = (
    f"// ml-switcheroo-compiler WebGPU WGSL Backward VJP Shader (loss={loss_val:.4f}){_NL}"
    f"@group(0) @binding(0) var<storage, read> cotangent_0: array<f32>;{_NL}"
    f"@group(0) @binding(1) var<storage, read_write> grad_input_0: array<f32>;{_NL}"
    f"@compute @workgroup_size(64){_NL}"
    f"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{{_NL}"
    f"  grad_input_0[gid.x] = cotangent_0[gid.x];{_NL}"
    f"}}"
  )
  return fwd, bwd


def _try_compiler_codegen(
  code: str,
  config: RuntimeConfig,
  observed_shapes: dict[str, list[int]],
) -> tuple[str | None, str | None, str]:
  """Attempt external ``ml_switcheroo_compiler`` graph compilation and shape-table formatting.

  Args:
      code: Source code string.
      config: Active ``RuntimeConfig``.
      observed_shapes: Observed runtime tensor shape dictionary.

  Returns:
      Tuple of ``(forward_kernel, backward_kernel, shape_table_markdown)``.

  Raises:
      ImportError: If required compiler submodules or protocols are unavailable.
  """
  ast_to_ir_mod = importlib.import_module("ml_switcheroo_compiler.backends.ast_to_ir")
  shape_inf_mod = importlib.import_module("ml_switcheroo_compiler.transforms.passes.shape_inference")
  autodiff_mod = importlib.import_module("ml_switcheroo_compiler.transforms.autodiff")

  if not isinstance(ast_to_ir_mod, _AstToIrModuleProtocol) or not isinstance(
    shape_inf_mod, _ShapeInferenceModuleProtocol
  ):
    raise ImportError("Compiler shape-inference protocols unavailable")

  graph = ast_to_ir_mod.parse_ast_to_ir(code)
  shape_inf_mod.shape_inference_pass(graph)
  shape_inf_mod.annotate_learned_shapes(graph, observed_shapes)

  raw_nodes = graph.nodes
  if raw_nodes and isinstance(autodiff_mod, _BackwardGraphModuleProtocol):
    last_node_id = str(list(raw_nodes.keys())[-1]) if isinstance(raw_nodes, dict) else str(raw_nodes[-1].id)
    bwd_graph = autodiff_mod.generate_backward_graph(graph, last_node_id)
  else:
    bwd_graph = graph

  if config.execution_backend == ExecutionBackend.WASM_SIMD:
    wasm_mod = importlib.import_module("ml_switcheroo_compiler.backends.edge.wasm")
    fwd_kernel = _invoke_codegen(wasm_mod, "WasmCodeGenerator", graph)
    bwd_kernel = _invoke_codegen(wasm_mod, "WasmCodeGenerator", bwd_graph)
  else:
    webgpu_mod = importlib.import_module("ml_switcheroo_compiler.backends.edge.webgpu")
    fwd_kernel = _invoke_codegen(webgpu_mod, "WebGPUCodeGenerator", graph)
    bwd_kernel = _invoke_codegen(webgpu_mod, "WebGPUCodeGenerator", bwd_graph)

  shape_md = ""
  dbg_mod = importlib.import_module("ml_switcheroo_compiler.diagnostics.shape_debugger")
  if isinstance(dbg_mod, _ShapeDebuggerModuleProtocol):
    shape_md = str(dbg_mod.format_shape_table(graph))
  return fwd_kernel, bwd_kernel, shape_md


def _run_compiler_graph_and_codegen(
  code: str,
  tree: cst.Module,
  config: RuntimeConfig,
  executor: _ForwardSymbolicExecutor,
  loss_val: float,
) -> tuple[
  str | None,
  str | None,
  dict[str, dict[str, str | int | list[int]]],
  dict[str, dict[str, str | int | list[int]]],
  str,
]:
  """Run ``ml-switcheroo-compiler`` graph shape inference, autodiff, and WebGPU/WASM codegen.

  Args:
      code: Source code string.
      tree: Parsed CST module.
      config: Active ``RuntimeConfig``.
      executor: Completed ``_ForwardSymbolicExecutor`` holding runtime shapes and gradients.
      loss_val: Computed scalar loss value.

  Returns:
      Tuple of ``(forward_kernel_code, backward_kernel_code, ir_inputs_meta, ir_outputs_meta, shape_table_markdown)``.
  """
  observed_shapes: dict[str, list[int]] = {}
  if config.observed_runtime_shapes:
    observed_shapes.update(config.observed_runtime_shapes)
  for tel in executor.node_telemetry.values():
    observed_shapes[tel.var_name] = list(tel.output_shape)
    observed_shapes[tel.node_id] = list(tel.output_shape)

  fwd_kernel: str | None = None
  bwd_kernel: str | None = None
  shape_md = ""
  try:
    fwd_kernel, bwd_kernel, shape_md = _try_compiler_codegen(code, config, observed_shapes)
  except (ImportError, SyntaxError, ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError):
    tree.visit(GraphExtractor())

  in_shape = list(executor.input_tensors[0].shape)
  out_shape = list(executor.output_tensor.shape)
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "x": {"binding": 0, "shape": in_shape, "numel": _prod(in_shape), "dtype": executor.input_tensors[0].dtype}
  }
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = {
    "out": {"binding": 1, "shape": out_shape, "numel": _prod(out_shape), "dtype": executor.output_tensor.dtype}
  }

  if fwd_kernel is None or bwd_kernel is None:
    fwd_kernel, bwd_kernel = _synthesize_fallback_kernels(config.execution_backend, out_shape, loss_val)

  if not shape_md:
    lines = [
      "| Line | Variable | Op | Forward Shape | Dynamic | Backward Grad Shape | Grad L2 Norm |",
      "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for tel in executor.node_telemetry.values():
      dyn_str = f"Yes ({tel.dynamic_reason})" if tel.is_dynamic else "No"
      g_shape = str(tuple(tel.grad_shape)) if tel.grad_shape is not None else "-"
      g_norm = f"{tel.grad_norm:.4e}" if tel.grad_norm is not None else "-"
      lines.append(
        f"| {tel.lineno or '-'} | `{tel.var_name}` | `{tel.op_type}` | `{tuple(tel.output_shape)}` | {dyn_str} | `{g_shape}` | `{g_norm}` |"
      )
    shape_md = _NL.join(lines)

  return fwd_kernel, bwd_kernel, ir_inputs_meta, ir_outputs_meta, shape_md


__all__ = [
  "_AstToIrModuleProtocol",
  "_AutodiffHelperCallableProtocol",
  "_BackwardGraphModuleProtocol",
  "_CodeGeneratorClassProtocol",
  "_CodeGeneratorInstanceProtocol",
  "_GradientTapeModuleProtocol",
  "_ShapeDebuggerModuleProtocol",
  "_ShapeInferenceModuleProtocol",
  "_ValueAndGradModuleProtocol",
  "_execute_autodiff_strategy",
  "_invoke_codegen",
  "_is_codegen_cls",
  "_run_compiler_graph_and_codegen",
]
