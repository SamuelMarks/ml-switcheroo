"""Strongly-typed Pydantic models for on-device forward and backward pass telemetry.

This module defines the structured trace report and node/line telemetry models
produced by `runtime_tracer` when executing models on-device via `zero-*` and
`ml-switcheroo-compiler`.
"""

from pydantic import BaseModel, ConfigDict, Field


class NodeRuntimeTelemetry(BaseModel):
  """Runtime shape, dtype, and backward gradient telemetry for a single operation or line."""

  model_config = ConfigDict(extra="forbid")

  node_id: str = Field(description="Unique identifier of the traced node or statement.")
  var_name: str = Field(description="Assigned variable name or 'return' for return statements.")
  op_type: str = Field(description="Operation or layer type name.")
  lineno: int | None = Field(default=None, description="1-based source line number if available.")
  input_shapes: list[list[int]] = Field(
    default_factory=list,
    description="Concrete input tensor shapes observed during the forward pass.",
  )
  output_shape: list[int] = Field(
    default_factory=list,
    description="Concrete output tensor shape observed during the forward pass.",
  )
  dtype: str = Field(default="float32", description="Data type of the output tensor.")
  grad_shape: list[int] | None = Field(
    default=None,
    description="Concrete gradient shape with respect to this tensor during the backward pass.",
  )
  grad_norm: float | None = Field(
    default=None,
    description="L2 norm of the gradient tensor during the backward pass.",
  )
  is_dynamic: bool = Field(
    default=False,
    description="True if the output shape is data-dependent or unknowable statically.",
  )
  dynamic_reason: str | None = Field(
    default=None,
    description="Explanation of why static analysis could not determine this shape.",
  )


class BranchResolutionRecord(BaseModel):
  """Record of a data-dependent conditional branch resolved during on-device execution."""

  model_config = ConfigDict(extra="forbid")

  lineno: int = Field(description="1-based source line number of the conditional statement.")
  condition_src: str = Field(description="Source expression of the branch condition.")
  taken_branch: str = Field(description="Branch taken at runtime ('true' or 'false').")


class ConstructorDimResolution(BaseModel):
  """Runtime-resolved input dimension for a lazy or unbuilt neural layer constructor."""

  model_config = ConfigDict(extra="forbid")

  layer_name: str = Field(description="Attribute name of the layer on self (e.g. 'fc1').")
  op_type: str = Field(description="Layer class name (e.g. 'LazyLinear', 'Linear', 'Dense', 'Conv2d').")
  inferred_in_features: int = Field(description="Input feature/channel dimension observed on first forward call.")
  out_features: int | None = Field(default=None, description="Output feature/channel dimension if known.")


class DeclaredLayerSpec:
  """Metadata for a layer attribute declared in `__init__` or `setup`."""

  def __init__(
    self,
    layer_name: str,
    op_type: str,
    in_features: int | None,
    out_features: int | None,
    is_lazy: bool,
  ) -> None:
    """Initialize declared layer specification.

    Args:
        layer_name: Attribute name on `self`.
        op_type: Layer class name.
        in_features: Static input feature count if known.
        out_features: Static output feature count if known.
        is_lazy: True if input dimension is deferred to runtime.
    """
    self.layer_name = layer_name
    self.op_type = op_type
    self.in_features = in_features
    self.out_features = out_features
    self.is_lazy = is_lazy


class RuntimeTraceReport(BaseModel):
  """Complete report of an on-device forward and backward execution pass."""

  model_config = ConfigDict(extra="forbid")

  executed: bool = Field(default=False, description="True if on-device execution completed.")
  backend_used: str = Field(default="pyodide", description="Execution backend used ('pyodide', 'webgpu', 'wasm_simd').")
  source_zero_pkg: str = Field(default="", description="Source zero-* runtime package name.")
  target_zero_pkg: str = Field(default="", description="Target zero-* runtime package name.")
  target_supported: bool = Field(default=True, description="True if target framework is supported for emission.")
  input_shapes: list[list[int]] = Field(
    default_factory=list,
    description="Input tensor shapes fed into the forward pass.",
  )
  output_shape: list[int] = Field(
    default_factory=list,
    description="Final output tensor shape produced by the forward pass.",
  )
  loss_value: float | None = Field(
    default=None,
    description="Scalar loss value computed from the forward output.",
  )
  total_grad_norm: float | None = Field(
    default=None,
    description="Global L2 norm across all computed backward gradients.",
  )
  parameter_count: int = Field(
    default=0,
    description="Total number of scalar parameters discovered in the model.",
  )
  finite_check_passed: bool = Field(
    default=True,
    description="True if all forward activations and backward gradients are finite (no NaN/Inf).",
  )
  node_telemetry: dict[str, NodeRuntimeTelemetry] = Field(
    default_factory=dict,
    description="Mapping from node ID to runtime telemetry.",
  )
  line_telemetry: dict[int, NodeRuntimeTelemetry] = Field(
    default_factory=dict,
    description="Mapping from 1-based source line number to runtime telemetry.",
  )
  resolved_constructors: dict[str, ConstructorDimResolution] = Field(
    default_factory=dict,
    description="Mapping from layer attribute name to runtime-resolved input dimensions.",
  )
  resolved_branches: list[BranchResolutionRecord] = Field(
    default_factory=list,
    description="List of data-dependent conditional branches resolved during execution.",
  )
  loop_trip_counts: dict[int, int] = Field(
    default_factory=dict,
    description="Mapping from while-loop line number to executed iteration count.",
  )
  parameter_gradients: dict[str, list[int]] = Field(
    default_factory=dict,
    description="Mapping from parameter name to backward gradient shape.",
  )
  parameter_grad_norms: dict[str, float] = Field(
    default_factory=dict,
    description="Mapping from parameter name to backward gradient L2 norm.",
  )
  shape_table_markdown: str = Field(
    default="",
    description="Formatted Markdown table summarizing tensor shapes and gradients.",
  )
  forward_kernel_code: str | None = Field(
    default=None,
    description="Compiled WebGPU WGSL or WASM SIMD kernel source for the forward pass.",
  )
  backward_kernel_code: str | None = Field(
    default=None,
    description="Compiled WebGPU WGSL or WASM SIMD kernel source for the backward pass.",
  )
  ir_inputs_meta: dict[str, dict[str, str | int | list[int]]] = Field(
    default_factory=dict,
    description="Input tensor buffer metadata for browser WebGPU/WASM runners.",
  )
  ir_outputs_meta: dict[str, dict[str, str | int | list[int]]] = Field(
    default_factory=dict,
    description="Output tensor buffer metadata for browser WebGPU/WASM runners.",
  )
  conversion_mode: str = Field(
    default="preserve_trivia",
    description="Active conversion mode ('preserve_trivia' or 'optimized').",
  )
  opt_level: str = Field(
    default="O2",
    description="Active optimization level ('O0', 'O1', 'O2', 'O3').",
  )
  executed_passes: list[str] = Field(
    default_factory=list,
    description="Compiler transformation passes executed on the shared LogicalGraph.",
  )
  forward_ir_python: str = Field(
    default="",
    description="Serialized ml-switcheroo-ir Python representation of forward_graph.",
  )
  backward_ir_python: str = Field(
    default="",
    description="Serialized ml-switcheroo-ir Python representation of backward_graph.",
  )
  forward_ir_json: str = Field(
    default="",
    description="Serialized JSON representation of forward_graph.",
  )
  backward_ir_json: str = Field(
    default="",
    description="Serialized JSON representation of backward_graph.",
  )
  forward_mermaid: str = Field(
    default="",
    description="Mermaid DAG representation of forward_graph with concrete TensorSpec edge labels.",
  )
  backward_mermaid: str = Field(
    default="",
    description="Mermaid DAG representation of backward_graph with concrete TensorSpec edge labels.",
  )
  synthesized_backward_code: str = Field(
    default="",
    description="Target-framework Python backward/VJP code synthesized from backward_graph.",
  )
  estimated_memory_bytes: float = Field(
    default=0.0,
    description="Estimated total memory footprint in bytes (parameters + activations + gradients).",
  )
  estimated_comm_bytes: float = Field(
    default=0.0,
    description="Estimated collective communication volume in bytes.",
  )
  error: str | None = Field(
    default=None,
    description="Error message if on-device execution encountered a failure.",
  )
