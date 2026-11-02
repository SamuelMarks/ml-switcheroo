"""Unit tests for runtime configuration and telemetry models."""

import pytest
from pydantic import ValidationError

from ml_switcheroo.config import (
  ExecutionBackend,
  LossReduction,
  RuntimeConfig,
  parse_input_shapes_string,
)
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.runtime_models import (
  BranchResolutionRecord,
  ConstructorDimResolution,
  NodeRuntimeTelemetry,
  RuntimeTraceReport,
)


def test_parse_input_shapes_string_cases() -> None:
  """Verify parse_input_shapes_string handles empty, single, multi, and invalid inputs."""
  assert parse_input_shapes_string("") is None
  assert parse_input_shapes_string("   ") is None
  assert parse_input_shapes_string("auto") is None
  assert parse_input_shapes_string("AUTO") is None
  assert parse_input_shapes_string(";; () ;") is None
  assert parse_input_shapes_string("( , )") is None
  assert parse_input_shapes_string("2, 10") == [(2, 10)]
  assert parse_input_shapes_string("(2, 3, 8, 8); (2, 10)") == [(2, 3, 8, 8), (2, 10)]
  assert parse_input_shapes_string("[4, 16, ]") == [(4, 16)]
  assert parse_input_shapes_string("2, abc") is None
  assert parse_input_shapes_string("2, 0") is None
  assert parse_input_shapes_string("-1, 4") is None


def test_runtime_config_on_device_fields() -> None:
  """Verify RuntimeConfig defaults and custom values for on-device execution."""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    enable_on_device_execution=True,
    execution_backend=ExecutionBackend.WEBGPU,
    input_shapes=[(2, 8)],
    input_dtype="float16",
    random_seed=7,
    loss_reduction=LossReduction.MEAN,
    inject_runtime_annotations=False,
    inject_shape_assertions=True,
    observed_runtime_shapes={"x": [2, 8]},
  )
  assert cfg.enable_on_device_execution is True
  assert cfg.execution_backend == ExecutionBackend.WEBGPU
  assert cfg.input_shapes == [(2, 8)]
  assert cfg.input_dtype == "float16"
  assert cfg.random_seed == 7
  assert cfg.loss_reduction == LossReduction.MEAN
  assert cfg.inject_runtime_annotations is False
  assert cfg.inject_shape_assertions is True
  assert cfg.observed_runtime_shapes == {"x": [2, 8]}


def test_runtime_telemetry_models_and_conversion_result() -> None:
  """Verify NodeRuntimeTelemetry, BranchResolutionRecord, ConstructorDimResolution, and RuntimeTraceReport."""
  node_tel = NodeRuntimeTelemetry(
    node_id="n0",
    var_name="x",
    op_type="Linear",
    lineno=12,
    input_shapes=[[2, 16]],
    output_shape=[2, 10],
    dtype="float32",
    grad_shape=[2, 10],
    grad_norm=1.25,
    is_dynamic=False,
  )
  branch_rec = BranchResolutionRecord(lineno=15, condition_src="x.sum() > 0", taken_branch="true")
  ctor_res = ConstructorDimResolution(
    layer_name="fc1",
    op_type="LazyLinear",
    inferred_in_features=16,
    out_features=10,
  )
  report = RuntimeTraceReport(
    executed=True,
    backend_used="pyodide",
    source_zero_pkg="zero_torch",
    target_zero_pkg="zero_flax",
    input_shapes=[[2, 16]],
    output_shape=[2, 10],
    loss_value=3.5,
    total_grad_norm=2.0,
    parameter_count=170,
    finite_check_passed=True,
    node_telemetry={"n0": node_tel},
    line_telemetry={12: node_tel},
    resolved_constructors={"fc1": ctor_res},
    resolved_branches=[branch_rec],
    loop_trip_counts={20: 3},
    parameter_gradients={"fc1.weight": [10, 16]},
    parameter_grad_norms={"fc1.weight": 1.5},
    shape_table_markdown="| Node | Shape |",
  )

  res = ConversionResult(code="pass", success=True, runtime_trace=report)
  assert res.runtime_trace is not None
  assert res.runtime_trace.executed is True
  assert res.runtime_trace.resolved_constructors["fc1"].inferred_in_features == 16
  assert res.runtime_trace.resolved_branches[0].taken_branch == "true"

  with pytest.raises(ValidationError):
    NodeRuntimeTelemetry.model_validate({"node_id": "n0", "var_name": "x", "op_type": "Add", "extra_field": 1})
