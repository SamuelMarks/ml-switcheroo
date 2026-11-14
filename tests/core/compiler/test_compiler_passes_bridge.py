"""Unit tests for `src/ml_switcheroo/core/compiler/compiler_passes.py`."""

from __future__ import annotations

import types
import typing
from pathlib import Path
from unittest import mock

import libcst as cst
import pytest
from ml_switcheroo_ir import DType, LogicalEdge, LogicalGraph, LogicalNode

import ml_switcheroo as mls
from ml_switcheroo.config import ConversionMode, OptimizationLevel
from ml_switcheroo.core.compiler.compiler_passes import (
  CSTConstantFolder,
  _invoke_graph_pass,
  _load_compiler_passes_module,
  _load_pass_manager_module,
  ensure_compiler_and_zero_on_path,
  estimate_graph_telemetry,
  run_algebraic_simplification_passes,
  run_canonicalization_passes,
  run_fusion_and_scheduling_passes,
  run_multi_pass_optimizer,
)
from ml_switcheroo.core.tracer import TraceLogger


def _build_sample_graph() -> LogicalGraph:
  """Construct a sample LogicalGraph with constant folding and CSE opportunities.

  Returns:
      Populated `LogicalGraph`.
  """
  g = LogicalGraph(name="SampleNet")
  g.nodes["in_x"] = LogicalNode(id="in_x", op_type="Input", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes={"value": 2}, shape_metadata=(1,))
  g.nodes["c2"] = LogicalNode(id="c2", op_type="Constant", attributes={"value": 3}, shape_metadata=(1,))
  g.nodes["add1"] = LogicalNode(id="add1", op_type="Add", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["add2"] = LogicalNode(id="add2", op_type="Add", shape_metadata=(2, 4), dtype=DType.float32)
  g.nodes["dead_node"] = LogicalNode(id="dead_node", op_type="Relu", shape_metadata=(2, 4))
  g.nodes["out"] = LogicalNode(id="out", op_type="Output", shape_metadata=(2, 4), dtype=DType.float32)
  g.edges.extend(
    [
      LogicalEdge(source="in_x", target="add1"),
      LogicalEdge(source="c1", target="add1"),
      LogicalEdge(source="in_x", target="add2"),
      LogicalEdge(source="c1", target="add2"),
      LogicalEdge(source="add1", target="out"),
      LogicalEdge(source="c2", target="dead_node"),
    ]
  )
  g.inputs = ["in_x"]
  g.outputs = ["out"]
  return g


def test_ensure_compiler_and_zero_on_path(tmp_path: Path) -> None:
  """Verify sibling repository src directories are added to sys.path when present.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  comp_src = tmp_path / "ml-switcheroo-compiler" / "src"
  comp_src.mkdir(parents=True)
  added = ensure_compiler_and_zero_on_path(workspace_parent=tmp_path)
  assert str(comp_src) in added

  # Calling a second time should not re-add the same path
  added_again = ensure_compiler_and_zero_on_path(workspace_parent=tmp_path)
  assert added_again == []


def test_load_modules_and_invoke_graph_pass_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify module loader fallbacks and _invoke_graph_pass error handling.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  g = _build_sample_graph()
  dummy_mod = types.ModuleType("dummy_passes")

  # Missing attribute returns False
  assert _invoke_graph_pass(dummy_mod, "nonexistent_pass", g) is False

  # Raising pass returns False
  def _bad_pass(graph: LogicalGraph) -> None:
    """Raise RuntimeError for testing.

    Args:
        graph: Input graph.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("boom")

  monkeypatch.setattr(dummy_mod, "bad_pass", _bad_pass, raising=False)
  assert _invoke_graph_pass(dummy_mod, "bad_pass", g) is False

  def _good_pass(graph: LogicalGraph) -> None:
    """No-op pass that succeeds.

    Args:
        graph: Input graph.
    """

  monkeypatch.setattr(dummy_mod, "good_pass", _good_pass, raising=False)
  assert _invoke_graph_pass(dummy_mod, "good_pass", g) is True

  stub_passes = types.ModuleType("ml_switcheroo_compiler.transforms.passes")
  stub_pm = types.ModuleType("ml_switcheroo_compiler.transforms.pass_manager")
  with mock.patch("importlib.import_module", side_effect=lambda name: stub_passes if "passes" in name else stub_pm):
    assert _load_compiler_passes_module() is stub_passes
    assert _load_pass_manager_module() is stub_pm

  with mock.patch("importlib.import_module", side_effect=ImportError("missing")):
    assert _load_compiler_passes_module() is None
    assert _load_pass_manager_module() is None
    assert run_canonicalization_passes(g) == ["propagate_shapes_and_constants"]
    assert run_algebraic_simplification_passes(g, include_cse_and_bn=True) == [
      "propagate_shapes_and_constants",
      "eliminate_common_subexpressions",
      "eliminate_dead_nodes",
    ]
    assert run_fusion_and_scheduling_passes(g, include_aggressive=True) == []


def test_run_multi_pass_optimizer_all_levels(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify run_multi_pass_optimizer across O0, O1, O2, and O3 with and without TraceLogger.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  tracer = TraceLogger()
  stub_passes_mod = types.ModuleType("stub_passes")
  for p_name in (
    "polyfill_lowering_pass",
    "type_promotion_explicitizer_pass",
    "broadcast_explicitizer_pass",
    "shape_inference_pass",
    "dtype_inference_pass",
    "constant_folding_pass",
    "common_subexpression_elimination_pass",
    "batch_norm_folding_pass",
    "control_flow_optimization_pass",
    "dead_code_elimination_pass",
    "operator_fusion_pass",
    "graph_scheduling_pass",
    "buffer_allocation_pass",
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
  ):
    monkeypatch.setattr(stub_passes_mod, p_name, lambda graph: None, raising=False)

  stub_pm_mod = types.ModuleType("stub_pm")

  class _WorkingPM:
    """Stub PassManager that converges cleanly."""

    def add_pass(self, fn: typing.Callable[[LogicalGraph], None], name: str | None = None) -> None:
      """Record pass function.

      Args:
          fn: Pass function.
          name: Pass name.
      """

    def run_until_converged(self, graph: LogicalGraph, max_iterations: int = 5) -> LogicalGraph:
      """Return graph unchanged.

      Args:
          graph: Input graph.
          max_iterations: Iteration limit.

      Returns:
          Converged graph.
      """
      return graph

  monkeypatch.setattr(stub_pm_mod, "PassManager", _WorkingPM, raising=False)

  with (
    mock.patch(
      "ml_switcheroo.core.compiler.compiler_passes._load_compiler_passes_module",
      return_value=stub_passes_mod,
    ),
    mock.patch(
      "ml_switcheroo.core.compiler.compiler_passes._load_pass_manager_module",
      return_value=stub_pm_mod,
    ),
  ):
    for level in (
      OptimizationLevel.O0,
      OptimizationLevel.O1,
      OptimizationLevel.O2,
      OptimizationLevel.O3,
    ):
      g = _build_sample_graph()
      opt_g, passes = run_multi_pass_optimizer(g, opt_level=level, tracer=tracer)
      assert isinstance(opt_g, LogicalGraph)
      assert "propagate_shapes_and_constants" in passes
      if level in (OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3):
        assert "eliminate_dead_nodes" in passes
      if level in (OptimizationLevel.O2, OptimizationLevel.O3):
        assert "eliminate_common_subexpressions" in passes
        assert "PassManager.run_until_converged" in passes

  # Also test without tracer and when PassManager raises an exception
  g2 = _build_sample_graph()
  bad_pm_mod = types.ModuleType("bad_pm")

  class BrokenPM:
    """Mock PassManager that raises on add_pass."""

    def add_pass(self, fn: typing.Callable[[LogicalGraph], None], name: str | None = None) -> None:
      """Raise RuntimeError when adding pass.

      Args:
          fn: Pass function.
          name: Pass name.

      Raises:
          RuntimeError: Always.
      """
      raise RuntimeError("pm failure")

  monkeypatch.setattr(bad_pm_mod, "PassManager", BrokenPM, raising=False)
  with (
    mock.patch(
      "ml_switcheroo.core.compiler.compiler_passes._load_compiler_passes_module",
      return_value=stub_passes_mod,
    ),
    mock.patch(
      "ml_switcheroo.core.compiler.compiler_passes._load_pass_manager_module",
      return_value=bad_pm_mod,
    ),
  ):
    _, passes_no_pm = run_multi_pass_optimizer(g2, opt_level=OptimizationLevel.O2, tracer=None)
    assert "PassManager.run_until_converged" not in passes_no_pm

  with mock.patch(
    "ml_switcheroo.core.compiler.compiler_passes._load_pass_manager_module",
    return_value=None,
  ):
    _, passes_none_pm = run_multi_pass_optimizer(_build_sample_graph(), opt_level=OptimizationLevel.O2, tracer=None)
    assert "PassManager.run_until_converged" not in passes_none_pm

  # Graph without explicit outputs skips DCE
  g_no_out = _build_sample_graph()
  g_no_out.outputs = []
  _, passes_no_out = run_multi_pass_optimizer(g_no_out, opt_level=OptimizationLevel.O2, tracer=None)
  assert "eliminate_dead_nodes" not in passes_no_out


def test_estimate_graph_telemetry() -> None:
  """Verify estimate_graph_telemetry returns memory and communication byte estimates."""
  g = _build_sample_graph()
  g.nodes["no_shape"] = LogicalNode(id="no_shape", op_type="Identity")
  mem_bytes, comm_bytes = estimate_graph_telemetry(g)
  assert mem_bytes > 0.0
  assert comm_bytes >= 0.0

  g_with_attr = LogicalGraph(name="AttrNet")
  g_with_attr.nodes["p"] = LogicalNode(id="p", op_type="Parameter", attributes={"shape": [4, 4], "dtype": "float32"})
  mem_attr, comm_attr = estimate_graph_telemetry(g_with_attr)
  assert mem_attr > 0.0
  assert comm_attr >= 0.0


def test_cst_constant_folder_expressions() -> None:
  """Verify CSTConstantFolder folds int, float, negative, and binary operations accurately."""
  code = """
a = 2 + 3
b = 10 - 14
c = 4 * 5
d = 9 // 2
e = 5.0 / 2.0
f = -1.5 + -2.5
g = x + (2 * 3)
h = 10 // 0
i = 10 % 3
"""
  tree = cst.parse_module(code)
  folded = tree.visit(CSTConstantFolder()).code
  assert "a = 5" in folded
  assert "b = -4" in folded
  assert "c = 20" in folded
  assert "d = 4" in folded
  assert "e = 2.5" in folded
  assert "f = -4.0" in folded
  assert "g = x + 6" in folded
  assert "h = 10 // 0" in folded
  assert "i = 10 % 3" in folded

  # Non-numeric expressions return None safely
  assert CSTConstantFolder._extract_numeric_literal(cst.Name("x")) is None
  assert (
    CSTConstantFolder._extract_numeric_literal(cst.UnaryOperation(operator=cst.Plus(), expression=cst.Integer("1")))
    is None
  )


def test_preserve_trivia_vs_optimized_mode_end_to_end() -> None:
  """Verify ConversionMode.PRESERVE_TRIVIA keeps 1:1 trivia while ConversionMode.OPTIMIZED folds constants."""
  source_code = """# Top-level file comment

import torch


def compute(x):
    # Keep this inline comment and blank line above
    scale = 2 * 3
    return torch.abs(x) + scale
"""
  preserved = mls.convert(
    source_code,
    source="torch",
    target="jax",
    options=mls.ConvertOptions(mode=ConversionMode.PRESERVE_TRIVIA),
  )
  assert "# Top-level file comment" in preserved
  assert "# Keep this inline comment and blank line above" in preserved
  assert "scale = 2 * 3" in preserved

  optimized = mls.convert(
    source_code,
    source="torch",
    target="jax",
    options=mls.ConvertOptions(
      mode=ConversionMode.OPTIMIZED,
      opt_level=OptimizationLevel.O2,
    ),
  )
  assert "scale = 6" in optimized
