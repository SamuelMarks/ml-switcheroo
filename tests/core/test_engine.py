"""End-to-end and branch unit tests for `ASTEngine` (`ml_switcheroo.core.engine`)."""

import json
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.analysis.static_safety import StaticSafetyScanner
from ml_switcheroo.config import RuntimeConfig, RuntimeConfigOverrides
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.compiler.registry import lift_isa_pair
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.cst.base import CSTNode
from ml_switcheroo.core.dsl import PatternDef
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeAliasSimplifier
from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan
from ml_switcheroo.core.strategies import PipelineStrategy
from ml_switcheroo.core.tracer import TraceEventType, TraceLogger
from ml_switcheroo.exceptions import ConfigurationError
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


class _CustomIsaNode(CSTNode):
  """Minimal `CSTNode` used to verify generic `lift_isa_pair` dispatch."""

  def to_text(self) -> str:
    """Render literal `'nop'`.

    Returns:
        `'nop'` instruction string.
    """
    return "nop"


class _CustomIsaProgram:
  """Parsed ISA container exposing `statements`.

  Attributes:
      statements: Sequence of `_CustomIsaNode` statements.
  """

  def __init__(self, statements: list[_CustomIsaNode]) -> None:
    """Store parsed `statements`.

    Args:
        statements: Parsed custom ISA statements.
    """
    self.statements = statements


class _CustomIsaParser:
  """Minimal ISA parser implementing `IsaParserClassProtocol`."""

  def __init__(self, code: str) -> None:
    """Store raw ISA `code`.

    Args:
        code: Raw ISA source string.
    """
    self.code = code

  def parse(self) -> _CustomIsaProgram:
    """Parse into a single-statement `_CustomIsaProgram`.

    Returns:
        `_CustomIsaProgram` instance.
    """
    return _CustomIsaProgram([_CustomIsaNode()])


class _CustomIsaLifter:
  """Minimal ISA lifter implementing `IsaLifterClassProtocol`."""

  def lift(self, nodes: typing.Sequence[_CustomIsaNode]) -> LogicalGraph:
    """Lift `nodes` into a named `LogicalGraph`.

    Args:
        nodes: Parsed custom ISA nodes.

    Returns:
        `LogicalGraph` named `'custom_<count>'`.
    """
    return LogicalGraph(name=f"custom_{len(nodes)}")


def test_engine_init_helpers_and_validation_report(tmp_path: Path) -> None:
  """Verify `ASTEngine` initialization, `parse`/`to_source`, Mermaid export, and validation reports.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  default_engine = ASTEngine()
  assert isinstance(default_engine.semantics, SemanticsManager)

  report_path = tmp_path / "validation_report.json"
  report_path.write_text(json.dumps({"Abs": True, "Add": False}), encoding="utf-8")

  cfg = RuntimeConfig.load(
    RuntimeConfigOverrides(
      source="torch",
      target="jax",
      strict_mode=True,
      intermediate="python",
      validation_report=report_path,
      plugin_settings={"alpha": 1},
    )
  )
  engine = ASTEngine(config=cfg)
  assert engine.strict_mode is True
  assert engine.source == "torch"
  assert engine.target == "jax"
  assert engine.semantics.is_verified("Abs") is True
  assert engine.semantics.is_verified("Add") is False

  tree = engine.parse("x = 1")
  assert engine.to_source(tree) == "x = 1"
  assert "graph TD" in engine._graph_to_mermaid(tree)

  state = ImportFixerState.create(ResolutionPlan())
  simplifier = AttributeAliasSimplifier(state)
  attr1 = cst.ensure_type(cst.parse_expression("a.b"), cst.Attribute)
  attr2 = cst.ensure_type(cst.parse_expression("a.module.c"), cst.Attribute)
  assert simplifier.rewrite_attribute(attr1, attr1) == attr1
  assert simplifier.simplify_reexports(attr2) == attr2

  custom_graph = lift_isa_pair((_CustomIsaParser, _CustomIsaLifter), "nop")
  assert custom_graph.name == "custom_1"


def test_engine_real_rewriter_pipeline_modes_and_escape_hatches() -> None:
  """Verify real Torch->JAX/Torch AST rewriting across import-fixer, strict, and escape-hatch paths."""
  # Identity roundtrip
  same_res = ASTEngine.from_options(source="torch", target="torch").run(f"x = 5{_NL}print(x)")
  assert same_res.success is True
  assert same_res.code == f"x = 5{_NL}print(x)"
  assert same_res.has_errors is False

  # Real Torch -> JAX transpilation with and without ImportFixer
  res_jax = ASTEngine.from_options(source="torch", target="jax").run(f"import torch{_NL}y = torch.abs(x){_NL}")
  assert res_jax.success is True
  assert "jnp.abs(x)" in res_jax.code

  cfg_no_imp = RuntimeConfig(source_framework="torch", target_framework="jax", enable_import_fixer=False)
  res_no_imp = ASTEngine(config=cfg_no_imp).run(f"import torch{_NL}y = torch.abs(x){_NL}")
  assert res_no_imp.success is True
  assert "jax.numpy.abs(x)" in res_no_imp.code

  # Strict mode clean code vs unmapped Torch op emitting EscapeHatch failure marker + StructuralLinter diagnostics
  cfg_strict = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    strict_mode=True,
    enable_import_fixer=False,
  )
  clean_strict = ASTEngine(config=cfg_strict).run("x = 1")
  assert clean_strict.success is True
  assert clean_strict.errors == []

  strict_escape = ASTEngine.from_options(source="torch", target="jax", strict_mode=True).run(
    f"import torch{_NL}y = torch.nonexistent_magic_op(x){_NL}"
  )
  assert strict_escape.has_errors is True
  assert any("Escape Hatches Detected" in err for err in strict_escape.errors)

  # Top-level exception inside `run` returns failed `ConversionResult`
  err_engine = ASTEngine.from_options(source="torch", target="jax")
  with patch.object(err_engine, "_run_rewriter_pipeline", side_effect=ValueError("Pipeline Boom")):
    failed = err_engine.run("x = 1")
    assert failed.success is False
    assert "Pipeline Boom" in failed.errors[0]


def test_engine_real_mlir_stablehlo_ir_and_intermediate_bridges() -> None:
  """Verify real MLIR, StableHLO, LogicalGraph JSON/Python IR, and multi-hop `intermediate` pipelines."""
  # Unsupported intermediate raises ConfigurationError
  with pytest.raises(ConfigurationError, match="Unsupported intermediate format 'onnx'"):
    ASTEngine.from_options(RuntimeConfigOverrides(source="torch", target="jax", intermediate="onnx"))
  with pytest.raises(ConfigurationError, match="Unsupported intermediate format 'onnx'"):
    RuntimeConfig.load(RuntimeConfigOverrides(source="torch", target="jax", intermediate="onnx"))

  # Direct target="stablehlo" and target="mlir"
  shlo_direct = ASTEngine.from_options(source="torch", target="stablehlo").run(
    f"import torch{_NL}def forward(x: Tensor):{_NL}    return torch.abs(x){_NL}"
  )
  assert shlo_direct.success is True
  assert "stablehlo.abs" in shlo_direct.code

  with patch("ml_switcheroo.core.engine.get_adapter", return_value=None):
    shlo_no_adapter = ASTEngine.from_options(source="torch", target="stablehlo").run("def f(): pass")
    assert shlo_no_adapter.success is True
    assert "func.func" in shlo_no_adapter.code

  mlir_engine = ASTEngine.from_options(source="torch", target="mlir")
  mlir_direct = mlir_engine.run(f"def foo():{_NL}    return 1{_NL}")
  assert mlir_direct.success is True
  assert "func.func" in mlir_direct.code

  mlir_rew = mlir_engine._run_rewriter_pipeline(f"def foo():{_NL}    return 1{_NL}", TraceLogger())
  assert mlir_rew.success is True
  assert "def foo():" in mlir_rew.code

  # Direct MLIR -> JAX and StableHLO -> JAX lifting via `MlirToPythonStrategy`
  lifted_from_mlir = ASTEngine.from_options(source="mlir", target="jax").run(
    f'sw.func {{sym_name = "f"}} {{{_NL}  %0 = sw.constant {{value = 7}}{_NL}  sw.return %0{_NL}}}'
  )
  assert lifted_from_mlir.success is True
  assert "def f():" in lifted_from_mlir.code

  lifted_from_shlo = ASTEngine.from_options(source="stablehlo", target="jax").run(shlo_direct.code)
  assert lifted_from_shlo.success is True

  # Multi-hop intermediate="ir" roundtrip (success and first-hop syntax error propagation)
  res_inter_ir_err = ASTEngine(
    config=RuntimeConfig(source_framework="nvidia_sass", target_framework="jax", intermediate="ir")
  ).run("@@@")
  assert res_inter_ir_err.success is False

  res_inter_ir = ASTEngine(config=RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="ir")).run(
    _NL.join(
      [
        "import torch.nn as nn",
        "class Tiny(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.fc = nn.Linear(4, 2)",
        "    def forward(self, x):",
        "        return self.fc(x)",
        "",
      ]
    )
  )
  assert res_inter_ir.success is True

  # Multi-hop intermediate="mlir", intermediate="stablehlo", and intermediate="ir"
  res_inter_mlir = ASTEngine(
    config=RuntimeConfig(source_framework="torch", target_framework="jax", intermediate="mlir")
  ).run(f"import torch{_NL}def forward(a, b):{_NL}    return torch.add(a, b){_NL}")
  assert res_inter_mlir.success is True
  assert "jnp.add(a, b)" in res_inter_mlir.code or "jax.lax.add(a, b)" in res_inter_mlir.code
  phases = [str(evt["description"]) for evt in res_inter_mlir.trace_events if evt["type"] == TraceEventType.PHASE_START]
  assert "MLIR Bridge" in phases
  assert "Rewriter Pipeline" in phases

  eng_inter_shlo = ASTEngine.from_options(RuntimeConfigOverrides(source="torch", target="jax", intermediate="stablehlo"))
  res_inter_shlo = eng_inter_shlo.run(f"import torch{_NL}def forward(x):{_NL}    return torch.abs(x){_NL}")
  assert res_inter_shlo.success is True
  assert "jnp.abs" in res_inter_shlo.code or "jax.numpy.abs" in res_inter_shlo.code

  with patch(
    "ml_switcheroo.core.strategies.MlirCompilationStrategy.execute",
    return_value=ConversionResult(code="", success=False, errors=["shlo emit failed"]),
  ):
    failed_shlo = eng_inter_shlo.run(f"import torch{_NL}x = torch.abs(y){_NL}")
    assert failed_shlo.success is False
    assert "shlo emit failed" in failed_shlo.errors[0]

  # Target 'ir' (JSON) and 'ml_switcheroo_ir' (Python builder code)
  sample_net = _NL.join(
    [
      "import torch",
      "import torch.nn as nn",
      "class SampleNet(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.linear = nn.Linear(4, 8)",
      "    def forward(self, x):",
      "        return self.linear(x)",
      "",
    ]
  )
  res_ir = ASTEngine(config=RuntimeConfig.load(source="torch", target="ir")).run(sample_net)
  assert res_ir.success is True
  ir_json = json.loads(res_ir.code)
  assert ir_json["name"] == "SampleNet"
  assert any(n["id"] == "linear" for n in ir_json["nodes"])

  res_sw_ir = ASTEngine(config=RuntimeConfig.load(source="torch", target="ml_switcheroo_ir")).run(sample_net)
  assert res_sw_ir.success is True
  assert "import ml_switcheroo_ir as sw_ir" in res_sw_ir.code
  assert "def build_graph() -> sw_ir.LogicalGraph:" in res_sw_ir.code

  # Visual target strategy (`tikz`) and ISA -> StableHLO strategy
  res_tikz = ASTEngine.from_options(source="torch", target="tikz").run(sample_net)
  assert res_tikz.success is True
  assert "tikzpicture" in res_tikz.code

  res_sass_shlo = ASTEngine.from_options(source="nvidia_sass", target="stablehlo").run("FADD R0, R1, R2;")
  assert res_sass_shlo.success is True


def test_engine_real_compiler_isa_graph_opt_sharding_and_safety() -> None:
  """Verify real ISA/compiler pipelines, graph fusion/sharding passes, and static safety diagnostics."""
  # Real SASS <-> RDNA and Python -> SASS/RDNA compiler pipelines
  sass_to_rdna = ASTEngine.from_options(source="nvidia_sass", target="rdna").run("FADD R0, R1, R2;")
  assert sass_to_rdna.success is True
  assert "v_add_f32" in sass_to_rdna.code

  rdna_to_sass = ASTEngine.from_options(source="rdna", target="nvidia_sass").run("v_add_f32 v0, v1, v2")
  assert rdna_to_sass.success is True
  assert "fadd" in rdna_to_sass.code.lower()

  sass_to_py = ASTEngine.from_options(source="nvidia_sass", target="jax").run("FADD R0, R1, R2;")
  assert sass_to_py.success is True
  assert len(sass_to_py.code) > 0

  # MLIR source through compiler pipeline (valid MLIR and fallback to PythonFrontend on invalid MLIR)
  mlir_to_sass = ASTEngine.from_options(source="mlir", target="nvidia_sass").run(
    f'sw.func {{sym_name = "f"}} {{{_NL}  %0 = sw.constant {{value = 1}}{_NL}  sw.return %0{_NL}}}'
  )
  assert mlir_to_sass.success is True

  with patch("ml_switcheroo.core.engine.ingest_code", side_effect=RuntimeError("mlir parse failure")):
    fallback_sass = ASTEngine.from_options(source="mlir", target="nvidia_sass").run("x = 1")
    assert fallback_sass.success is True

  # Compiler pipeline error guards: missing backend and unregistered ISA frontend
  with (
    patch("ml_switcheroo.core.engine.get_backend_class", return_value=None),
    pytest.raises(ValueError, match="No backend found"),
  ):
    ASTEngine.from_options(source="nvidia_sass", target="jax")._run_compiler_pipeline("FADD R0, R1, R2;", TraceLogger())

  with (
    patch("ml_switcheroo.core.engine.is_isa_source", return_value=True),
    pytest.raises(NotImplementedError),
  ):
    ASTEngine.from_options(source="html", target="nvidia_sass")._run_compiler_pipeline("<div/>", TraceLogger())

  from ml_switcheroo.exceptions import TranslationError

  with (
    patch("ml_switcheroo.core.engine.PythonFrontend.parse_to_graph", return_value=None),
    pytest.raises(TranslationError, match="failed to produce a LogicalGraph"),
  ):
    ASTEngine.from_options(source="torch", target="rdna")._run_compiler_pipeline("x = 1", TraceLogger())

  # Non-LogicalGraph optimizer output skips `run_multi_pass_optimizer` cleanly
  with (
    patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize", return_value="non_graph"),
    patch("ml_switcheroo.core.engine.instantiate_backend") as mock_inst_be,
  ):
    fake_be = MagicMock()
    fake_be.compile.return_value = "compiled"
    mock_inst_be.return_value = fake_be
    non_graph_res = ASTEngine(
      config=RuntimeConfig(source_framework="torch", target_framework="rdna", enable_graph_optimization=True)
    )._run_compiler_pipeline("x = 1", TraceLogger())
    assert non_graph_res.code == "compiled"

  # On-device execution across rewriter, compiler, StableHLO emission, and sharded StableHLO source
  from ml_switcheroo.core.runtime_models import RuntimeTraceReport

  executed_report = RuntimeTraceReport(executed=True, loop_trip_counts={1: 4})
  with patch("ml_switcheroo.core.engine.run_on_device_trace", return_value=executed_report):
    assert (
      ASTEngine(config=RuntimeConfig(source_framework="torch", target_framework="jax", enable_on_device_execution=True))
      .run("x = 1")
      .success
      is True
    )
    assert (
      ASTEngine(config=RuntimeConfig(source_framework="torch", target_framework="rdna", enable_on_device_execution=True))
      .run("x = 1")
      .success
      is True
    )

  assert (
    ASTEngine(
      config=RuntimeConfig(source_framework="torch", target_framework="stablehlo", enable_on_device_execution=True)
    )
    .run(f"def f():{_NL}    return 1{_NL}")
    .success
    is True
  )
  assert (
    ASTEngine(
      config=RuntimeConfig(
        source_framework="stablehlo",
        target_framework="jax",
        enable_sharding=True,
        enable_on_device_execution=True,
      )
    )
    .run("module {}")
    .success
    is True
  )

  # Real compiler-pipeline optimization & sharding across fusion and non-fusion targets
  model_src = _NL.join(
    [
      "import torch.nn as nn",
      "class M(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.conv = nn.Conv2d(3, 8, 3)",
      "        self.relu = nn.ReLU()",
      "    def forward(self, x):",
      "        return self.relu(self.conv(x))",
      "",
    ]
  )
  for src_fw, tgt_fw in (("torch", "jax"), ("torch", "flax_nnx"), ("jax", "torch")):
    cfg_opt = RuntimeConfig(
      source_framework=src_fw,
      target_framework=tgt_fw,
      enable_graph_optimization=True,
      enable_sharding=True,
    )
    opt_res = ASTEngine(config=cfg_opt).run(model_src)
    assert opt_res.success is True

  # Real rewriter-pipeline graph optimization (empty module, no-diff module, real fusion pattern, sharding, exception)
  rew_opt_empty = ASTEngine.from_options(source="torch", target="jax", enable_graph_optimization=True).run("")
  assert rew_opt_empty.success is True

  rew_opt_nodiff = ASTEngine.from_options(source="torch", target="jax", enable_graph_optimization=True).run(model_src)
  assert rew_opt_nodiff.success is True

  sem_fused = SemanticsManager()
  with patch.object(
    sem_fused,
    "get_patterns",
    return_value=[PatternDef(name="ConvReLU", sequence=["Conv2d", "ReLU"], replace_with="FusedConvReLU")],
  ):
    cfg_rew_fused = RuntimeConfig(
      source_framework="torch",
      target_framework="jax",
      enable_graph_optimization=True,
      strict_mode=True,
    )
    rew_fused = ASTEngine(config=cfg_rew_fused, semantics=sem_fused)._run_rewriter_pipeline(model_src, TraceLogger())
    assert rew_fused.success is True

    cfg_rew_shard = RuntimeConfig(
      source_framework="torch",
      target_framework="jax",
      enable_graph_optimization=True,
      enable_sharding=True,
    )
    rew_shard = ASTEngine(config=cfg_rew_shard, semantics=sem_fused)._run_rewriter_pipeline(model_src, TraceLogger())
    assert rew_shard.success is True

    cfg_rew_torch = RuntimeConfig(
      source_framework="torch",
      target_framework="torch",
      enable_graph_optimization=True,
      enable_sharding=True,
    )
    rew_torch = ASTEngine(config=cfg_rew_torch, semantics=sem_fused)._run_rewriter_pipeline(model_src, TraceLogger())
    assert rew_torch.success is True

  with patch("ml_switcheroo.core.engine.GraphExtractor", side_effect=RuntimeError("extract boom")):
    rew_exc = ASTEngine.from_options(source="torch", target="jax", enable_graph_optimization=True).run("x = 1")
    assert rew_exc.success is True

  # `_is_sharding_fusion_target` with empty framework config
  eng_check = ASTEngine.from_options(source="torch", target="jax")
  with patch.object(eng_check.semantics, "get_framework_config", return_value={}):
    eng_check.target = "custom_non_fusion"
    assert eng_check._is_sharding_fusion_target() is False
  eng_check.target = "jax"

  # Static safety warnings and scanner exception resilience on both compiler and rewriter pipelines
  dyn_code = f"import torch{_NL}def f(x):{_NL}    if x.item() > 0:{_NL}        return x{_NL}"
  comp_warn = ASTEngine(config=RuntimeConfig.load(source="torch", target="rdna")).run(dyn_code)
  assert any("VALUE_DEPENDENT_CONTROL_FLOW" in w for w in comp_warn.warnings)

  rew_warn = ASTEngine(config=RuntimeConfig.load(source="torch", target="jax")).run(dyn_code)
  assert any("VALUE_DEPENDENT_CONTROL_FLOW" in w for w in rew_warn.warnings)

  with patch.object(StaticSafetyScanner, "scan", side_effect=RuntimeError("scanner error")):
    assert ASTEngine(config=RuntimeConfig.load(source="torch", target="rdna")).run("x = 1").success is True
    assert ASTEngine(config=RuntimeConfig.load(source="torch", target="jax")).run("x = 1").success is True

  # `PipelineStrategy` abstract methods and empty `STRATEGY_REGISTRY`
  base_strat = PipelineStrategy()
  with pytest.raises(NotImplementedError):
    base_strat.can_handle(eng_check)
  with pytest.raises(NotImplementedError):
    base_strat.execute(eng_check, "x = 1", MagicMock(spec=TraceLogger))

  with patch("ml_switcheroo.core.strategies.STRATEGY_REGISTRY", []):
    empty_strat_res = eng_check.run("x = 1")
    assert empty_strat_res.success is False
