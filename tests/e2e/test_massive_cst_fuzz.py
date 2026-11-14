"""Property-based Hypothesis fuzzing suite for `ASTEngine`, `ImportFixer`, and `GraphExtractor`."""

import ast

import libcst as cst
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.import_fixer import ImportFixer, ImportReq, ResolutionPlan

_NL = chr(10)
_UNARY_OPS = st.sampled_from(["abs", "exp"])
_IDENTIFIERS = st.sampled_from(["x", "y", "z", "hidden", "logits"])
_TARGET_FRAMEWORKS = st.sampled_from(["jax", "numpy", "mlx"])


@given(op_name=_UNARY_OPS, var_name=_IDENTIFIERS, target_fw=_TARGET_FRAMEWORKS)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ast_engine_emits_syntactically_valid_python_and_target_idiom(
  op_name: str,
  var_name: str,
  target_fw: str,
) -> None:
  """Verify `ASTEngine` produces valid Python ASTs and strips source `torch.` calls across fuzzed ops.

  Args:
      op_name: Sampled unary math operation name.
      var_name: Sampled tensor variable identifier.
      target_fw: Sampled target framework identifier.
  """
  source_code = _NL.join(
    [
      "import torch",
      "",
      f"def compute({var_name}):",
      f"  return torch.{op_name}({var_name})",
      "",
    ]
  )
  cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  result = ASTEngine(config=cfg).run(source_code)

  assert result.success is True, f"Transpilation failed with errors: {result.errors}"
  parsed_target = ast.parse(result.code)
  assert any(isinstance(node, ast.FunctionDef) and node.name == "compute" for node in parsed_target.body)
  assert f"torch.{op_name}(" not in result.code


@given(alias_name=st.sampled_from(["jnp", "np", "mx"]), fn_name=_UNARY_OPS)
@settings(max_examples=12, deadline=None)
def test_import_fixer_idempotency_and_reexport_collapse(alias_name: str, fn_name: str) -> None:
  """Verify `ImportFixer` injects required imports, collapses re-exports, and is idempotent on second pass.

  Args:
      alias_name: Sampled framework alias identifier.
      fn_name: Sampled math function name.
  """
  plan = ResolutionPlan(
    required_imports=[ImportReq(module="jax.numpy", alias=alias_name)],
    mappings={"torch": ImportReq(module="jax.numpy", alias=alias_name)},
    path_to_alias={"jax.numpy": alias_name},
    target_framework="jax",
  )
  raw_code = _NL.join(
    [
      "import torch",
      f"res = jax.numpy.{fn_name}(jax.module.Array)",
      "",
    ]
  )
  first_tree = cst.parse_module(raw_code).visit(
    ImportFixer(plan=plan, source_frameworks={"torch"}, used_names={alias_name, "jax", "res"})
  )
  second_tree = cst.parse_module(first_tree.code).visit(
    ImportFixer(plan=plan, source_frameworks={"torch"}, used_names={alias_name, "jax", "res"})
  )

  assert f"import jax.numpy as {alias_name}" in first_tree.code
  assert f"{alias_name}.{fn_name}(" in first_tree.code
  assert "jax.Array" in first_tree.code
  assert "import torch" not in first_tree.code
  assert first_tree.code == second_tree.code


@given(bad_op=st.sampled_from(["nonexistent_xyz_op", "hallucinated_kernel_99"]))
@settings(max_examples=6, deadline=None)
def test_strict_mode_rejects_unmapped_operations_soundly(bad_op: str) -> None:
  """Verify `ASTEngine` in `strict_mode=True` flags unmapped source API calls rather than silently passing.

  Args:
      bad_op: Unmapped synthetic `torch` attribute name.
  """
  source_code = _NL.join(["import torch", f"out = torch.{bad_op}(x)", ""])
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  result = ASTEngine(config=cfg).run(source_code)

  assert result.has_errors or "EscapeHatch" in result.code or not result.success


def test_graph_extractor_preserves_multi_op_chain_invariants() -> None:
  """Verify `GraphExtractor` extracts ordered `LogicalNode` and `LogicalEdge` topology from CST."""
  snippet = _NL.join(
    [
      "import torch.nn as nn",
      "class TinyBlock(nn.Module):",
      "  def __init__(self):",
      "    super().__init__()",
      "    self.fc = nn.Linear(4, 8)",
      "  def forward(self, x):",
      "    h = self.fc(x)",
      "    return h",
      "",
    ]
  )
  extractor = GraphExtractor()
  cst.parse_module(snippet).visit(extractor)
  assert len(extractor.graph.nodes) >= 2
  assert len(extractor.graph.edges) >= 1
  with pytest.raises(cst.ParserSyntaxError):
    cst.parse_module(f"def broken(:{_NL}")
