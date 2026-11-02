"""Unit tests for Phase 20: Optional Data-Dependent `if` Branch Specialization (`specialize_dynamic_branches`).

Verifies:
- `RuntimeConfig.specialize_dynamic_branches = True` replaces tensor-conditioned `if/else` statements with the branch taken on-device.
- Leading comment `# [on-device branch specialized: cond=<condition_src> evaluated to <taken_branch>]` is prepended.
- Untaken branch is preserved as `# [untaken] ...` comments, and comments inside the taken branch are preserved.
- Covers `"true"` branch, `"false"` branch, `elif` chains, and `else`-less `if` statements at 100% coverage.
"""

from __future__ import annotations

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.runtime_annotations import RuntimeAnnotationPass
from ml_switcheroo.core.runtime_models import BranchResolutionRecord, RuntimeTraceReport
from ml_switcheroo.semantics.manager import SemanticsManager


def test_specialize_true_branch_with_else_and_trivia() -> None:
  """Verify specializing a `'true'` branch preserves taken statements, inner trivia, and comments out the `else` branch."""
  code = """import torch

def forward(x):
    # Pre-if user comment
    if x.sum() > -100.0:  # inline if header comment
        # Inside true branch comment
        y = torch.sin(x)
    else:
        y = torch.cos(x)
    return y
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    specialize_dynamic_branches=True,
    input_shapes=[(2, 4)],
  )
  res = ASTEngine(semantics=SemanticsManager(), config=cfg).run(code)
  assert res.success
  assert "# Pre-if user comment" in res.code
  assert "# [on-device branch specialized: cond=x.sum() > -100.0 evaluated to true]" in res.code
  assert "# inline if header comment" in res.code
  assert "# [untaken] else:" in res.code
  assert "# Inside true branch comment" in res.code
  assert "y = jnp.sin(x" in res.code
  assert "    if " not in res.code.split("def backward_fn")[0]


def test_specialize_false_branch_with_else_and_elif() -> None:
  """Verify specializing `'false'` branches across `else`, `elif`, and `else`-less `if` statements."""
  # 1. False branch with `else` header comment
  code_else = """import torch

def forward(x):
    if x.sum() > 9999.0:
        y = torch.sin(x)
    else:  # inline else header comment
        # Inside false branch comment
        y = torch.cos(x)
    return y
"""
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    enable_on_device_execution=True,
    inject_runtime_annotations=True,
    specialize_dynamic_branches=True,
    input_shapes=[(2, 4)],
  )
  res_else = ASTEngine(semantics=SemanticsManager(), config=cfg).run(code_else)
  assert res_else.success
  assert "# [on-device branch specialized: cond=x.sum() > 9999.0 evaluated to false]" in res_else.code
  assert "# inline else header comment" in res_else.code
  assert "# [untaken] y = " in res_else.code
  assert "sin(x)" in res_else.code
  assert "# Inside false branch comment" in res_else.code
  assert "y = jnp.cos(x" in res_else.code

  # 2. `elif` chain where first condition is false and `elif` condition is true, plus `else` without header comment
  sem = SemanticsManager()
  ctx_else_no_comment = RewriterContext(
    semantics=sem,
    config=cfg,
    runtime_trace=RuntimeTraceReport(
      executed=True,
      resolved_branches=[
        BranchResolutionRecord(lineno=2, condition_src="x.sum() > 100.0", taken_branch="false"),
      ],
    ),
  )
  mod_else_no_comment = cst.parse_module(
    """def forward(x):
    if x.sum() > 100.0:
        y = x + 1
    else:
        y = x + 2
    return y
"""
  )
  out_else_no_comment = RuntimeAnnotationPass().transform(mod_else_no_comment, ctx_else_no_comment).code
  assert "y = x + 2" in out_else_no_comment

  ctx_elif = RewriterContext(
    semantics=sem,
    config=cfg,
    runtime_trace=RuntimeTraceReport(
      executed=True,
      resolved_branches=[
        BranchResolutionRecord(lineno=2, condition_src="x.sum() > 100.0", taken_branch="false"),
        BranchResolutionRecord(lineno=4, condition_src="x.sum() > -100.0", taken_branch="true"),
      ],
    ),
  )
  mod_elif = cst.parse_module(
    """def forward(x):
    if x.sum() > 100.0:
        y = x + 1
    elif x.sum() > -100.0:
        # Elif taken comment
        y = x + 2
    else:
        y = x + 3
    return y
"""
  )
  out_elif = RuntimeAnnotationPass().transform(mod_elif, ctx_elif).code
  assert "# [on-device branch specialized: cond=x.sum() > 100.0 evaluated to false]" in out_elif
  assert "# [on-device branch specialized: cond=x.sum() > -100.0 evaluated to true]" in out_elif
  assert "# Elif taken comment" in out_elif
  assert "y = x + 2" in out_elif

  # 3. `else`-less `if` evaluating to `"true"` and `"false"`
  ctx_true_no_else = RewriterContext(
    semantics=sem,
    config=cfg,
    runtime_trace=RuntimeTraceReport(
      executed=True,
      resolved_branches=[
        BranchResolutionRecord(lineno=2, condition_src="x.sum() > -100.0", taken_branch="true"),
      ],
    ),
  )
  mod_true_no_else = cst.parse_module(
    """def forward(x):
    if x.sum() > -100.0:
        x = x + 1
    return x
"""
  )
  out_true_no_else = RuntimeAnnotationPass().transform(mod_true_no_else, ctx_true_no_else).code
  assert "# [on-device branch specialized: cond=x.sum() > -100.0 evaluated to true]" in out_true_no_else
  assert "x = x + 1" in out_true_no_else

  ctx_no_else = RewriterContext(
    semantics=sem,
    config=cfg,
    runtime_trace=RuntimeTraceReport(
      executed=True,
      resolved_branches=[
        BranchResolutionRecord(lineno=2, condition_src="x.sum() > 100.0", taken_branch="false"),
      ],
    ),
  )
  mod_no_else = cst.parse_module(
    """def forward(x):
    if x.sum() > 100.0:
        x = x + 1
    return x
"""
  )
  out_no_else = RuntimeAnnotationPass().transform(mod_no_else, ctx_no_else).code
  assert "# [on-device branch specialized: cond=x.sum() > 100.0 evaluated to false]" in out_no_else
  assert "# [untaken] x = x + 1" in out_no_else
  assert "pass" in out_no_else

  # 4. Module-level `if` (outside forward) and exhausted resolved_branches
  ctx_exhausted = RewriterContext(
    semantics=sem,
    config=cfg,
    runtime_trace=RuntimeTraceReport(executed=True, resolved_branches=[]),
  )
  mod_top = cst.parse_module(
    """if True:
    a = 1
def forward(x):
    if x.sum() > 0:
        x = x + 1
    return x
"""
  )
  out_top = RuntimeAnnotationPass().transform(mod_top, ctx_exhausted).code
  assert "if x.sum() > 0:" in out_top
