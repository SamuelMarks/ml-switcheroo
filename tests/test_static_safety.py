"""Unit tests for the StaticSafetyScanner and feasibility boundary detection."""

import libcst as cst
from ml_switcheroo.analysis.static_safety import (
  StaticSafetyCategory,
  StaticSafetyDiagnostic,
  StaticSafetyScanner,
)


def test_static_safety_categories() -> None:
  """Verify StaticSafetyCategory enum values and string representations."""
  assert StaticSafetyCategory.DYNAMIC_SHAPE.value == "DYNAMIC_SHAPE"
  assert StaticSafetyCategory.VALUE_DEPENDENT_CONTROL_FLOW.value == "VALUE_DEPENDENT_CONTROL_FLOW"
  assert StaticSafetyCategory.DATA_DEPENDENT_LOOP.value == "DATA_DEPENDENT_LOOP"
  assert StaticSafetyCategory.DYNAMIC_AUTOGRAD.value == "DYNAMIC_AUTOGRAD"


def test_static_safety_diagnostic_dataclass() -> None:
  """Verify StaticSafetyDiagnostic instantiation and field access."""
  diag = StaticSafetyDiagnostic(
    category=StaticSafetyCategory.DYNAMIC_SHAPE,
    message="Test message",
    recommendation="Test recommendation",
    snippet="x[x > 0]",
  )
  assert diag.category == StaticSafetyCategory.DYNAMIC_SHAPE
  assert diag.message == "Test message"
  assert diag.recommendation == "Test recommendation"
  assert diag.snippet == "x[x > 0]"


def test_detect_dynamic_boolean_masking() -> None:
  """Verify detection of dynamic tensor boolean masking in subscripts."""
  code_cmp = "y = x[x > 0]"
  diags_cmp = StaticSafetyScanner.scan(code_cmp)
  assert len(diags_cmp) == 1
  assert diags_cmp[0].category == StaticSafetyCategory.DYNAMIC_SHAPE

  code_bool = "y = x[(x > 0) & (x < 5)]"
  diags_bool = StaticSafetyScanner.scan(code_bool)
  assert len(diags_bool) >= 1
  assert any(d.category == StaticSafetyCategory.DYNAMIC_SHAPE for d in diags_bool)

  code_normal = "y = x[0]"
  assert len(StaticSafetyScanner.scan(code_normal)) == 0

  code_slice = "y = x[1:5]"
  assert len(StaticSafetyScanner.scan(code_slice)) == 0


def test_detect_value_dependent_control_flow() -> None:
  """Verify detection of value-dependent if branching on tensor contents."""
  code_item = "if x.item() > 0:\n    y = 1"
  diags = StaticSafetyScanner.scan(code_item)
  assert len(diags) == 1
  assert diags[0].category == StaticSafetyCategory.VALUE_DEPENDENT_CONTROL_FLOW

  code_any = "if x.any():\n    y = 1"
  diags_any = StaticSafetyScanner.scan(code_any)
  assert len(diags_any) == 1
  assert diags_any[0].category == StaticSafetyCategory.VALUE_DEPENDENT_CONTROL_FLOW

  code_static = "if is_training:\n    y = 1"
  assert len(StaticSafetyScanner.scan(code_static)) == 0

  code_other_call = "if model.eval():\n    y = 1"
  assert len(StaticSafetyScanner.scan(code_other_call)) == 0

  code_func_call = "if isinstance(x, int):\n    y = 1"
  assert len(StaticSafetyScanner.scan(code_func_call)) == 0


def test_detect_dynamic_while_loop() -> None:
  """Verify detection of dynamic while loops without static trip counts."""
  code_dynamic = "while should_continue:\n    step()"
  diags = StaticSafetyScanner.scan(code_dynamic)
  assert len(diags) == 1
  assert diags[0].category == StaticSafetyCategory.DATA_DEPENDENT_LOOP

  code_static = "while i < 10:\n    i += 1"
  assert len(StaticSafetyScanner.scan(code_static)) == 0

  code_static_left = "while 10 > i:\n    i += 1"
  assert len(StaticSafetyScanner.scan(code_static_left)) == 0


def test_detect_dynamic_autograd() -> None:
  """Verify detection of dynamic autograd invocations in forward execution."""
  code_grad = "grads = torch.autograd.grad(loss, weights)"
  diags = StaticSafetyScanner.scan(code_grad)
  assert len(diags) == 1
  assert diags[0].category == StaticSafetyCategory.DYNAMIC_AUTOGRAD

  code_backward = "loss.backward()"
  diags_bwd = StaticSafetyScanner.scan(code_backward)
  assert len(diags_bwd) == 1
  assert diags_bwd[0].category == StaticSafetyCategory.DYNAMIC_AUTOGRAD

  code_alias = "grads = autograd.grad(loss, weights)"
  diags_alias = StaticSafetyScanner.scan(code_alias)
  assert len(diags_alias) == 1
  assert diags_alias[0].category == StaticSafetyCategory.DYNAMIC_AUTOGRAD

  code_normal = "y = torch.abs(x)"
  assert len(StaticSafetyScanner.scan(code_normal)) == 0


def test_scan_with_cst_module() -> None:
  """Verify scan method works with parsed cst.Module instance."""
  mod = cst.parse_module("y = x[x > 0]")
  diags = StaticSafetyScanner.scan(mod)
  assert len(diags) == 1
  assert diags[0].category == StaticSafetyCategory.DYNAMIC_SHAPE
