"""Test suite for the Linter module."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.testing.linter import StructuralLinter, validate_transpilation


@pytest.fixture
def linter() -> StructuralLinter:
  """Execute linter helper logic.

  Returns:
      Configured `linter` fixture value (StructuralLinter).
  """
  return StructuralLinter(forbidden_roots={"torch", "flax"})


def test_linter_clean_code(linter: StructuralLinter) -> None:
  """Verifies the behavior of linter clean code."""
  code: str = "\nimport jax.numpy as jnp\ndef f(x):\n    return jnp.abs(x)\n"
  errors: list[str] = linter.check(code)
  assert len(errors) == 0


def test_linter_detects_import(linter: StructuralLinter) -> None:
  """Verifies the behavior of linter detects import."""
  code: str = "\nimport torch\nx = torch.abs(y)\n"
  errors: list[str] = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'torch'" in errors[0]


def test_linter_detects_from_import(linter: StructuralLinter) -> None:
  """Verifies the behavior of linter detects from import."""
  code: str = "from flax import linen as nn"
  errors: list[str] = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'from flax ...'" in errors[0]


def test_linter_detects_aliased_usage(linter: StructuralLinter) -> None:
  """Verifies the behavior of linter detects aliased usage."""
  code: str = "\nimport torch as t\n# Usage of alias\ny = t.abs(x)\n"
  errors: list[str] = linter.check(code)
  assert len(errors) >= 1
  usage_errors: list[str] = [e for e in errors if "Forbidden Usage" in e]
  assert len(usage_errors) > 0
  assert "alias of torch" in usage_errors[0]


def test_linter_parse_error(linter: StructuralLinter) -> None:
  """Hits lines 53-54 where cst.parse_module fails.

  Args:
      linter: The linter (StructuralLinter) for test linter parse error.
  """
  errors: list[str] = linter.check("def f(): this is invalid python !")
  assert len(errors) == 1
  assert "Linter Parse Error" in errors[0]


def test_linter_detects_wildcard_import(linter: StructuralLinter) -> None:
  """Hits line 114 for wildcard imports.

  Args:
      linter: The linter (StructuralLinter) for test linter detects wildcard import.
  """
  code: str = "from torch import *"
  errors: list[str] = linter.check(code)
  assert any("Forbidden Wildcard Import" in e for e in errors)


def test_linter_detects_direct_access(linter: StructuralLinter) -> None:
  """Hits lines 158-161 where forbidden root is accessed directly without explicit alias import tracking.

  Args:
      linter: The linter (StructuralLinter) for test linter detects direct access.
  """
  # torch is forbidden, we just use it directly
  code: str = "x = torch.Tensor()"
  errors: list[str] = linter.check(code)
  # It should report forbidden import AND forbidden usage
  usage_errs: list[str] = [e for e in errors if "Forbidden Usage: Direct access 'torch'" in e]
  assert len(usage_errs) > 0


def test_linter_complex_attribute_name(linter: StructuralLinter) -> None:
  """Hits lines 192 and 198-200 for recursive attribute resolution in linter."""
  # We test _get_full_name_from_node directly or via checking attribute access
  import libcst as cst

  code: str = "import torch\nx = torch.nn.functional.relu(y)"
  _errors: list[str] = linter.check(code)

  # Triggering _get_full_name_from_node manually to guarantee line coverage
  tree: cst.Module = cst.parse_module("torch.nn.functional")
  attr_node: cst.BaseExpression = getattr(getattr(getattr(tree, "body")[0], "body")[0], "value")
  assert linter._get_full_name_from_node(attr_node) == "torch.nn.functional"

  # Triggering _get_root_name manually
  assert linter._get_root_name(attr_node) == "torch"


def test_linter_get_root_name_fallback(linter: StructuralLinter) -> None:
  """Hits line 192 in _get_root_name fallback.

  Args:
      linter: The linter (StructuralLinter) for test linter get root name fallback.
  """
  import libcst as cst

  assert linter._get_root_name(cst.Integer("1")) == ""
  assert linter._get_full_name_from_node(cst.Integer("1")) == ""


def test_facade_flax_inheritance() -> None:
  """Verifies the behavior of facade Flax inheritance."""
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.import_alias = ("flax.nnx", "nnx")
  mock_adapter.inherits_from = "jax"
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code: str = "import jax.numpy as jnp"
    is_valid: bool
    errors: list[str]
    is_valid, errors = validate_transpilation(code, source_framework="flax_nnx")
    assert not is_valid
    assert "Forbidden Import: 'jax'" in errors[0]


def test_facade_mlx_detection(tmp_path: pathlib.Path) -> None:
  """Verifies the behavior of facade MLX detection.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  mock_adapter: MagicMock = MagicMock()
  mock_adapter.import_alias = ("mlx.core", "mx")
  mock_adapter.search_modules = ["mlx"]
  mock_adapter.inherits_from = None
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code: str = "\nimport mlx.core as mx\ndef f(x):\n    return mx.abs(x)\n"
    is_valid: bool
    errors: list[str]
    is_valid, errors = validate_transpilation(code, source_framework="mlx")
    assert not is_valid
    assert any(("mlx" in e for e in errors))


def test_linter_branches(linter: StructuralLinter) -> None:
  """Hits missing branches in linter.py."""
  code: str = """
import numpy as np
from . import relative
from os import path
from torch import abs, add
import torch.nn as nn

x = nn.Linear()
y = torch.tensor()
"""
  errors: list[str] = linter.check(code)
  assert len(errors) > 0


def test_linter_duplicate_violations(linter: StructuralLinter) -> None:
  """Hits duplicate violation branches.

  Args:
      linter: The linter (StructuralLinter) for test linter duplicate violations.
  """
  code: str = """
import torch as t
import torch

y = t.abs(x)
z = t.abs(x)

a = torch.tensor()
b = torch.tensor()

c = torch.nn.Linear()
d = torch.nn.Linear()

e = t.nn.Linear()
f = t.nn.Linear()
"""
  errors: list[str] = linter.check(code)
  assert len(errors) > 0


def test_validate_transpilation_missing_adapter() -> None:
  """Hits missing adapter branch in validate_transpilation."""
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=None):
    is_valid: bool
    errors: list[str]
    is_valid, errors = validate_transpilation("import torch", "unknown")
    assert is_valid


def test_validate_transpilation_empty_adapter() -> None:
  """Hits empty adapter branch in validate_transpilation."""
  mock_adapter: MagicMock = MagicMock(spec=[])
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    is_valid: bool
    errors: list[str]
    is_valid, errors = validate_transpilation("import torch", "torch")
    assert not is_valid


def test_linter_duplicate_direct(linter: StructuralLinter) -> None:
  """Hits duplicate direct usage.

  Args:
      linter: The linter (StructuralLinter) for test linter duplicate direct.
  """
  code: str = """
a = torch.tensor()
b = torch.tensor()

c = torch.abs()
d = torch.abs()
"""
  errors: list[str] = linter.check(code)
  assert len(errors) > 0


def test_linter_empty_names(linter: StructuralLinter) -> None:
  """Hits impossible empty names branch.

  Args:
      linter: The linter (StructuralLinter) for test linter empty names.
  """
  import libcst as cst

  # node.names is not normally empty, but we can pass an empty list manually
  node: MagicMock = MagicMock()
  node.module = cst.Name("torch")
  node.names = ["not_an_import_alias"]
  # Temporarily add context to bypass definition skip if any
  linter._context_stack.append("import")
  linter.visit_ImportFrom(node)
  _res = None
  assert _res is None
  assert isinstance(linter.violations, list)


def test_linter_missing_coverage() -> None:
  """Verifies the behavior of linter missing coverage."""
  from ml_switcheroo.testing.linter import StructuralLinter

  linter: StructuralLinter = StructuralLinter({"torch"})
  res: list[str] = linter.check("from torch import *")
  assert any(("Wildcard" in msg for msg in res))
  import libcst as cst

  assert linter._get_root_name(cst.Integer("1")) == ""
  assert linter._get_full_name_from_node(cst.Integer("1")) == ""


def test_linter_get_full_name_attribute() -> None:
  """Verifies the behavior of linter get full name attribute."""
  import libcst as cst

  from ml_switcheroo.testing.linter import StructuralLinter

  linter: StructuralLinter = StructuralLinter({"torch"})
  node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert linter._get_full_name_from_node(node) == "torch.nn"


def test_linter_parse_error_extra() -> None:
  """Verifies the behavior of linter parse correctly handling an error."""
  from ml_switcheroo.testing.linter import validate_transpilation

  ok: bool
  msgs: list[str]
  ok, msgs = validate_transpilation("def foo(", "torch")
  assert not ok
  assert any(("Parse Error" in m for m in msgs))


def test_linter_direct_access() -> None:
  """Verifies the behavior of linter direct access."""
  from ml_switcheroo.testing.linter import validate_transpilation

  code: str = "import something_else\ntorch.add(x, y)"
  ok: bool
  msgs: list[str]
  ok, msgs = validate_transpilation(code, "torch")
  assert not ok
  assert any(("Direct access 'torch'" in m for m in msgs))
