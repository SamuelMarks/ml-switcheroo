"""Test suite for the Linter module."""

import pytest
from ml_switcheroo.testing.linter import StructuralLinter, validate_transpilation
from unittest.mock import patch, MagicMock


@pytest.fixture
def linter():
  """Provides a mock linter for testing."""
  return StructuralLinter(forbidden_roots={"torch", "flax"})


def test_linter_clean_code(linter):
  """Verifies the behavior of linter clean code."""
  code = "\nimport jax.numpy as jnp\ndef f(x):\n    return jnp.abs(x)\n"
  errors = linter.check(code)
  assert len(errors) == 0


def test_linter_detects_import(linter):
  """Verifies the behavior of linter detects import."""
  code = "\nimport torch\nx = torch.abs(y)\n"
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'torch'" in errors[0]


def test_linter_detects_from_import(linter):
  """Verifies the behavior of linter detects from import."""
  code = "from flax import linen as nn"
  errors = linter.check(code)
  assert len(errors) > 0
  assert "Forbidden Import: 'from flax ...'" in errors[0]


def test_linter_detects_aliased_usage(linter):
  """Verifies the behavior of linter detects aliased usage."""
  code = "\nimport torch as t\n# Usage of alias\ny = t.abs(x)\n"
  errors = linter.check(code)
  assert len(errors) >= 1
  usage_errors = [e for e in errors if "Forbidden Usage" in e]
  assert len(usage_errors) > 0
  assert "alias of torch" in usage_errors[0]


def test_linter_parse_error(linter):
  """Hits lines 53-54 where cst.parse_module fails."""
  errors = linter.check("def f(): this is invalid python !")
  assert len(errors) == 1
  assert "Linter Parse Error" in errors[0]


def test_linter_detects_wildcard_import(linter):
  """Hits line 114 for wildcard imports."""
  code = "from torch import *"
  errors = linter.check(code)
  assert any("Forbidden Wildcard Import" in e for e in errors)


def test_linter_detects_direct_access(linter):
  """Hits lines 158-161 where forbidden root is accessed directly without explicit alias import tracking."""
  # torch is forbidden, we just use it directly
  code = "x = torch.Tensor()"
  errors = linter.check(code)
  # It should report forbidden import AND forbidden usage
  usage_errs = [e for e in errors if "Forbidden Usage: Direct access 'torch'" in e]
  assert len(usage_errs) > 0


def test_linter_complex_attribute_name(linter):
  """Hits lines 192 and 198-200 for recursive attribute resolution in linter."""
  # We test _get_full_name_from_node directly or via checking attribute access
  import libcst as cst

  code = "import torch\nx = torch.nn.functional.relu(y)"
  _errors = linter.check(code)

  # Triggering _get_full_name_from_node manually to guarantee line coverage
  tree = cst.parse_module("torch.nn.functional")
  attr_node = tree.body[0].body[0].value
  assert linter._get_full_name_from_node(attr_node) == "torch.nn.functional"

  # Triggering _get_root_name manually
  assert linter._get_root_name(attr_node) == "torch"


def test_linter_get_root_name_fallback(linter):
  """Hits line 192 in _get_root_name fallback."""
  import libcst as cst

  assert linter._get_root_name(cst.Integer("1")) == ""
  assert linter._get_full_name_from_node(cst.Integer("1")) == ""
  """Verifies the behavior of facade Flax inheritance."""
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("flax.nnx", "nnx")
  mock_adapter.inherits_from = "jax"
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = "import jax.numpy as jnp"
    (is_valid, errors) = validate_transpilation(code, source_fw="flax_nnx")
    assert not is_valid
    assert "Forbidden Import: 'jax'" in errors[0]


def test_facade_mlx_detection(tmp_path):
  """Verifies the behavior of facade MLX detection."""
  mock_adapter = MagicMock()
  mock_adapter.import_alias = ("mlx.core", "mx")
  mock_adapter.search_modules = ["mlx"]
  mock_adapter.inherits_from = None
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    code = "\nimport mlx.core as mx\ndef f(x):\n    return mx.abs(x)\n"
    (is_valid, errors) = validate_transpilation(code, source_fw="mlx")
    assert not is_valid
    assert any(("mlx" in e for e in errors))


def test_linter_branches(linter):
  """Hits missing branches in linter.py."""
  code = """
import numpy as np
from . import relative
from os import path
from torch import abs, add
import torch.nn as nn

x = nn.Linear()
y = torch.tensor()
"""
  errors = linter.check(code)
  assert len(errors) > 0


def test_linter_duplicate_violations(linter):
  """Hits duplicate violation branches."""
  code = """
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
  errors = linter.check(code)
  assert len(errors) > 0


def test_validate_transpilation_missing_adapter():
  """Hits missing adapter branch in validate_transpilation."""
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=None):
    is_valid, errors = validate_transpilation("import torch", "unknown")
    assert is_valid


def test_validate_transpilation_empty_adapter():
  """Hits empty adapter branch in validate_transpilation."""
  mock_adapter = MagicMock(spec=[])
  with patch("ml_switcheroo.testing.linter.get_adapter", return_value=mock_adapter):
    is_valid, errors = validate_transpilation("import torch", "torch")
    assert not is_valid


def test_linter_duplicate_direct(linter):
  """Hits duplicate direct usage."""
  code = """
a = torch.tensor()
b = torch.tensor()

c = torch.abs()
d = torch.abs()
"""
  errors = linter.check(code)
  assert len(errors) > 0


def test_linter_empty_names(linter):
  """Hits impossible empty names branch."""
  import libcst as cst

  # node.names is not normally empty, but we can pass an empty list manually
  node = MagicMock()
  node.module = cst.Name("torch")
  node.names = ["not_an_import_alias"]
  # Temporarily add context to bypass definition skip if any
  linter._context_stack.append("import")
  linter.visit_ImportFrom(node)
