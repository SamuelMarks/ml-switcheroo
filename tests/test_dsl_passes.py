"""Tests for Plugin and DSL Passes."""

import libcst as cst

from ml_switcheroo.core.rewriter.passes.plugins.dsl import FrameworkMacroPass, ImportFixerPass, LayoutPermutationPass


def test_layout_permutation_pass_nchw_to_nhwc() -> None:
  """Test docstring."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NCHW", target_layout="NHWC")
  modified = module.visit(transformer)
  assert modified.code == "conv2d(x).permute(0, 2, 3, 1)"


def test_layout_permutation_pass_nhwc_to_nchw() -> None:
  """Test docstring."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NHWC", target_layout="NCHW")
  modified = module.visit(transformer)
  assert modified.code == "conv2d(x).permute(0, 3, 1, 2)"


def test_layout_permutation_pass_same() -> None:
  """Test docstring."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NHWC", target_layout="NHWC")
  modified = module.visit(transformer)
  assert modified.code == source


def test_layout_permutation_pass_unknown() -> None:
  """Test docstring."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="UNKNOWN", target_layout="OTHER")
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_success() -> None:
  """Test docstring."""
  source = "silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == "tensor_x * sigmoid(tensor_x)"


def test_framework_macro_pass_unmatched() -> None:
  """Test docstring."""
  source = "relu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_not_name() -> None:
  """Test docstring."""
  source = "math.silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_wrong_args() -> None:
  """Test docstring."""
  source = "silu(x, y)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_malformed() -> None:
  """Test docstring."""
  source = "silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x}   # syntax error"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_import_fixer_pass_missing_imports() -> None:
  """Test docstring."""
  source = "def foo():\n    pass\nx = 1"
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os", "sys"])
  modified = module.visit(transformer)
  assert "import sys\nimport os\ndef foo():" in modified.code


def test_import_fixer_pass_existing_imports() -> None:
  """Test docstring."""
  source = "import os\nimport math\nx = 1"
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os", "sys"])
  modified = module.visit(transformer)
  assert "import sys\nimport os\nimport math\nx = 1" in modified.code


def test_import_fixer_pass_docstring() -> None:
  """Test docstring."""
  source = '"""My Docstring"""\nx = 1'
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os"])
  modified = module.visit(transformer)
  assert '"""My Docstring"""\nimport os\nx = 1' in modified.code
