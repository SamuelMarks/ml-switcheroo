"""Tests for Plugin and DSL Passes."""

import libcst as cst

from ml_switcheroo.core.rewriter.passes.plugins.dsl import FrameworkMacroPass, ImportFixerPass, LayoutPermutationPass


def test_layout_permutation_pass_nchw_to_nhwc() -> None:
  """Verify `LayoutPermutationPass` handles layout permutation pass nchw to nhwc as expected."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NCHW", target_layout="NHWC")
  modified = module.visit(transformer)
  assert modified.code == "conv2d(x).permute(0, 2, 3, 1)"


def test_layout_permutation_pass_nhwc_to_nchw() -> None:
  """Verify `LayoutPermutationPass` handles layout permutation pass nhwc to nchw as expected."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NHWC", target_layout="NCHW")
  modified = module.visit(transformer)
  assert modified.code == "conv2d(x).permute(0, 3, 1, 2)"


def test_layout_permutation_pass_same() -> None:
  """Verify `LayoutPermutationPass` handles layout permutation pass same as expected."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="NHWC", target_layout="NHWC")
  modified = module.visit(transformer)
  assert modified.code == source


def test_layout_permutation_pass_unknown() -> None:
  """Verify `LayoutPermutationPass` handles layout permutation pass unknown as expected."""
  source = "conv2d(x)"
  module = cst.parse_module(source)
  transformer = LayoutPermutationPass(source_layout="UNKNOWN", target_layout="OTHER")
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_success() -> None:
  """Verify `FrameworkMacroPass` handles framework macro pass success as expected."""
  source = "silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == "tensor_x * sigmoid(tensor_x)"


def test_framework_macro_pass_unmatched() -> None:
  """Verify `FrameworkMacroPass` handles framework macro pass unmatched as expected."""
  source = "relu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_not_name() -> None:
  """Verify `FrameworkMacroPass` handles framework macro pass not name as expected."""
  source = "math.silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_wrong_args() -> None:
  """Verify `FrameworkMacroPass` handles framework macro pass wrong args as expected."""
  source = "silu(x, y)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x})"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_framework_macro_pass_malformed() -> None:
  """Verify `FrameworkMacroPass` handles framework macro pass malformed as expected."""
  source = "silu(tensor_x)"
  module = cst.parse_module(source)
  macros = {"silu": "{x} * sigmoid({x}   # syntax error"}
  transformer = FrameworkMacroPass(macros=macros)
  modified = module.visit(transformer)
  assert modified.code == source


def test_import_fixer_pass_missing_imports() -> None:
  """Verify `ImportFixerPass` handles import fixer pass missing imports as expected."""
  source = "def foo():\n    pass\nx = 1"
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os", "sys"])
  modified = module.visit(transformer)
  assert "import sys\nimport os\ndef foo():" in modified.code


def test_import_fixer_pass_existing_imports() -> None:
  """Verify `ImportFixerPass` handles import fixer pass existing imports as expected."""
  source = "import os\nimport math\nx = 1"
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os", "sys"])
  modified = module.visit(transformer)
  assert "import sys\nimport os\nimport math\nx = 1" in modified.code


def test_import_fixer_pass_docstring() -> None:
  """Verify `ImportFixerPass` handles import fixer pass docstring as expected."""
  source = '"""My Docstring"""\nx = 1'
  module = cst.parse_module(source)
  transformer = ImportFixerPass(required_imports=["os"])
  modified = module.visit(transformer)
  assert '"""My Docstring"""\nimport os\nx = 1' in modified.code
