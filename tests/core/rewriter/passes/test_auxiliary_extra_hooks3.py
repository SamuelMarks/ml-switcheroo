"""Test module."""

import libcst as cst
from libcst.codemod import CodemodContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer


def test_cst_to_string_none_base():
  """Test function."""
  transformer = AuxiliaryTransformer(CodemodContext())
  attr_node = cst.Attribute(value=cst.Pass(), attr=cst.Name("bar"))
  assert transformer._cst_to_string(attr_node) is None
