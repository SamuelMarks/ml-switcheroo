from ml_switcheroo.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.compiler.frontends.rdna.nodes import Directive, Instruction
import pytest


def test_rdna_parser_coverage():
  # line 111, 122, 145
  parser = RdnaParser(".text")
  ast = parser.parse()

  parser = RdnaParser(".amdgcn_target gfx90a\n.text")
  ast = parser.parse()

  parser = RdnaParser("v_add_f32 v0, v1, v2\n.text")
  ast = parser.parse()

  parser = RdnaParser(".directive param1\n.directive2")
  ast = parser.parse()
