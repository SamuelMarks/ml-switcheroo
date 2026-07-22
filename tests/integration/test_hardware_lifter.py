"""Test suite for the Hardware Lifter module."""

import unittest
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter


class TestHardwareLifters(unittest.TestCase):
  """Test suite for the Hardware Lifters component."""

  def test_sass_lifter_conv2d(self):
    """Verifies the behavior of SASS lifter conv2d."""
    sass_code = "\nL_KY_conv:\n  MOV R2, RZ;\nL_KX_conv:\n  FFMA R0, R5, R6, R0;\n  ISETP.LT.AND P0, PT, R2, 3, PT;\n  BRA L_KX_conv;\n        "
    parser = SassParser(sass_code)
    ast_nodes = parser.parse()
    lifter = SassLifter()
    graph = lifter.lift(ast_nodes)
    self.assertIsNotNone(graph)
    kinds = [n.kind for n in graph.nodes]
    self.assertNotIn("Linear", kinds)

  def test_rdna_lifter_conv2d(self):
    """Verifies the behavior of RDNA lifter conv2d."""
    rdna_code = "\nv_mov_b32 v0, 0\nv_add_f32 v1, v2, v3\ns_cbranch_vccnz L_KX_conv\n        "
    parser = RdnaParser(rdna_code)
    ast_nodes = parser.parse()
    lifter = RdnaLifter()
    graph = lifter.lift(ast_nodes)
    self.assertIsNotNone(graph)
    kinds = [n.kind for n in graph.nodes]
    self.assertNotIn("Linear", kinds)
