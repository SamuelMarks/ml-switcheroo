"""Test hardware lifters."""

import unittest
from ml_switcheroo.core.compiler.frontends.sass.parser import SassParser
from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter


class TestHardwareLifters(unittest.TestCase):
  """Test hardware lifters class."""

  def test_sass_lifter_conv2d(self):
    """Test sass lifter conv2d."""
    sass_code = """
L_KY_conv:
  MOV R2, RZ;
L_KX_conv:
  FFMA R0, R5, R6, R0;
  ISETP.LT.AND P0, PT, R2, 3, PT;
  BRA L_KX_conv;
        """
    parser = SassParser(sass_code)
    ast_nodes = parser.parse()
    lifter = SassLifter()
    graph = lifter.lift(ast_nodes)

    self.assertIsNotNone(graph)
    kinds = [n.kind for n in graph.nodes]
    self.assertNotIn("Linear", kinds)

  def test_rdna_lifter_conv2d(self):
    """Test rdna lifter conv2d."""
    rdna_code = """
v_mov_b32 v0, 0
v_add_f32 v1, v2, v3
s_cbranch_vccnz L_KX_conv
        """
    parser = RdnaParser(rdna_code)
    ast_nodes = parser.parse()
    lifter = RdnaLifter()
    graph = lifter.lift(ast_nodes)

    self.assertIsNotNone(graph)
    kinds = [n.kind for n in graph.nodes]
    self.assertNotIn("Linear", kinds)
