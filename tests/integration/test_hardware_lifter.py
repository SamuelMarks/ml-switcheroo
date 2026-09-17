"""Test suite for the Hardware Lifter module."""

import typing
import unittest

from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.ir import LogicalGraph


class TestHardwareLifters(unittest.TestCase):
  """Docstring."""

  def test_nvidia_sass_lifter_conv2d(self) -> None:
    """Verifies the behavior of NVIDIA_SASS lifter conv2d."""
    sass_code: str = """
L_KY_conv:
  MOV R2, RZ;
L_KX_conv:
  FFMA R0, R5, R6, R0;
  ISETP.LT.AND P0, PT, R2, 3, PT;
  BRA L_KX_conv;
"""
    parser = NvidiaSassParser(sass_code)
    ast_nodes: list[typing.Any] = parser.parse().statements
    lifter = NvidiaSassLifter()
    graph: LogicalGraph = lifter.lift(ast_nodes)
    self.assertIsNotNone(graph)
    kinds: list[str] = [n.op_type for n in graph.nodes.values()]
    self.assertNotIn("Linear", kinds)

  def test_rdna_lifter_conv2d(self) -> None:
    """Verifies the behavior of RDNA lifter conv2d."""
    rdna_code: str = """
v_mov_b32 v0, 0
v_add_f32 v1, v2, v3
s_cbranch_vccnz L_KX_conv
"""
    parser = RdnaParser(rdna_code)
    ast_nodes: list[typing.Any] = parser.parse().statements
    lifter = RdnaLifter()
    graph: LogicalGraph = lifter.lift(ast_nodes)
    self.assertIsNotNone(graph)
    kinds: list[str] = [n.op_type for n in graph.nodes.values()]
    self.assertNotIn("Linear", kinds)
