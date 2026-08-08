"""Tests for extra features in the RDNA adapter."""

from ml_switcheroo.frameworks.rdna import RdnaAdapter


def test_rdna_label_pass():
  """Test that RDNA label parsing passes successfully."""
  adapter = RdnaAdapter()
  code = """
L_123:
BB0_1:
v_mac_f32 v0, v1, v2
"""
  res = adapter.parse_rdna_to_graph(code)
  assert res is not None
