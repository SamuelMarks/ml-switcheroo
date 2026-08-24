"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel
from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter


def test_rdna_emitter():
  """Test element."""
  emitter = RdnaEmitter()
  nodes = [RdnaLabel(name="test_label")]
  result = emitter.emit(nodes)
  assert result == "test_label:\n"
