"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel, RdnaNode
from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from typing import List


def test_rdna_emitter() -> None:
  """Test element."""
  emitter: RdnaEmitter = RdnaEmitter()
  nodes: List[RdnaNode] = [RdnaLabel(name="test_label")]
  result: str = emitter.emit(nodes)
  assert result == "test_label:\n"
