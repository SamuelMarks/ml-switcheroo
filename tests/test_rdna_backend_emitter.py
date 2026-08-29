"""Test module."""

from typing import List

from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel, RdnaNode


def test_rdna_emitter() -> None:
  """Docstring."""
  emitter: RdnaEmitter = RdnaEmitter()
  nodes: List[RdnaNode] = [RdnaLabel(name="test_label")]
  result: str = emitter.emit(nodes)
  assert result == "test_label:\n"
