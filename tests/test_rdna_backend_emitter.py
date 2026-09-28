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


def test_rdna_emitter_validate_instruction() -> None:
  """Test validate_instruction method of RdnaEmitter."""
  from unittest.mock import patch

  emitter = RdnaEmitter()
  report = emitter.validate_instruction("v_fma_f32")
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.hardware.validate_rdna_instruction", side_effect=Exception("Failed")):
    assert emitter.validate_instruction("v_fma_f32") is None
