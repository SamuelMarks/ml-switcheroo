"""Test suite for rdna backend emitter."""

import typing

from ml_switcheroo.core.compiler.backends.rdna.emitter import RdnaEmitter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel, RdnaNode


def test_rdna_emitter() -> None:
  """Verify rdna emitter behavior."""
  emitter: RdnaEmitter = RdnaEmitter()
  nodes: typing.List[RdnaNode] = [RdnaLabel(name="test_label")]
  result: str = emitter.emit(nodes)
  assert result == "test_label:\n"


def test_rdna_emitter_validate_instruction() -> None:
  """Test validate_instruction method of RdnaEmitter."""
  from unittest.mock import patch

  emitter = RdnaEmitter()
  report = emitter.validate_instruction("v_fma_f32")
  assert report is not None

  with patch("ml_ecosystem_snapshots.grounding.hardware.validate_rdna_instruction", side_effect=RuntimeError("Failed")):
    assert emitter.validate_instruction("v_fma_f32") is None
