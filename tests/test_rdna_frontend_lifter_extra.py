"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaModule, RdnaComment


def test_lifter_end_mismatch():
  """Test element."""
  mod = RdnaModule(
    statements=[
      RdnaComment(text="; BEGIN: Linear(some_id)"),
      RdnaComment(text="; END: some_other_id"),  # mismatch
    ]
  )
  lifter = RdnaLifter()
  graph = lifter.lift(mod.statements)
  assert graph is not None


def test_lifter_return_seen():
  """Test element."""
  mod = RdnaModule(
    statements=[
      RdnaComment(text="; RETURN"),
      RdnaComment(text="; RETURN"),
    ]
  )
  lifter = RdnaLifter()
  lifter.lift(mod.statements)
