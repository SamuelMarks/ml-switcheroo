"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaModule, RdnaComment
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_lifter_end_mismatch() -> None:
  """Test element."""
  mod: RdnaModule = RdnaModule(
    statements=[
      RdnaComment(text="; BEGIN: Linear(some_id)"),
      RdnaComment(text="; END: some_other_id"),  # mismatch
    ]
  )
  lifter: RdnaLifter = RdnaLifter()
  graph: LogicalGraph = lifter.lift(mod.statements)
  assert graph is not None


def test_lifter_return_seen() -> None:
  """Test element."""
  mod: RdnaModule = RdnaModule(
    statements=[
      RdnaComment(text="; RETURN"),
      RdnaComment(text="; RETURN"),
    ]
  )
  lifter: RdnaLifter = RdnaLifter()
  lifter.lift(mod.statements)
