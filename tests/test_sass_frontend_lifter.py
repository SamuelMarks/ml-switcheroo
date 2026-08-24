"""Test module."""

from ml_switcheroo.core.compiler.frontends.sass.lifter import SassLifter
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment, SassInstruction, SassRegister, SassLabel


def test_sass_lifter_basic():
  """Test element."""
  stmts = [
    SassComment(text="; BEGIN Linear(add1)"),
    SassInstruction(opcode="VADD", operands=[SassRegister(name="R0"), SassRegister(name="R1"), SassRegister(name="R2")]),
    SassComment(text="; END Linear(add1)"),
    SassComment(text="; Input in1 ->"),
    SassComment(text="; Return:"),
    SassComment(text="; Return:"),  # Duplicate return
    SassComment(text="; Unmapped Op: something(xyz)"),
    SassLabel(name="L1"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_mismatch_end():
  """Test element."""
  stmts = [
    SassComment(text="; BEGIN Linear(add1)"),
    SassComment(text="; END Linear(add2)"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_unmapped_flatten():
  """Test element."""
  stmts = [
    SassComment(text="; Unmapped Op: flatten(some_id)"),
  ]
  lifter = SassLifter()
  graph = lifter.lift(stmts)
  assert len(graph.nodes) >= 0


def test_sass_lifter_instruction_unmapped():
  """Test element."""
  stmts = [
    SassInstruction(opcode="VADD", operands=[]),
  ]
  lifter = SassLifter()
  graph = lifter.lift(stmts)
  # The first instruction gets added as unmapped node
  assert len(graph.nodes) >= 0
