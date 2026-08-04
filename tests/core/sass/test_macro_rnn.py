"""Test suite for the RNN SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import expand_rnn, expand_lstm, expand_gru
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassComment


def test_sass_macro_rnn() -> None:
  """Verifies that expand_rnn generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "rnn1"
  metadata = {"seq_len": 20}

  nodes = expand_rnn(allocator, node_id, metadata)
  assert len(nodes) >= 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN RNN ({node_id})" in comments

  opcodes = [n.opcode for n in nodes if isinstance(n, SassInstruction)]
  assert "FFMA" in opcodes
  assert "MUFU" in opcodes


def test_sass_macro_lstm() -> None:
  """Verifies that expand_lstm generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "lstm1"
  metadata = {"seq_len": 20}

  nodes = expand_lstm(allocator, node_id, metadata)
  assert len(nodes) >= 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN LSTM ({node_id})" in comments


def test_sass_macro_gru() -> None:
  """Verifies that expand_gru generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "gru1"
  metadata = {"seq_len": 20}

  nodes = expand_gru(allocator, node_id, metadata)
  assert len(nodes) >= 10

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN GRU ({node_id})" in comments


def test_sass_analyzer_rnn():
  """Verifies analyzer handles rnn ops safely."""
  instructions = []
  assert len(SassAnalyzer.analyze_block("RNN", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("LSTM", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("GRU", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("LSTMCell", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("GRUCell", instructions)) == 0
