"""Test suite for macro rnn."""

from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_gru, expand_lstm, expand_rnn
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment, NvidiaSassInstruction
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator
from ml_switcheroo.core.compiler.frontends.nvidia_sass.analysis import NvidiaSassAnalyzer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode


def test_nvidia_sass_macro_rnn() -> None:
  """Verifies that expand_rnn generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: None)
  node_id = "rnn1"
  metadata = {"seq_len": 20}

  nodes: list[NvidiaSassNode] = expand_rnn(allocator, node_id, metadata)  # type: ignore[arg-type]
  assert len(nodes) >= 10

  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN RNN ({node_id})" in comments

  opcodes: list[str] = [n.opcode for n in nodes if isinstance(n, NvidiaSassInstruction)]
  assert "FFMA" in opcodes
  assert "MUFU" in opcodes


def test_nvidia_sass_macro_lstm() -> None:
  """Verifies that expand_lstm generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: None)
  node_id = "lstm1"
  metadata = {"seq_len": 20}

  nodes: list[NvidiaSassNode] = expand_lstm(allocator, node_id, metadata)  # type: ignore[arg-type]
  assert len(nodes) >= 10

  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN LSTM ({node_id})" in comments


def test_nvidia_sass_macro_gru() -> None:
  """Verifies that expand_gru generates correct NVIDIA_SASS instructions."""
  allocator = RegisterAllocator(256, lambda x: None)
  node_id = "gru1"
  metadata = {"seq_len": 20}

  nodes: list[NvidiaSassNode] = expand_gru(allocator, node_id, metadata)  # type: ignore[arg-type]
  assert len(nodes) >= 10

  comments: list[str] = [n.text for n in nodes if isinstance(n, NvidiaSassComment)]
  assert f"BEGIN GRU ({node_id})" in comments


def test_nvidia_sass_analyzer_rnn() -> None:
  """Verifies analyzer handles rnn ops safely."""
  instructions: list[NvidiaSassInstruction] = []
  assert len(NvidiaSassAnalyzer.analyze_block("RNN", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("LSTM", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("GRU", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("LSTMCell", instructions)) == 0
  assert len(NvidiaSassAnalyzer.analyze_block("GRUCell", instructions)) == 0
