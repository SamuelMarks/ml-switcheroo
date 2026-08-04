"""Test suite for the Attention SASS Macros."""

from ml_switcheroo.core.compiler.backends.sass.macros import (
  expand_multiheadattention,
  expand_transformer,
  expand_transformerencoder,
  expand_transformerdecoder,
)
from ml_switcheroo.core.compiler.frontends.sass.analysis import SassAnalyzer
from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment


def test_sass_macro_multiheadattention() -> None:
  """Verifies that expand_multiheadattention generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "mha1"
  metadata = {}

  nodes = expand_multiheadattention(allocator, node_id, metadata)
  assert len(nodes) > 5

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN MultiheadAttention ({node_id})" in comments


def test_sass_macro_transformer() -> None:
  """Verifies that expand_transformer generates correct SASS instructions."""
  allocator = RegisterAllocator()
  node_id = "tf1"
  metadata = {}

  nodes = expand_transformer(allocator, node_id, metadata)
  assert len(nodes) > 3

  comments = [n.text for n in nodes if isinstance(n, SassComment)]
  assert f"BEGIN Transformer ({node_id})" in comments


def test_sass_macro_transformer_enc_dec() -> None:
  """Verifies enc/dec macros."""
  allocator = RegisterAllocator()
  node_id = "enc1"
  metadata = {}

  nodes = expand_transformerencoder(allocator, node_id, metadata)
  assert len(nodes) >= 2

  nodes2 = expand_transformerdecoder(allocator, node_id, metadata)
  assert len(nodes2) >= 2


def test_sass_analyzer_attention():
  """Verifies analyzer handles attention ops safely."""
  instructions = []
  assert len(SassAnalyzer.analyze_block("MultiheadAttention", instructions)) == 0
  assert len(SassAnalyzer.analyze_block("Transformer", instructions)) == 0
