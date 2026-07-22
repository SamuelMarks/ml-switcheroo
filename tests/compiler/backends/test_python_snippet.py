"""Test suite for the Python Snippet module."""

import pytest
from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode


@pytest.fixture
def emitter_torch():
  """Provides a mock emitter PyTorch for testing."""
  return PythonSnippetEmitter("torch")


@pytest.fixture
def emitter_flax():
  """Provides a mock emitter Flax for testing."""
  return PythonSnippetEmitter("flax_nnx")


def test_emit_init_stateful_torch(emitter_torch):
  """Emits initialization stateful PyTorch."""
  node = LogicalNode(id="conv1", kind="Conv2d", metadata={"in_channels": 3, "out_channels": 64})
  stmt = emitter_torch.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert "self.conv1 = nn.Conv2d" in src
  assert "in_channels=3" in src
  assert "out_channels=64" in src


def test_emit_init_stateful_flax_rng(emitter_flax):
  """Emits initialization stateful Flax rng."""
  node = LogicalNode(id="fc1", kind="Linear", metadata={"features": 10})
  stmt = emitter_flax.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert "self.fc1 = nnx.Linear" in src
  assert "features=10" in src
  assert "rngs=rngs" in src


def test_emit_init_functional_noop(emitter_torch):
  """Emits initialization functional noop."""
  node = LogicalNode(id="r1", kind="func_relu")
  stmt = emitter_torch.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert src.strip() == "pass"


def test_emit_call_stateful(emitter_torch):
  """Emits call stateful."""
  node = LogicalNode(id="conv1", kind="Conv2d")
  stmt = emitter_torch.emit_call(node, input_vars=["x"], output_var="y")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert "y = self.conv1(x)" in src


def test_emit_call_functional_params(emitter_torch):
  """Emits call functional parameters."""
  node = LogicalNode(id="flat", kind="func_flatten", metadata={"arg_0": "1"})
  stmt = emitter_torch.emit_call(node, input_vars=["x"], output_var="z")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert "z = torch.flatten(x, 1)" in src


def test_emit_call_multi_input(emitter_torch):
  """Emits call multi input."""
  node = LogicalNode(id="add", kind="func_add")
  stmt = emitter_torch.emit_call(node, input_vars=["a", "b"], output_var="c")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert "c = torch.add(a, b)" in src


def test_resolve_api_heuristics(emitter_torch, emitter_flax):
  """Resolves API heuristics."""
  assert emitter_torch._resolve_api_name("Conv2d") == "nn.Conv2d"
  assert emitter_torch._resolve_api_name("abs") == "torch.abs"
  assert emitter_torch._resolve_api_name("custom.pkg.Layer") == "custom.pkg.Layer"
  assert emitter_flax._resolve_api_name("Conv") == "nnx.Conv"
  assert emitter_flax._resolve_api_name("abs") == "jnp.abs"
