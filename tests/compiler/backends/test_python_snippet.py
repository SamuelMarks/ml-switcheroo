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

  src = capture_node_source(stmt).replace(" ", "")
  assert "self.conv1=nn.Conv2d" in src
  assert "in_channels=3" in src
  assert "out_channels=64" in src


def test_emit_init_stateful_flax_rng(emitter_flax):
  """Emits initialization stateful Flax rng."""
  node = LogicalNode(id="fc1", kind="Linear", metadata={"features": 10})
  stmt = emitter_flax.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt).replace(" ", "")
  assert "self.fc1=nnx.Linear" in src
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


def test_emit_call_input_vars(emitter_torch):
  """Emits call input vars."""
  node = LogicalNode(id="x", kind="Input")
  stmt1 = emitter_torch.emit_call(node, input_vars=["x"], output_var="x")
  from ml_switcheroo.utils.node_diff import capture_node_source

  assert capture_node_source(stmt1).strip() == "pass"

  stmt2 = emitter_torch.emit_call(node, input_vars=["x_in"], output_var="x")
  assert capture_node_source(stmt2).strip() == "x = x_in"


def test_emit_expression_syntax_error(emitter_torch):
  """Emits expression syntax error fallback."""
  node = LogicalNode(id="bad", kind="func_bad")
  expr = emitter_torch.emit_expression(node, input_vars=["*invalid syntax*"])
  from ml_switcheroo.utils.node_diff import capture_node_source

  assert capture_node_source(expr).strip() == "None"


def test_is_stateful_layer_checks(emitter_torch):
  """Checks stateful layer logic."""
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="o", kind="Output"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="i", kind="functional_add"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="o", kind="ops_something"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="l", kind="lower"))


def test_resolve_api_keras():
  """Resolves api keras."""
  emitter_keras = PythonSnippetEmitter("keras")
  assert emitter_keras._resolve_api_name("Dense") == "keras.layers.Dense"
  assert emitter_keras._resolve_api_name("add") == "keras.ops.add"

  emitter_other = PythonSnippetEmitter("numpy")
  assert emitter_other._resolve_api_name("add") == "add"


def test_resolve_api_dotted_and_jax(emitter_flax):
  """Resolves dotted api and jax api."""
  assert emitter_flax._resolve_api_name("jax.numpy.add") == "jax.numpy.add"
  assert emitter_flax._resolve_api_name("add") == "jnp.add"


def test_format_args_from_metadata_empty(emitter_torch):
  """Tests format args from metadata empty."""
  assert emitter_torch._build_args_from_metadata({}) == []


def test_python_snippet_emit_init_flax_with_rngs():
  """Test function."""
  from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
  from ml_switcheroo.core.graph import LogicalNode

  # To cover the 'any(...)' branch evaluating to True, we provide 'rngs' in metadata
  # Note: `_build_args_from_metadata` processes metadata
  node = LogicalNode("l1", "nn.Linear", metadata={"in_features": 10, "rngs": "rngs"})
  backend = PythonSnippetEmitter(framework="flax")
  stmt = backend.emit_init(node)
  # ensure it generated correctly without crashing
  code = __import__("libcst").Module(body=[stmt]).code
  assert "rngs" in code
