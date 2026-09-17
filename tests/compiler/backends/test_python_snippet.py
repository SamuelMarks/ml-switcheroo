"""Test suite for the Python Snippet module."""

import typing

import pytest

from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode


@pytest.fixture
def emitter_torch() -> PythonSnippetEmitter:
  """Docstring."""
  return PythonSnippetEmitter("torch")


@pytest.fixture
def emitter_flax() -> PythonSnippetEmitter:
  """Docstring."""
  return PythonSnippetEmitter("flax_nnx")


def test_emit_init_stateful_torch(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits initialization stateful PyTorch."""
  node = LogicalNode(id="conv1", op_type="Conv2d", attributes={"in_channels": 3, "out_channels": 64})
  stmt: typing.Any = emitter_torch.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt).replace(" ", "")
  assert "self.conv1=nn.Conv2d" in src
  assert "in_channels=3" in src
  assert "out_channels=64" in src


def test_emit_init_stateful_flax_rng(emitter_flax: PythonSnippetEmitter) -> None:
  """Emits initialization stateful Flax rng."""
  node = LogicalNode(id="fc1", op_type="Linear", attributes={"features": 10})
  stmt: typing.Any = emitter_flax.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt).replace(" ", "")
  assert "self.fc1=nnx.Linear" in src
  assert "features=10" in src
  assert "rngs=rngs" in src


def test_emit_init_functional_noop(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits initialization functional noop."""
  node = LogicalNode(id="r1", op_type="func_relu")
  stmt: typing.Any = emitter_torch.emit_init(node)
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt)
  assert src.strip() == "pass"


def test_emit_call_stateful(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits call stateful."""
  node = LogicalNode(id="conv1", op_type="Conv2d")
  stmt: typing.Any = emitter_torch.emit_call(node, input_vars=["x"], output_var="y")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt)
  assert "y = self.conv1(x)" in src


def test_emit_call_functional_params(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits call functional parameters."""
  node = LogicalNode(id="flat", op_type="func_flatten", attributes={"arg_0": "1"})
  stmt: typing.Any = emitter_torch.emit_call(node, input_vars=["x"], output_var="z")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt)
  assert "z = torch.flatten(x, 1)" in src


def test_emit_call_multi_input(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits call multi input."""
  node = LogicalNode(id="add", op_type="func_add")
  stmt: typing.Any = emitter_torch.emit_call(node, input_vars=["a", "b"], output_var="c")
  from ml_switcheroo.utils.node_diff import capture_node_source

  src: str = capture_node_source(stmt)
  assert "c = torch.add(a, b)" in src


def test_emit_call_input_vars(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits call input vars."""
  node = LogicalNode(id="x", op_type="Input")
  stmt1: typing.Any = emitter_torch.emit_call(node, input_vars=["x"], output_var="x")
  from ml_switcheroo.utils.node_diff import capture_node_source

  assert capture_node_source(stmt1).strip() == "pass"

  stmt2: typing.Any = emitter_torch.emit_call(node, input_vars=["x_in"], output_var="x")
  assert capture_node_source(stmt2).strip() == "x = x_in"


def test_emit_expression_syntax_error(emitter_torch: PythonSnippetEmitter) -> None:
  """Emits expression syntax error fallback."""
  node = LogicalNode(id="bad", op_type="func_bad")
  expr: typing.Any = emitter_torch.emit_expression(node, input_vars=["*invalid syntax*"])
  from ml_switcheroo.utils.node_diff import capture_node_source

  assert capture_node_source(expr).strip() == "None"


def test_is_stateful_layer_checks(emitter_torch: PythonSnippetEmitter) -> None:
  """Checks stateful layer logic."""
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="o", op_type="Output"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="i", op_type="functional_add"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="o", op_type="ops_something"))
  assert not emitter_torch._is_stateful_layer(LogicalNode(id="l", op_type="lower"))


def test_resolve_api_keras() -> None:
  """Resolves api keras."""
  emitter_keras = PythonSnippetEmitter("keras")
  assert emitter_keras._resolve_api_name("Dense") == "keras.layers.Dense"
  assert emitter_keras._resolve_api_name("add") == "keras.ops.add"

  emitter_other = PythonSnippetEmitter("numpy")
  assert emitter_other._resolve_api_name("add") == "add"


def test_resolve_api_dotted_and_jax(emitter_flax: PythonSnippetEmitter) -> None:
  """Resolves dotted api and jax api."""
  assert emitter_flax._resolve_api_name("jax.numpy.add") == "jax.numpy.add"
  assert emitter_flax._resolve_api_name("add") == "jnp.add"


def test_format_args_from_metadata_empty(emitter_torch: PythonSnippetEmitter) -> None:
  """Docstring."""
  assert emitter_torch._build_args_from_metadata({}) == []


def test_python_snippet_emit_init_flax_with_rngs() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
  from ml_switcheroo.core.graph import LogicalNode

  # To cover the 'any(...)' branch evaluating to True, we provide 'rngs' in metadata
  # Note: `_build_args_from_metadata` processes metadata
  node = LogicalNode("l1", "nn.Linear", attributes={"in_features": 10, "rngs": "rngs"})
  backend = PythonSnippetEmitter(framework="flax")
  stmt: typing.Any = backend.emit_init(node)
  # ensure it generated correctly without crashing
  code: str = __import__("libcst").Module(body=[stmt]).code
  assert "rngs" in code
