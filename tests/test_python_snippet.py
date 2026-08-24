"""Docstring."""

from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode
import libcst as cst


def test_emit_init():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10", "bias": "True"})
  stmt = emitter.emit_init(node)
  code = cst.Module([stmt]).code
  assert "self.layer1 = nn.Linear(10, bias = True)" in code


def test_emit_init_flax():
  """Docstring."""
  emitter = PythonSnippetEmitter("flax_nnx")
  node = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt = emitter.emit_init(node)
  code = cst.Module([stmt]).code
  assert "self.layer1 = nnx.Linear(10, rngs = rngs)" in code


def test_emit_init_flax_existing_rngs():
  """Docstring."""
  emitter = PythonSnippetEmitter("flax_nnx")
  node = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10", "rngs": "rngs"})
  stmt = emitter.emit_init(node)
  code = cst.Module([stmt]).code
  assert "self.layer1 = nnx.Linear(10, rngs = rngs)" in code


def test_emit_init_not_stateful():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="layer1", kind="func_add", metadata={"arg_1": "10"})
  stmt = emitter.emit_init(node)
  code = cst.Module([stmt]).code
  assert "pass" in code


def test_emit_call_stateful():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="layer1", kind="Linear", metadata={"arg_1": "10"})
  stmt = emitter.emit_call(node, input_vars=["x"], output_var="y")
  code = cst.Module([stmt]).code
  assert "y = self.layer1(x)" in code


def test_emit_call_stateless():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="op1", kind="func_add", metadata={"arg_1": "1"})
  stmt = emitter.emit_call(node, input_vars=["x"], output_var="y")
  code = cst.Module([stmt]).code
  assert "y = torch.add(x, 1)" in code


def test_emit_call_input():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  node = LogicalNode(id="in1", kind="Input")
  stmt = emitter.emit_call(node, input_vars=["x"], output_var="y")
  code = cst.Module([stmt]).code
  assert "y = x" in code

  stmt2 = emitter.emit_call(node, input_vars=["x"], output_var="x")
  code2 = cst.Module([stmt2]).code
  assert "pass" in code2


def test_emit_expression_syntax_error():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  # invalid var name to trigger parse error
  node = LogicalNode(id="op1", kind="func_add")
  expr = emitter.emit_expression(node, input_vars=["1invalid"])
  assert isinstance(expr, cst.Name)
  assert expr.value == "None"


def test_resolve_api_name():
  """Docstring."""
  e_torch = PythonSnippetEmitter("torch")
  assert e_torch._resolve_api_name("torch.add") == "torch.add"
  assert e_torch._resolve_api_name("Linear") == "nn.Linear"
  assert e_torch._resolve_api_name("add") == "torch.add"

  e_jax = PythonSnippetEmitter("jax")
  assert e_jax._resolve_api_name("Linear") == "nnx.Linear"
  assert e_jax._resolve_api_name("add") == "jnp.add"

  e_keras = PythonSnippetEmitter("keras")
  assert e_keras._resolve_api_name("Linear") == "keras.layers.Linear"
  assert e_keras._resolve_api_name("add") == "keras.ops.add"

  e_other = PythonSnippetEmitter("other")
  assert e_other._resolve_api_name("Linear") == "Linear"


def test_is_stateful_layer():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="Input")) is False
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="func_add")) is False
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="torch.nn.functional.relu")) is False
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="Linear")) is True
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="torch.nn.Linear")) is True
  assert emitter._is_stateful_layer(LogicalNode(id="n", kind="add")) is False


def test_build_args_from_metadata_empty():
  """Docstring."""
  emitter = PythonSnippetEmitter("torch")
  assert emitter._build_args_from_metadata({}) == []
