"""Unit tests for `PythonSnippetEmitter` (`ml_switcheroo.core.compiler.backends.python_snippet`)."""

import libcst as cst

from ml_switcheroo.core.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.core.compiler.ir import LogicalNode


def test_python_snippet_emitter_init_call_expression_and_api_resolution() -> None:
  """Verify `emit_init`, `emit_call`, `emit_expression`, `_is_stateful_layer`, and `_resolve_api_name`."""
  e_torch = PythonSnippetEmitter("torch")
  e_flax = PythonSnippetEmitter("flax_nnx")
  e_keras = PythonSnippetEmitter("keras")
  e_other = PythonSnippetEmitter("numpy")

  # `emit_init`: Torch stateful, Flax stateful (auto-injected and pre-existing `rngs`), and stateless noop
  conv_init = cst.Module(
    [e_torch.emit_init(LogicalNode("conv1", "Conv2d", attributes={"in_channels": 3, "bias": "True"}))]
  ).code
  assert "self.conv1 = nn.Conv2d" in conv_init
  assert "in_channels = 3" in conv_init

  flax_init = cst.Module([e_flax.emit_init(LogicalNode("fc1", "Linear", attributes={"arg_1": "10"}))]).code
  assert "self.fc1 = nnx.Linear(10, rngs = rngs)" in flax_init

  flax_existing = cst.Module(
    [e_flax.emit_init(LogicalNode("fc1", "Linear", attributes={"arg_1": "10", "rngs": "rngs"}))]
  ).code
  assert flax_existing.count("rngs") == 2

  assert cst.Module([e_torch.emit_init(LogicalNode("r1", "func_relu"))]).code.strip() == "pass"

  # `emit_call`: stateful, stateless with positional metadata, multi-input, and `Input` identity/rename
  assert "y = self.conv1(x)" in cst.Module([e_torch.emit_call(LogicalNode("conv1", "Conv2d"), ["x"], "y")]).code
  assert (
    "z = torch.flatten(x, 1)"
    in cst.Module([e_torch.emit_call(LogicalNode("f", "func_flatten", attributes={"arg_0": "1"}), ["x"], "z")]).code
  )
  assert "c = torch.add(a, b)" in cst.Module([e_torch.emit_call(LogicalNode("add", "func_add"), ["a", "b"], "c")]).code
  assert cst.Module([e_torch.emit_call(LogicalNode("in", "Input"), ["x"], "x")]).code.strip() == "pass"
  assert cst.Module([e_torch.emit_call(LogicalNode("in", "Input"), ["x_in"], "x")]).code.strip() == "x = x_in"

  # `emit_expression` invalid identifier fallback -> `cst.Name("None")`
  bad_expr = e_torch.emit_expression(LogicalNode("bad", "func_add"), input_vars=["*invalid*"])
  assert isinstance(bad_expr, cst.Name)
  assert bad_expr.value == "None"

  # `_is_stateful_layer`, `_resolve_api_name`, and `_build_args_from_metadata({})`
  for non_stateful in ("Input", "Output", "func_add", "functional_add", "ops_add", "torch.nn.functional.relu", "add"):
    assert e_torch._is_stateful_layer(LogicalNode("n", non_stateful)) is False
  assert e_torch._is_stateful_layer(LogicalNode("n", "Linear")) is True
  assert e_torch._is_stateful_layer(LogicalNode("n", "torch.nn.Linear")) is True

  assert e_torch._resolve_api_name("torch.add") == "torch.add"
  assert e_torch._resolve_api_name("Linear") == "nn.Linear"
  assert e_torch._resolve_api_name("add") == "torch.add"
  assert e_flax._resolve_api_name("Linear") == "nnx.Linear"
  assert e_flax._resolve_api_name("add") == "jnp.add"
  assert e_keras._resolve_api_name("Dense") == "keras.layers.Dense"
  assert e_keras._resolve_api_name("add") == "keras.ops.add"
  assert e_other._resolve_api_name("Linear") == "Linear"
  assert e_torch._build_args_from_metadata({}) == []
