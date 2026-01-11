"""
Tests for Python Snippet Emitter.

Verifies:
1.  **Initialization Generation**: Correct syntax for `self.layer = ...`.
2.  **Call Generation**: Correct syntax for `y = self.layer(x)`.
3.  **Framework idiomatic prefixes**: `nn.` (Torch) vs `nnx.` (Flax).
4.  **Metadata rendering**: Correct argument formatting.
5.  **Functional vs Stateful**: Correct handling of `func_` prefix logic.
"""

import pytest
from ml_switcheroo.compiler.backends.python_snippet import PythonSnippetEmitter
from ml_switcheroo.compiler.ir import LogicalNode


@pytest.fixture
def emitter_torch():
  return PythonSnippetEmitter("torch")


@pytest.fixture
def emitter_flax():
  return PythonSnippetEmitter("flax_nnx")


def test_emit_init_stateful_torch(emitter_torch):
  """
  Scenario: Torch Conv2d initialization.
  Expect: self.conv1 = nn.Conv2d(in_channels=3, out_channels=64)
  """
  node = LogicalNode(id="conv1", kind="Conv2d", metadata={"in_channels": 3, "out_channels": 64})
  stmt = emitter_torch.emit_init(node)

  # Verify code generation as string (simplest robustness check against AST structure)

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)

  assert "self.conv1 = nn.Conv2d" in src
  assert "in_channels=3" in src
  assert "out_channels=64" in src


def test_emit_init_stateful_flax_rng(emitter_flax):
  """
  Scenario: Flax Linear initialization.
  Expect: Injection of `rngs=rngs`.
  """
  node = LogicalNode(id="fc1", kind="Linear", metadata={"features": 10})
  stmt = emitter_flax.emit_init(node)

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)

  # Should resolve to nnx prefix
  assert "self.fc1 = nnx.Linear" in src
  assert "features=10" in src
  # Must inject rngs
  assert "rngs=rngs" in src


def test_emit_init_functional_noop(emitter_torch):
  """
  Scenario: Functional node (relu).
  Expect: pass (No init).
  """
  node = LogicalNode(id="r1", kind="func_relu")
  stmt = emitter_torch.emit_init(node)

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)
  assert src.strip() == "pass"


def test_emit_call_stateful(emitter_torch):
  """
  Scenario: Calling the previously initialized layer.
  """
  node = LogicalNode(id="conv1", kind="Conv2d")
  stmt = emitter_torch.emit_call(node, input_vars=["x"], output_var="y")

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)

  assert "y = self.conv1(x)" in src


def test_emit_call_functional_params(emitter_torch):
  """
  Scenario: Calling functional op with metadata args.
  Flatten(x, 1) -> torch.flatten(x, start_dim=1)  (if keyword) or positional.
  """
  # Metadata keys starting with arg_N map to positional in emitter
  node = LogicalNode(id="flat", kind="func_flatten", metadata={"arg_0": "1"})
  stmt = emitter_torch.emit_call(node, input_vars=["x"], output_var="z")

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)

  # torch.flatten comes from _resolve_api_name logic for "flatten"
  assert "z = torch.flatten(x, 1)" in src


def test_emit_call_multi_input(emitter_torch):
  """
  Scenario: Binary op (Add).
  """
  node = LogicalNode(id="add", kind="func_add")
  stmt = emitter_torch.emit_call(node, input_vars=["a", "b"], output_var="c")

  from ml_switcheroo.utils.node_diff import capture_node_source

  src = capture_node_source(stmt)

  assert "c = torch.add(a, b)" in src


def test_resolve_api_heuristics(emitter_torch, emitter_flax):
  """
  Verify API prefix generation.
  """
  # Torch
  assert emitter_torch._resolve_api_name("Conv2d") == "nn.Conv2d"
  assert emitter_torch._resolve_api_name("abs") == "torch.abs"
  assert emitter_torch._resolve_api_name("custom.pkg.Layer") == "custom.pkg.Layer"

  # Flax
  assert emitter_flax._resolve_api_name("Conv") == "nnx.Conv"
  assert emitter_flax._resolve_api_name("abs") == "jnp.abs"
