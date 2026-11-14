"""Test suite for the Jax Stack module."""

from pydantic import JsonValue

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin


class MockHighLevelAdapter(JAXStackMixin):
  """Minimal `JAXStackMixin` test double used in JAX stack tests."""


def test_get_serialization_imports() -> None:
  """Verify `adapter.get_serialization_imports` emits `import orbax.checkpoint` across JAX stack inputs."""
  adapter = MockHighLevelAdapter()
  imports: list[str] = adapter.get_serialization_imports()
  assert "import orbax.checkpoint" in imports


def test_get_weight_conversion_imports() -> None:
  """Verify get weight conversion imports works as expected."""
  adapter = MockHighLevelAdapter()
  imports: list[str] = adapter.get_weight_conversion_imports()
  assert "import jax.numpy as jnp" in imports
  assert "import orbax.checkpoint" in imports
  assert "from flax.traverse_util import unflatten_dict, flatten_dict" in imports


def test_device_syntax_cuda() -> None:
  """Verify `adapter.get_device_syntax` handles device syntax cuda as expected."""
  adapter = MockHighLevelAdapter()
  syntax: str = adapter.get_device_syntax("'cuda'")
  assert syntax == "jax.devices('gpu')[0]"


def test_device_syntax_cpu_index() -> None:
  """Verify `adapter.get_device_syntax` handles device syntax cpu index as expected."""
  adapter = MockHighLevelAdapter()
  syntax: str = adapter.get_device_syntax("'cpu'", "1")
  assert syntax == "jax.devices('cpu')[1]"


def test_serialization_syntax_orbax() -> None:
  """Verify serialization syntax orbax works as expected."""
  adapter = MockHighLevelAdapter()

  # test direct call branch
  save_code_direct: str = adapter.get_serialization_syntax("save", "dir", "state")
  assert "checkpointer().save" in save_code_direct.lower()

  load_code_direct: str = adapter.get_serialization_syntax("load", "dir")
  assert "checkpointer().restore" in load_code_direct.lower()

  # Use the mixin's newly added weight code
  save_code: str = adapter.get_weight_save_code("state", "dir")
  assert "PyTreeCheckpointer" in save_code
  assert "tuple_params" in save_code


def test_weight_load_code() -> None:
  """Verify `adapter.get_weight_load_code` emits `flatten_dict` for weight load code."""
  adapter = MockHighLevelAdapter()
  code: str = adapter.get_weight_load_code("path")
  assert "orbax.checkpoint.PyTreeCheckpointer" in code
  assert "raw_tree = checkpointer.restore(path)" in code
  assert "flatten_dict" in code


def test_tensor_conversion_syntax() -> None:
  """Verify `adapter.get_tensor_to_numpy_expr` handles tensor conversion syntax as expected."""
  adapter = MockHighLevelAdapter()
  code: str = adapter.get_tensor_to_numpy_expr("x")
  assert code == "np.array(x)"


def test_weight_save_code() -> None:
  """Verify `adapter.get_weight_save_code` emits `tuple_params =` for weight save code."""
  adapter = MockHighLevelAdapter()
  code: str = adapter.get_weight_save_code("state", "path")
  assert "tuple_params =" in code
  assert "unflatten_dict(tuple_params)" in code
  assert "orbax.checkpoint.PyTreeCheckpointer" in code
  assert "checkpointer.save(path" in code


def test_wiring_injection() -> None:
  """Verify `adapter._apply_stack_wiring` emits `import chex` for wiring injection."""
  adapter = MockHighLevelAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter._apply_stack_wiring(snapshot)
  mappings = snapshot.get("mappings", {})
  templates = snapshot.get("templates", {})
  assert isinstance(templates["import"], str)
  assert "import chex" in templates["import"]
  assert templates["to_numpy"] == "{res_var}"
  assert isinstance(mappings["Adam"], dict)
  assert mappings["Adam"]["api"] == "optax.adam"
  assert isinstance(mappings["step"], dict)
  assert mappings["step"]["requires_plugin"] == "optimizer_step"


def test_get_to_numpy_code() -> None:
  """Gets to NumPy code."""
  adapter = MockHighLevelAdapter()
  assert "np.asarray(obj)" in adapter.get_to_numpy_code()


def test_get_device_check_syntax() -> None:
  """Gets device check syntax."""
  adapter = MockHighLevelAdapter()
  assert adapter.get_device_check_syntax() == "len(jax.devices('gpu')) > 0"


def test_get_rng_split_syntax() -> None:
  """Gets rng split syntax."""
  adapter = MockHighLevelAdapter()
  assert adapter.get_rng_split_syntax("rng", "key") == "rng, key = jax.random.split(rng)"


def test_get_doc_url() -> None:
  """Gets documentation URL."""
  adapter = MockHighLevelAdapter()
  url: str | None = adapter.get_doc_url("jax.numpy.abs")
  assert url is not None
  assert "jax.numpy.abs.html" in url


def test_wiring_injection_existing_templates() -> None:
  """Verify `adapter._apply_stack_wiring` handles wiring injection existing templates as expected."""
  adapter = MockHighLevelAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {"templates": {"existing": "template"}}
  adapter._apply_stack_wiring(snapshot)
  assert "existing" in snapshot["templates"]
  assert "import" not in snapshot["templates"]


def test_serialization_syntax_invalid_op() -> None:
  """Verify serialization syntax invalid operation works as expected."""
  adapter = MockHighLevelAdapter()
  assert adapter.get_serialization_syntax("invalid", "dir") == ""
