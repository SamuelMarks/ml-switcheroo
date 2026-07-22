"""Test suite for the Macro Ops module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

SOURCE_CODE = "\nimport torch\n\ndef activation(x):\n    s = torch.swish(x)\n    m = torch.mish(x)\n    return s, m\n"
EXPECTED_JAX = "\nimport jax.numpy as jnp\nimport jax.nn as nn\n\ndef activation(x):\n    s = x * nn.sigmoid(x)\n    m = x * jnp.tanh(nn.softplus(x))\n    return s, m\n"


class MacroSemantics(SemanticsManager):
  """Test suite for the Macro Semantics component."""

  def __init__(self):
    """Initializes the MacroSemantics instance."""
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self._inject("Swish", ["x"], source="torch.swish", target_macro="{x} * jax.nn.sigmoid({x})")
    self._inject("Mish", ["x"], source="torch.mish", target_macro="{x} * jax.numpy.tanh(jax.nn.softplus({x}))")

  def get_all_rng_methods(self):
    """Gets all rng methods."""
    return set()

  def get_framework_config(self, framework):
    """Gets framework configuration."""
    return {}

  def _inject(self, name, args, source, target_macro):
    """Helper to  inject."""
    variants = {"torch": {"api": source}, "jax": {"macro_template": target_macro}}
    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[source] = (name, self.data[name])


def test_macro_expansion():
  """Verifies the behavior of macro expansion."""
  semantics = MacroSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_CODE)
  assert result.success, f"Failed: {result.errors}"
  code = result.code
  assert "x * jax.nn.sigmoid(x)" in code or "x*jax.nn.sigmoid(x)" in code.replace(" ", "")
  assert "tanh(jax.nn.softplus(x))" in code


def test_macro_argument_rename():
  """Verifies the behavior of macro argument rename."""
  mgr = SemanticsManager()
  mgr.data = {
    "Swish": {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.swish", "args": {"x": "input"}},
        "jax": {"macro_template": "{x} * sigmoid({x})"},
      },
    }
  }
  mgr._reverse_index = {"torch.swish": ("Swish", mgr.data["Swish"])}
  mgr._key_origins = {}
  mgr._providers = {}
  mgr._source_registry = {}
  mgr.framework_configs = {}
  mgr._known_rng_methods = set()
  mgr.get_all_rng_methods = lambda: set()
  mgr.get_framework_config = lambda f: {}
  code = "y = torch.swish(input=val)"
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=mgr, config=config)
  result = engine.run(code)
  assert result.success
  assert "val * sigmoid(val)" in result.code
