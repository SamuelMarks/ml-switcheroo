"""Unit tests for the AttributeMixin class in the import fixer core module.

This module validates that attributes referenced in source code nodes (such as
`jax.numpy.abs`) are correctly simplified and resolved using configuration-driven
mappings and re-export simplification logic.
"""

import libcst as cst
from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeMixin


class MockFixer(AttributeMixin):
  """A mock import fixer class used to test the functionality of `AttributeMixin`.

  This class implements the minimal interface and attribute state required to
  test attribute-matching and rewriting capabilities of the mixin.
  """

  def __init__(self):
    """Initializes the mock fixer with predefined namespace aliases and target frameworks.

    Sets up internal maps to simulate alias resolutions and define target frameworks
    during testing of attribute traversal.

    Returns:
        None
    """
    self._path_to_alias = {"jax.numpy": "jnp", "torch.nn.functional": "F"}
    self._defined_names = {"jnp", "F"}
    self.target_fw = "jax"


def test_attributemixin_leave_attribute():
  """Verifies that `leave_Attribute` simplifies attributes to their aliased names.

  This test parses an expression like `jax.numpy.abs` and checks that the fixer
  rewrites it to use the `jnp` alias configured in `_path_to_alias`.

  Returns:
      None
  """
  fixer = MockFixer()

  # jax.numpy.abs -> jnp.abs
  node = cst.parse_expression("jax.numpy.abs")
  updated = node

  result = fixer.leave_Attribute(node, updated)
  assert isinstance(result, cst.Attribute)
  assert result.value.value == "jnp"
  assert result.attr.value == "abs"


def test_attributemixin_simplify_reexports():
  """Ensures that nested or deep attributes are simplified correctly.

  This test checks that when a nested package hierarchy (such as
  `flax.nnx.module.Module`) is encountered, the fixer simplifies it
  to `flax.nnx.Module` by stripping intermediate internal submodules when appropriate.

  Returns:
      None
  """
  fixer = MockFixer()

  # flax.nnx.module.Module -> flax.nnx.Module (assuming nnx is defined)
  fixer._defined_names.add("flax")
  fixer.target_fw = "flax"
  node = cst.parse_expression("flax.nnx.module.Module")
  updated = node

  result = fixer.leave_Attribute(node, updated)
  # Since flax.nnx is not in path_to_alias, it falls through to _simplify_reexports
  assert isinstance(result, cst.Attribute)
  assert result.value.attr.value == "nnx"


def test_attributemixin_no_alias():
  """Validates that attributes without matching aliases are left unmodified.

  This test parses an attribute that has no predefined translation rule in the MockFixer
  and verifies that the returned node is exactly the same reference as the input.

  Returns:
      None
  """
  fixer = MockFixer()

  # Something.else -> Something.else
  node = cst.parse_expression("Something.other")
  updated = node

  result = fixer.leave_Attribute(node, updated)
  assert result is updated


def test_attributemixin_no_path_to_alias():
  """Ensures the fixer fails gracefully when the path-to-alias dictionary is missing.

  This test deletes `_path_to_alias` from the fixer and verifies that `leave_Attribute`
  returns the original node unmodified without raising unexpected errors.

  Returns:
      None
  """
  fixer = MockFixer()
  del fixer._path_to_alias

  node = cst.parse_expression("jax.numpy.abs")
  result = fixer.leave_Attribute(node, node)
  assert result is node


def test_attributemixin_simplify_no_attribute():
  """Verifies that non-Attribute nodes are ignored by `_simplify_reexports`.

  This test checks that passing a simple name node instead of an attribute hierarchy
  returns the unmodified input node unchanged.

  Returns:
      None
  """
  fixer = MockFixer()
  node = cst.parse_expression("my_var")
  # Not an attribute, shouldn't be simplified
  assert fixer._simplify_reexports(node) is node
