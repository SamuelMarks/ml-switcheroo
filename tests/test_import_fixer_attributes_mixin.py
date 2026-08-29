"""Test module."""

from typing import Dict, Optional, Set

import libcst as cst

from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeMixin


class MockFixer(AttributeMixin):
  """Docstring."""

  def __init__(
    self,
    path_to_alias: Optional[Dict[str, str]] = None,
    defined_names: Optional[Set[str]] = None,
    target_fw: Optional[str] = None,
  ) -> None:
    """Docstring."""
    if path_to_alias is not None:
      self._path_to_alias: Dict[str, str] = path_to_alias
    if defined_names is not None:
      self._defined_names: Set[str] = defined_names
    if target_fw is not None:
      self.target_fw: str = target_fw


def test_simplify_reexports_not_attribute() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer()
  # Not an attribute value
  node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert fixer._simplify_reexports(node) is node


def test_simplify_reexports_not_redundant() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer()
  # jax.numpy.sum
  node: cst.Attribute = cst.Attribute(
    value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")), attr=cst.Name("sum")
  )
  assert fixer._simplify_reexports(node) is node


def test_simplify_reexports_not_safe_root() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(defined_names={"my_module"})
  # unknown.module.X
  node: cst.Attribute = cst.Attribute(
    value=cst.Attribute(value=cst.Name("unknown"), attr=cst.Name("module")), attr=cst.Name("X")
  )
  assert fixer._simplify_reexports(node) is node


def test_simplify_reexports_success() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(defined_names={"nnx"})
  # nnx.module.Module -> nnx.Module
  node: cst.Attribute = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")), attr=cst.Name("Module")
  )
  simplified: cst.BaseExpression = fixer._simplify_reexports(node)
  assert isinstance(simplified, cst.Attribute)
  assert isinstance(simplified.value, cst.Name)
  assert simplified.value.value == "nnx"
  assert simplified.attr.value == "Module"


def test_leave_attribute_missing_path_to_alias() -> None:
  """Docstring."""
  fixer: AttributeMixin = AttributeMixin()  # No _path_to_alias set
  node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert fixer.leave_Attribute(node, node) is node


def test_leave_attribute_collapsing() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(path_to_alias={"jax.numpy": "jnp"}, defined_names={"jnp"})
  # original_node = jax.numpy.sum
  original_node: cst.Attribute = cst.Attribute(
    value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")), attr=cst.Name("sum")
  )
  updated: cst.BaseExpression = fixer.leave_Attribute(original_node, original_node)

  assert isinstance(updated, cst.Attribute)
  assert updated.attr.value == "sum"
  assert isinstance(updated.value, cst.Name)
  assert updated.value.value == "jnp"


def test_leave_attribute_no_collapsing() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(path_to_alias={"jax.numpy": "jnp"}, defined_names={"jnp"})
  original_node: cst.Attribute = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  updated: cst.BaseExpression = fixer.leave_Attribute(original_node, original_node)
  assert updated is original_node


def test_leave_attribute_with_simplify_after_collapse() -> None:
  """Docstring."""
  fixer: MockFixer = MockFixer(path_to_alias={"flax.nnx": "nnx"}, defined_names={"nnx"})
  # flax.nnx.module.Module -> nnx.module.Module -> nnx.Module
  original_node: cst.Attribute = cst.Attribute(
    value=cst.Attribute(value=cst.Attribute(value=cst.Name("flax"), attr=cst.Name("nnx")), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  updated: cst.BaseExpression = fixer.leave_Attribute(original_node, original_node)
  assert isinstance(updated, cst.Attribute)
  assert updated.attr.value == "Module"
  assert isinstance(updated.value, cst.Name)
  assert updated.value.value == "nnx"
