"""Unit tests for the ImportMixin class within the import fixer module.

This module contains tests verifying that import statements (both `import` and `from ... import`)
are correctly parsed, replaced, removed, or preserved according to the target framework mappings,
source frameworks, and other configuration settings in the `ResolutionPlan`.
"""

import libcst as cst
from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan, ImportReq


class MockFixer(ImportMixin):
  """A mock implementation of ImportMixin used for testing import resolution behavior.

  This mock fixer overrides necessary attributes and registers mock resolution plans
  (e.g., mapping PyTorch subcomponents to JAX equivalents) to isolate and test the
  transformation logic of the `ImportMixin` class.
  """

  def __init__(self):
    """Initializes the MockFixer with pre-configured resolution plans and mock settings.

    This sets up default mappings, such as mapping `"torch.nn"` to `"jax.nn"`, setting the source
    framework to `"torch"`, and initializing tracking structures like definition sets and satisfied injections.
    """
    self.plan = ResolutionPlan()
    self.source_fws = ["torch"]
    self.preserve_source = False
    self._defined_names = set()
    self._satisfied_injections = set()

    self.plan.mappings["torch.nn"] = ImportReq(module="jax", subcomponent="nn", alias="nn")
    self.plan.required_imports.append(ImportReq(module="jax", subcomponent="numpy", alias="jnp"))

  def _track_definition(self, node):
    """Tracks node definitions within the mock fixer (stub implementation).

    Args:
        node (Any): The AST node representing the defined symbol.
    """
    pass


def test_importmixin_leave_import_replacement():
  """Tests that a standard import of a source framework subcomponent is correctly replaced.

  This test verifies that `import torch.nn` is replaced with `import jax.nn as nn` according
  to the configured resolution plan, and that the target import is registered as a satisfied injection.
  """
  fixer = MockFixer()
  node = cst.parse_statement("import torch.nn").body[0]

  result = fixer.leave_Import(node, node)
  assert isinstance(result, cst.Import)
  assert len(result.names) == 1
  assert result.names[0].name.value.value == "jax"
  assert result.names[0].name.attr.value == "nn"
  assert result.names[0].asname.name.value == "nn"
  assert "jax.nn" in fixer._satisfied_injections


def test_importmixin_leave_import_remove():
  """Tests that an unmapped standard import of a source framework is removed.

  This test verifies that when `preserve_source` is `False`, importing an unmapped source framework
  module like `import torch` returns a `RemovalSentinel` to prune it from the final AST.
  """
  fixer = MockFixer()
  node = cst.parse_statement("import torch")

  result = fixer.leave_Import(node.body[0], node.body[0])
  assert isinstance(result, cst.RemovalSentinel)


def test_importmixin_leave_import_preserve():
  """Tests that standard imports of a source framework are preserved when `preserve_source` is True.

  This test verifies that `import torch` is left unmodified when the fixer is configured to
  preserve the original source framework imports.
  """
  fixer = MockFixer()
  fixer.preserve_source = True
  node = cst.parse_statement("import torch").body[0]

  result = fixer.leave_Import(node, node)
  assert cst.Module([]).code_for_node(result) == cst.Module([]).code_for_node(node)


def test_importmixin_leave_importfrom_remove_star():
  """Tests that wildcard/star imports from a source framework are removed.

  This test verifies that `from torch import *` is pruned (returns a `RemovalSentinel`) during
  the transformation to avoid polluting the target namespace with unmapped symbols.
  """
  fixer = MockFixer()
  node = cst.parse_statement("from torch import *")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert isinstance(result, cst.RemovalSentinel)


def test_importmixin_leave_importfrom_preserve_star():
  """Tests that wildcard/star imports from a source framework are preserved when `preserve_source` is True.

  This test verifies that `from torch import *` remains unmodified when source preservation is enabled.
  """
  fixer = MockFixer()
  fixer.preserve_source = True
  node = cst.parse_statement("from torch import *")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert cst.Module([]).code_for_node(result) == cst.Module([]).code_for_node(node.body[0])


def test_importmixin_leave_importfrom_replacement_subcomp():
  """Tests `from ... import` replacement when the mapping specifies a target subcomponent.

  This test verifies that `from torch import nn` is successfully transformed into `import jax.nn as nn`
  when the resolution plan maps `torch.nn` to `jax.nn`.
  """
  fixer = MockFixer()
  node = cst.parse_statement("from torch import nn")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert isinstance(result, cst.Import)
  assert len(result.names) == 1
  assert result.names[0].name.value.value == "jax"
  assert result.names[0].name.attr.value == "nn"
  assert result.names[0].asname.name.value == "nn"
  assert "jax.nn" in fixer._satisfied_injections


def test_importmixin_leave_importfrom_replacement_no_subcomp():
  """Tests `from ... import` replacement when the mapping does not specify a target subcomponent.

  This test verifies that `from torch import nn` is transformed into `import jax as jnn` when the
  resolution plan maps `torch.nn` directly to a module import with an alias.
  """
  fixer = MockFixer()
  fixer.plan.mappings["torch.nn"] = ImportReq(module="jax", subcomponent="", alias="jnn")
  node = cst.parse_statement("from torch import nn")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert isinstance(result, cst.Import)
  assert len(result.names) == 1
  assert result.names[0].name.value == "jax"
  assert result.names[0].asname.name.value == "jnn"
  assert "jax : jnn" in fixer._satisfied_injections


def test_importmixin_leave_importfrom_remove():
  """Tests that unmapped name imports from a source framework are removed.

  This test verifies that `from torch import tensor` returns a `RemovalSentinel` because the
  imported symbol/module has no corresponding resolution mapping.
  """
  fixer = MockFixer()
  node = cst.parse_statement("from torch import tensor")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert isinstance(result, cst.RemovalSentinel)


def test_importmixin_leave_importfrom_preserve():
  """Tests that unmapped name imports from a source framework are preserved when `preserve_source` is True.

  This test verifies that `from torch import tensor` remains intact when source preservation is enabled.
  """
  fixer = MockFixer()
  fixer.preserve_source = True
  node = cst.parse_statement("from torch import tensor")

  result = fixer.leave_ImportFrom(node.body[0], node.body[0])
  assert result is node.body[0]


def test_importmixin_leave_importfrom_empty_module():
  """Tests that relative imports with an empty module name are left unmodified.

  This test verifies that relative imports such as `from . import module` (represented as an `ImportFrom`
  node with `module=None`) are returned as-is by the `leave_ImportFrom` handler.
  """
  fixer = MockFixer()
  node = cst.parse_statement("from . import module").body[0]

  node = node.with_changes(module=None)
  result = fixer.leave_ImportFrom(node, node)
  assert result is node
