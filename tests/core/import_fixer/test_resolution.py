"""Test suite for the Import Resolver module."""

import libcst as cst
from unittest.mock import Mock

from ml_switcheroo.core.import_fixer.resolution import (
  ImportReq,
  _QualNameScanner,
  ImportResolver,
  _deduplicate,
)


def test_importreq_signature():
  """Verifies the behavior of ImportReq.signature."""
  req1 = ImportReq(module="torch")
  assert req1.signature == "torch"

  req2 = ImportReq(module="torch", subcomponent="nn")
  assert req2.signature == "torch.nn"

  req3 = ImportReq(module="jax", subcomponent="numpy", alias="jnp")
  assert req3.signature == "jax.numpy : jnp"

  # redundant alias
  req4 = ImportReq(module="torch", subcomponent="nn", alias="nn")
  assert req4.signature == "torch.nn"

  req5 = ImportReq(module="jax", alias="jax")
  assert req5.signature == "jax"


def test_deduplicate():
  """Verifies the behavior of _deduplicate."""
  reqs = [
    ImportReq(module="torch"),
    ImportReq(module="torch", alias="torch"),
    ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
    ImportReq(module="jax", subcomponent="numpy", alias="jnp"),
  ]
  deduped = _deduplicate(reqs)
  assert len(deduped) == 2


def test_qualnamescanner(monkeypatch):
  """Verifies the behavior of _QualNameScanner."""
  code = "import torch; torch.nn.Linear(10, 10); x = 5"
  tree = cst.parse_module(code)

  scanner = _QualNameScanner("torch.nn")
  tree.visit(scanner)
  assert scanner.found is True

  scanner_not_found = _QualNameScanner("flax.nnx")
  tree.visit(scanner_not_found)
  assert scanner_not_found.found is False

  scanner_name = _QualNameScanner("x")
  tree.visit(scanner_name)
  assert scanner_name.found is True

  scanner_name_not_found = _QualNameScanner("y")
  tree.visit(scanner_name_not_found)
  assert scanner_name_not_found.found is False

  # test already found logic
  scanner2 = _QualNameScanner("x")
  scanner2.found = True
  scanner2.visit_Name(cst.Name("x"))
  scanner2.visit_Attribute(cst.Attribute(value=cst.Name("x"), attr=cst.Name("y")))
  assert scanner2.found is True

  # test exception swallowing in visit_Attribute
  # Monkeypatch get_full_name to raise an exception
  monkeypatch.setattr(
    "ml_switcheroo.core.import_fixer.resolution.get_full_name", Mock(side_effect=ValueError("Test exception"))
  )
  scanner3 = _QualNameScanner("x")
  attr_fail = cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar"))
  scanner3.visit_Attribute(attr_fail)
  assert scanner3.found is False


def test_importresolver():
  """Verifies the behavior of ImportResolver."""
  mock_semantics = Mock()
  # Mock alias
  mock_semantics.get_framework_aliases.return_value = {"jax": ("jax.numpy", "jnp")}
  # Mock import map
  mock_semantics.get_import_map.return_value = {
    "torch.nn": ("flax", "nnx", "nnx"),
    "torch.nn.functional": ("jax.nn", None, None),
  }

  resolver = ImportResolver(semantics=mock_semantics)

  # Test tree where things are used
  code = "import jax; jnp.zeros(5); flax.nnx.Linear(); jax.nn.relu()"
  tree = cst.parse_module(code)

  plan = resolver.resolve(tree, "jax")

  # jax used, jnp used, flax.nnx used (nnx check name used), jax.nn used
  assert len(plan.required_imports) == 4

  # target framework check
  assert plan.required_imports[0].module == "jax"
  # alias check
  assert plan.required_imports[1].module == "jax.numpy"
  assert plan.required_imports[1].alias == "jnp"
  # map 1
  assert plan.required_imports[2].module == "flax"
  assert plan.required_imports[2].subcomponent == "nnx"
  assert plan.required_imports[2].alias == "nnx"
  # map 2
  assert plan.required_imports[3].module == "jax.nn"
  assert plan.required_imports[3].subcomponent is None

  # test not used path
  code_empty = "x = 1"
  tree_empty = cst.parse_module(code_empty)
  plan_empty = resolver.resolve(tree_empty, "jax")
  assert len(plan_empty.required_imports) == 0


def test_importresolver_full_path():
  """Verifies the behavior of ImportResolver using full paths."""
  mock_semantics = Mock()
  mock_semantics.get_framework_aliases.return_value = {"jax": ("jax.numpy", "jnp")}
  mock_semantics.get_import_map.return_value = {
    "torch.nn": ("flax", "nnx", "nnx"),
  }
  resolver = ImportResolver(semantics=mock_semantics)

  # Even if we use jax.numpy, it should detect it and inject alias
  code = "jax.numpy.zeros(5)"
  tree = cst.parse_module(code)
  plan = resolver.resolve(tree, "jax")
  assert len(plan.required_imports) == 2
  assert plan.required_imports[0].module == "jax"
  assert plan.required_imports[1].module == "jax.numpy"
  assert plan.required_imports[1].alias == "jnp"
