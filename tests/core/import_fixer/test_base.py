"""Unit tests for ``ImportFixerState`` and the composed ``ImportFixer`` transformer."""

import libcst as cst

from ml_switcheroo.core.import_fixer import (
  AttributeAliasSimplifier,
  ImportFixer,
  ImportFixerState,
  ImportInjector,
  ImportReq,
  ImportStatementRewriter,
  ResolutionPlan,
)


def test_import_fixer_state_defaults() -> None:
  """Verify default field initialization when constructing ``ImportFixerState`` directly."""
  plan = ResolutionPlan()
  state = ImportFixerState(plan=plan)

  assert state.plan is plan
  assert state.source_frameworks == set()
  assert state.used_names == set()
  assert state.defined_names == set()
  assert state.satisfied_injections == set()
  assert state.path_to_alias == {}
  assert state.target_framework is None


def test_import_fixer_state_create_variants() -> None:
  """Verify ``ImportFixerState.create`` normalizes source frameworks, used names, and plan metadata."""
  plan = ResolutionPlan(path_to_alias={"jax.numpy": "jnp"}, target_framework="jax")

  state_default = ImportFixerState.create(plan)
  assert state_default.plan is plan
  assert state_default.source_frameworks == set()
  assert state_default.used_names == set()
  assert state_default.defined_names == set()
  assert state_default.satisfied_injections == set()
  assert state_default.path_to_alias == {"jax.numpy": "jnp"}
  assert state_default.target_framework == "jax"

  state_str = ImportFixerState.create(plan, source_frameworks="torch", used_names={"x", "jnp"})
  assert state_str.source_frameworks == {"torch"}
  assert state_str.used_names == {"x", "jnp"}

  state_list = ImportFixerState.create(plan, source_frameworks=["torch", "numpy"])
  assert state_list.source_frameworks == {"torch", "numpy"}

  state_set = ImportFixerState.create(plan, source_frameworks={"torch", "keras"})
  assert state_set.source_frameworks == {"torch", "keras"}


def test_import_fixer_state_track_definition_with_alias() -> None:
  """Verify ``track_definition`` records local alias identifiers for ``as`` imports."""
  state = ImportFixerState.create(ResolutionPlan())

  alias_nn = cst.ImportAlias(
    name=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")),
    asname=cst.AsName(name=cst.Name("nn")),
  )
  state.track_definition(alias_nn)
  assert "nn" in state.defined_names

  alias_simple = cst.ImportAlias(
    name=cst.Name("a"),
    asname=cst.AsName(name=cst.Name("b")),
  )
  state.track_definition(alias_simple)
  assert "b" in state.defined_names

  tuple_asname = cst.AsName(name=cst.Tuple(elements=[]))
  from unittest.mock import MagicMock

  fake_alias = MagicMock(spec=cst.ImportAlias)
  fake_alias.asname = tuple_asname
  state.track_definition(fake_alias)


def test_import_fixer_state_track_definition_without_alias() -> None:
  """Verify ``track_definition`` records root package names for unaliased imports."""
  state = ImportFixerState.create(ResolutionPlan())

  alias_root = cst.ImportAlias(name=cst.Name("torch"))
  state.track_definition(alias_root)
  assert "torch" in state.defined_names

  alias_sub = cst.ImportAlias(name=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")))
  state.track_definition(alias_sub)
  assert "jax" in state.defined_names

  alias_deep = cst.ImportAlias(
    name=cst.Attribute(
      value=cst.Attribute(value=cst.Name("flax"), attr=cst.Name("nnx")),
      attr=cst.Name("module"),
    )
  )
  state.track_definition(alias_deep)
  assert "flax" in state.defined_names


def test_import_fixer_composition_and_properties() -> None:
  """Verify ``ImportFixer`` wires collaborators over shared ``ImportFixerState`` and exposes properties."""
  plan = ResolutionPlan(
    required_imports=[ImportReq(module="jax", subcomponent="numpy", alias="jnp")],
    path_to_alias={"jax.numpy": "jnp"},
    target_framework="jax",
  )
  fixer = ImportFixer(plan=plan, source_frameworks="torch", used_names={"jnp"})

  assert isinstance(fixer.state, ImportFixerState)
  assert isinstance(fixer.attribute_simplifier, AttributeAliasSimplifier)
  assert isinstance(fixer.import_rewriter, ImportStatementRewriter)
  assert isinstance(fixer.import_injector, ImportInjector)
  assert fixer.attribute_simplifier.state is fixer.state
  assert fixer.import_rewriter.state is fixer.state
  assert fixer.import_injector.state is fixer.state

  assert fixer.plan is plan
  assert fixer.source_frameworks == {"torch"}
  assert fixer.used_names == {"jnp"}
  assert fixer.target_framework == "jax"


def test_import_fixer_cst_traversal_delegation() -> None:
  """Verify ``ImportFixer`` delegates import rewriting, alias collapsing, and module injection."""
  plan = ResolutionPlan(
    required_imports=[ImportReq(module="jax", subcomponent="numpy", alias="jnp")],
    mappings={"torch.nn": ImportReq(module="flax", subcomponent="nnx", alias="nnx")},
    path_to_alias={"jax.numpy": "jnp"},
    target_framework="jax",
  )
  code = """\
import torch
from torch import nn
x = jax.numpy.abs(y)
z = nn.Linear(2, 4)
"""
  tree = cst.parse_module(code)
  fixer = ImportFixer(plan=plan, source_frameworks=["torch"], used_names={"jax", "numpy", "abs", "y", "nn", "x", "z"})
  transformed = tree.visit(fixer)

  output = transformed.code
  assert "import torch" not in output.splitlines()
  assert "import flax.nnx as nnx" in output
  assert "import jax.numpy as jnp" in output
  assert "jnp.abs(y)" in output
