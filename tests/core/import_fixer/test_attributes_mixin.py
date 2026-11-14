"""Unit tests for ``AttributeAliasSimplifier`` in the import fixer module."""

import libcst as cst

from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeAliasSimplifier
from ml_switcheroo.core.import_fixer.base import ImportFixerState
from ml_switcheroo.core.import_fixer.resolution import ResolutionPlan


def _make_simplifier(
  path_to_alias: dict[str, str] | None = None,
  defined_names: set[str] | None = None,
  target_framework: str | None = None,
) -> AttributeAliasSimplifier:
  """Construct an ``AttributeAliasSimplifier`` backed by configured ``ImportFixerState``.

  Args:
      path_to_alias: Optional mapping from dotted module paths to local aliases.
      defined_names: Optional set of already defined local identifier names.
      target_framework: Optional target framework name.

  Returns:
      Configured ``AttributeAliasSimplifier`` instance.
  """
  plan = ResolutionPlan(
    path_to_alias=path_to_alias if path_to_alias is not None else {},
    target_framework=target_framework,
  )
  state = ImportFixerState.create(plan)
  if defined_names is not None:
    state.defined_names.update(defined_names)
  return AttributeAliasSimplifier(state)


def test_simplify_reexports_non_attribute_value() -> None:
  """Verify ``simplify_reexports`` ignores single-level attributes whose value is a ``cst.Name``."""
  simplifier = _make_simplifier()
  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert simplifier.simplify_reexports(node) is node


def test_simplify_reexports_non_redundant_segment() -> None:
  """Verify ``simplify_reexports`` leaves attributes unchanged when the middle segment is non-redundant."""
  simplifier = _make_simplifier(defined_names={"jax"}, target_framework="jax")
  node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")),
    attr=cst.Name("sum"),
  )
  assert simplifier.simplify_reexports(node) is node


def test_simplify_reexports_unsafe_root() -> None:
  """Verify ``simplify_reexports`` preserves redundant segments when the root is not a known framework root."""
  simplifier_empty = _make_simplifier()
  nnx_node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  assert simplifier_empty.simplify_reexports(nnx_node) is nnx_node

  simplifier_other = _make_simplifier(defined_names={"my_module"})
  unknown_node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("unknown"), attr=cst.Name("module")),
    attr=cst.Name("X"),
  )
  assert simplifier_other.simplify_reexports(unknown_node) is unknown_node


def test_simplify_reexports_safe_via_defined_names() -> None:
  """Verify ``simplify_reexports`` strips redundant segments when root is in ``defined_names``."""
  simplifier = _make_simplifier(defined_names={"nnx", "flax"}, target_framework="flax")

  node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  simplified = simplifier.simplify_reexports(node)
  assert isinstance(simplified, cst.Attribute)
  assert isinstance(simplified.value, cst.Name)
  assert simplified.value.value == "nnx"
  assert simplified.attr.value == "Module"

  deep_node = cst.ensure_type(cst.parse_expression("flax.nnx.module.Module"), cst.Attribute)
  deep_res = simplifier.rewrite_attribute(deep_node, deep_node)
  assert isinstance(deep_res, cst.Attribute)
  assert isinstance(deep_res.value, cst.Attribute)
  assert deep_res.value.attr.value == "nnx"
  assert deep_res.attr.value == "Module"


def test_simplify_reexports_safe_via_target_framework() -> None:
  """Verify ``simplify_reexports`` strips redundant segments when root matches ``target_framework``."""
  simplifier = _make_simplifier(target_framework="nnx")
  node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  simplified = simplifier.simplify_reexports(node)
  assert isinstance(simplified, cst.Attribute)
  assert isinstance(simplified.value, cst.Name)
  assert simplified.value.value == "nnx"
  assert simplified.attr.value == "Module"


def test_simplify_reexports_safe_via_path_to_alias_values() -> None:
  """Verify ``simplify_reexports`` treats mapped alias values in ``path_to_alias`` as safe roots."""
  simplifier = _make_simplifier(path_to_alias={"foo.bar": "bar"}, target_framework=None)
  node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("bar"), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  simplified = simplifier.rewrite_attribute(node, node)
  assert simplified is not node
  assert isinstance(simplified, cst.Attribute)
  assert isinstance(simplified.value, cst.Name)
  assert simplified.value.value == "bar"


def test_rewrite_attribute_immediate_reexport_simplification() -> None:
  """Verify ``rewrite_attribute`` returns immediately when ``simplify_reexports`` rewrites ``updated_node``."""
  simplifier = _make_simplifier(target_framework="nnx")
  node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("nnx"), attr=cst.Name("module")),
    attr=cst.Name("Module"),
  )
  result = simplifier.rewrite_attribute(node, node)
  assert result is not node
  assert isinstance(result, cst.Attribute)
  assert isinstance(result.value, cst.Name)
  assert result.value.value == "nnx"


def test_rewrite_attribute_empty_path_to_alias() -> None:
  """Verify ``rewrite_attribute`` leaves attributes unchanged when ``path_to_alias`` is empty."""
  simplifier = _make_simplifier()
  node = cst.ensure_type(cst.parse_expression("jax.numpy.abs"), cst.Attribute)
  assert simplifier.rewrite_attribute(node, node) is node


def test_rewrite_attribute_collapse_to_alias() -> None:
  """Verify ``rewrite_attribute`` collapses qualified module paths to configured local aliases."""
  simplifier = _make_simplifier(
    path_to_alias={"jax.numpy": "jnp", "torch.nn.functional": "F"},
    defined_names={"jnp", "F"},
    target_framework="jax",
  )

  abs_node = cst.ensure_type(cst.parse_expression("jax.numpy.abs"), cst.Attribute)
  abs_res = simplifier.rewrite_attribute(abs_node, abs_node)
  assert isinstance(abs_res, cst.Attribute)
  assert isinstance(abs_res.value, cst.Name)
  assert abs_res.value.value == "jnp"
  assert abs_res.attr.value == "abs"

  sum_node = cst.Attribute(
    value=cst.Attribute(value=cst.Name("jax"), attr=cst.Name("numpy")),
    attr=cst.Name("sum"),
  )
  sum_res = simplifier.rewrite_attribute(sum_node, sum_node)
  assert isinstance(sum_res, cst.Attribute)
  assert isinstance(sum_res.value, cst.Name)
  assert sum_res.value.value == "jnp"
  assert sum_res.attr.value == "sum"


def test_rewrite_attribute_collapse_multi_part_suffix_and_simplify() -> None:
  """Verify ``rewrite_attribute`` rebuilds multi-segment suffixes and runs re-export simplification."""
  simplifier = _make_simplifier(path_to_alias={"flax.nnx": "nnx"}, defined_names={"nnx"})
  original_node = cst.Attribute(
    value=cst.Attribute(
      value=cst.Attribute(value=cst.Name("flax"), attr=cst.Name("nnx")),
      attr=cst.Name("module"),
    ),
    attr=cst.Name("Module"),
  )
  updated = simplifier.rewrite_attribute(original_node, original_node)
  assert isinstance(updated, cst.Attribute)
  assert isinstance(updated.value, cst.Name)
  assert updated.value.value == "nnx"
  assert updated.attr.value == "Module"

  linalg_simplifier = _make_simplifier(path_to_alias={"jax": "j"}, defined_names={"j"})
  svd_node = cst.ensure_type(cst.parse_expression("jax.scipy.linalg.svd"), cst.Attribute)
  svd_res = linalg_simplifier.rewrite_attribute(svd_node, svd_node)
  assert cst.Module([]).code_for_node(svd_res) == "j.scipy.linalg.svd"


def test_rewrite_attribute_unmatched_prefix_and_call_receiver() -> None:
  """Verify ``rewrite_attribute`` returns ``updated_node`` when no prefix matches or receiver is a call."""
  simplifier = _make_simplifier(path_to_alias={"jax.numpy": "jnp"}, defined_names={"jnp"})

  unmatched = cst.ensure_type(cst.parse_expression("Something.other"), cst.Attribute)
  assert simplifier.rewrite_attribute(unmatched, unmatched) is unmatched

  call_attr = cst.Attribute(value=cst.Call(func=cst.Name("a")), attr=cst.Name("b"))
  assert simplifier.rewrite_attribute(call_attr, call_attr) is call_attr

  empty_simplifier = _make_simplifier()
  assert empty_simplifier.rewrite_attribute(call_attr, call_attr) is call_attr
