"""End-to-end AST unit tests for `ApiPass`, `ApiTransformer`, and `AttributeRewriteHandler`."""

import importlib.metadata
import typing

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiPass, ApiTransformer
from ml_switcheroo.core.rewriter.passes.api_attr_mixin import AttributeRewriteHandler
from ml_switcheroo.core.rewriter.passes.plugins.dsl import ImportFixerPass
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.enums import RuntimeTensorOrValue, SemanticTier
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

_NL = chr(10)


def _build_in_memory_semantics() -> SemanticsManager:
  """Create an in-memory `SemanticsManager` populated with Torch -> JAX operations and traits.

  Returns:
      Configured `SemanticsManager` instance without mutating global singleton caches.
  """
  sem = SemanticsManager(load_defaults=False)
  sem.framework_configs["torch"] = {
    "version": "2.0.0",
    "alias": {"module": "torch", "name": "torch"},
    "traits": {
      "module_base": "torch.nn.Module",
      "functional_execution_method": "apply",
      "implicit_method_roots": ["torch"],
    },
  }
  sem.framework_configs["jax"] = {
    "version": "1.0.0",
    "alias": {"module": "jax.numpy", "name": "jnp"},
    "traits": {"module_base": "flax.nnx.Module"},
  }

  ops: dict[str, tuple[dict[str, JsonValue], str]] = {
    "Abs": (
      {
        "operation": "Abs",
        "std_args": ["x"],
        "variants": {
          "torch": {"api": "torch.abs"},
          "jax": {"api": "jnp.abs", "min_version": "0.1.0", "max_version": "2.0.0"},
        },
      },
      SemanticTier.ARRAY_API.value,
    ),
    "FutureOp": (
      {
        "operation": "FutureOp",
        "std_args": ["x"],
        "variants": {
          "torch": {"api": "torch.future_op"},
          "jax": {"api": "jnp.future_op", "min_version": "9.0.0"},
        },
      },
      SemanticTier.ARRAY_API.value,
    ),
    "LegacyOp": (
      {
        "operation": "LegacyOp",
        "std_args": ["x"],
        "variants": {
          "torch": {"api": "torch.legacy_op"},
          "jax": {"api": "jnp.legacy_op", "max_version": "0.5.0"},
        },
      },
      SemanticTier.ARRAY_API.value,
    ),
    "Linear": (
      {
        "operation": "Linear",
        "std_args": ["in_features", "out_features"],
        "variants": {
          "torch": {"api": "torch.nn.Linear"},
          "jax": {"api": "flax.nnx.Linear"},
        },
      },
      SemanticTier.NEURAL.value,
    ),
    "Float32": (
      {
        "operation": "Float32",
        "std_args": [],
        "variants": {
          "torch": {"api": "torch.float32"},
          "jax": {"api": "jnp.float32"},
        },
      },
      SemanticTier.EXTRAS.value,
    ),
    "InfConst": (
      {
        "operation": "InfConst",
        "std_args": [],
        "variants": {
          "torch": {"api": "torch.inf"},
          "jax": {"macro_template": "float('inf')"},
        },
      },
      SemanticTier.EXTRAS.value,
    ),
    "BrokenMacroConst": (
      {
        "operation": "BrokenMacroConst",
        "std_args": [],
        "variants": {
          "torch": {"api": "torch.broken_const"},
          "jax": {"macro_template": ":::"},
        },
      },
      SemanticTier.EXTRAS.value,
    ),
    "PluginConst": (
      {
        "operation": "PluginConst",
        "std_args": [],
        "variants": {
          "torch": {"api": "torch.plugin_const"},
          "jax": {"requires_plugin": "custom_attr_hook"},
        },
      },
      SemanticTier.EXTRAS.value,
    ),
    "EmptyConst": (
      {
        "operation": "EmptyConst",
        "std_args": [],
        "variants": {
          "torch": {"api": "torch.empty_const"},
          "jax": {"description": "no api or macro"},
        },
      },
      SemanticTier.EXTRAS.value,
    ),
    "OnlySrc": (
      {
        "operation": "OnlySrc",
        "std_args": ["x"],
        "variants": {
          "torch": {"api": "torch.only_src"},
        },
      },
      SemanticTier.ARRAY_API.value,
    ),
  }

  for op_id, (raw_def, tier) in ops.items():
    op_model = OperationDef.model_validate(raw_def)
    sem.data[op_id] = raw_def
    sem.operations[op_id] = op_model
    sem._key_origins[op_id] = tier
    variants = raw_def["variants"]
    assert isinstance(variants, dict)
    torch_var = variants["torch"]
    assert isinstance(torch_var, dict)
    api = torch_var["api"]
    assert isinstance(api, str)
    sem._reverse_index[api] = (op_id, raw_def)
    sem._typed_reverse_index[api] = (op_id, op_model)

  return sem


def test_api_pass_end_to_end_calls_attributes_and_stateful_tracking() -> None:
  """Verify `ApiPass` rewrites calls, dtype/constant attributes, functional `apply` tuples, and stateful layers."""
  sem = _build_in_memory_semantics()
  ctx = RewriterContext(
    semantics=sem,
    config=RuntimeConfig(source_framework="torch", target_framework="jax"),
  )
  source = _NL.join(
    [
      "import torch",
      "import torch.nn as nn",
      "from . import local_mod",
      "from math import *",
      "",
      "class PlainClass(dyn_base(), CustomBase):",
      "    pass",
      "",
      "class MyModel(nn.Module):",
      "    def __init__(self):",
      "        super().__init__()",
      "        self.fc = nn.Linear(4, 8)",
      "",
      "    def forward(self, x):",
      "        local_layer = (p, q) = nn.Linear(8, 2)",
      "        dt = torch.float32",
      "        lim = torch.inf",
      "        bad = torch.broken_const",
      "        plug = torch.plugin_const",
      "        emp = torch.empty_const",
      "        [] = obj.apply(vars, x)",
      "        dyn_res = (lambda: 1)()",
      "        x.only_src()",
      "        z = x.abs()",
      "        y, state = obj.apply(vars, x)",
      "        return torch.abs(x)",
    ]
  )
  rewritten = ApiPass().transform(cst.parse_module(source), ctx).code
  assert "self.fc = flax.nnx.Linear(4, 8)" in rewritten
  assert "local_layer = (p, q) = flax.nnx.Linear(8, 2)" in rewritten
  assert "dt = jnp.float32" in rewritten
  assert "lim = float('inf')" in rewritten
  assert "bad = torch.broken_const" in rewritten
  assert "plug = torch.plugin_const" in rewritten
  assert "emp = torch.empty_const" in rewritten
  assert "y = obj(x)" in rewritten
  assert "return jnp.abs(x)" in rewritten


def test_api_transformer_version_warnings_strict_errors_and_preambles(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify version constraint warnings, strict-mode escape-hatch wrapping, and preamble/signature injection.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  sem = _build_in_memory_semantics()
  ctx = RewriterContext(
    semantics=sem,
    config=RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True),
  )

  # FutureOp triggers min_version warning; unknown Torch op in strict_mode triggers EscapeHatch failure marker
  source = _NL.join(
    [
      "import torch",
      "a = torch.future_op(x)",
      "b = torch.legacy_op(x)",
      "c = torch.completely_unknown(x)",
      "d = torch.only_src(x)",
    ]
  )
  out = ApiPass().transform(cst.parse_module(source), ctx)
  assert "a = jnp.future_op(x)" in out.code
  assert "b = jnp.legacy_op(x)" in out.code
  assert EscapeHatch.START_MARKER in out.code
  assert "9.0.0" in out.code
  assert "0.5.0" in out.code

  # Module preamble deduplication, invalid preamble syntax resilience, and signature/docstring injection
  transformer = ApiTransformer(ctx)
  ctx.module_preamble = ["import os", "import os", "import ??? bad"]
  mod_out = transformer.leave_Module(cst.parse_module("x = 1"), cst.parse_module("x = 1"))
  assert mod_out.code.count("import os") == 1

  sig = SignatureContext()
  sig.injected_args.append(("rng", "Any"))
  sig.preamble_stmts.append("seed = 0")
  transformer.context.signature_stack.append(sig)
  transformer.context.scope_stack.append({"state_var"})
  assert transformer._is_stateful("state_var") is True

  fn_doc = cst.ensure_type(cst.parse_module(f"def f(self):{_NL}    'doc'{_NL}    pass").body[0], cst.FunctionDef)
  fn_rewritten = transformer.leave_FunctionDef(fn_doc, fn_doc)
  fn_code = cst.Module(body=[fn_rewritten]).code
  assert "def f(self, rng: Any):" in fn_code
  assert "'doc'" in fn_code
  assert "seed = 0" in fn_code

  # Single-line suite conversion and duplicate injected parameter guard
  one_line_fn = cst.FunctionDef(
    name=cst.Name("g"),
    params=cst.Parameters(params=[cst.Param(name=cst.Name("rng"))]),
    body=cst.SimpleStatementSuite(body=[cst.Pass()]),
  )
  indented = transformer._convert_to_indented_block(one_line_fn)
  assert isinstance(indented.body, cst.IndentedBlock)
  dedup_fn = transformer._inject_argument_to_signature(indented, "rng", "Any")
  assert len(dedup_fn.params.params) == 1

  # Variant import normalization across strings and dicts
  transformer._handle_variant_imports(
    {"required_imports": ["import sys", "json", {"module": "numpy", "alias": "np"}, {"module": "math"}]}
  )
  assert "import numpy as np" in transformer.context.module_preamble
  assert "import math" in transformer.context.module_preamble

  # `ImportFixerPass` DSL plugin helper
  fixed_mod = cst.parse_module("a = 1").visit(ImportFixerPass(required_imports=["os", "sys"]))
  assert "import os" in fixed_mod.code
  assert "import sys" in fixed_mod.code

  # Package metadata fallback when framework config lacks `version`
  sem.framework_configs["jax"] = {}
  monkeypatch.setattr(importlib.metadata, "version", lambda _pkg: "1.5.0")
  assert transformer.check_version_constraints("1.0.0", "2.0.0") is None


def test_api_helpers_and_attribute_rewrite_handler_edge_cases() -> None:
  """Verify `_cst_to_string`, `_is_module_alias`, `_is_framework_base`, and `AttributeRewriteHandler` fallbacks."""
  sem = _build_in_memory_semantics()
  cfg = RuntimeConfig(
    source_framework="torch",
    target_framework="jax",
    source_flavour="torchvision.transforms",
  )
  ctx = RewriterContext(semantics=sem, config=cfg)
  t = ApiTransformer(ctx)

  assert t._cst_to_string(cst.BinaryOperation(cst.Name("a"), cst.Add(), cst.Name("b"))) == "Add"
  assert t._cst_to_string(cst.Integer("1")) is None
  assert t._cst_to_string(cst.Attribute(value=cst.Call(func=cst.Name("f")), attr=cst.Name("x"))) is None
  assert t._create_name_node("a.b.c").deep_equals(cst.parse_expression("a.b.c"))

  assert t._is_module_alias(cst.Name("torchvision")) is True
  assert t._is_module_alias(cst.Name("torch")) is True
  assert t._is_module_alias(cst.Name("unknown_pkg")) is False
  assert t._is_framework_base("") is False
  assert t._is_framework_base("nn.Module") is True
  assert t._get_source_lifecycle_lists() == (set(), set())

  cached_traits = StructuralTraits(module_base="CustomBase")
  t._cached_target_traits = cached_traits
  assert t._get_target_traits() is cached_traits

  # Unverified op in `_get_mapping`
  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(sem, "is_verified", lambda _op_id: False)
    assert t._get_mapping("torch.abs", silent=False) is None

  # `AttributeRewriteHandler` host whose `source_traits` raises `AttributeError`
  class _NoTraitsHost:
    """Minimal `AttributeRewriteHost` raising `AttributeError` on `source_traits`."""

    def __init__(self) -> None:
      """Initialize context with two scope frames."""
      self.context = RewriterContext(semantics=SemanticsManager(load_defaults=False))
      self.context.scope_stack = [set(), set()]
      self.semantics = SemanticsManager(load_defaults=False)
      self.target_framework = "jax"

    @property
    def source_traits(self) -> StructuralTraits:
      """Simulate missing `source_traits` property.

      Returns:
          Never returns; raises `AttributeError`.

      Raises:
          AttributeError: Always raised.
      """
      raise AttributeError("missing source_traits")

    def _get_qualified_name(self, _node: cst.BaseExpression) -> str | None:
      """Return fixed qualified name.

      Args:
          _node: CST expression node.

      Returns:
          String `'foo'`.
      """
      return "foo"

    def _mark_stateful(self, _var_name: str) -> None:
      """No-op stateful marker.

      Args:
          _var_name: Variable name.
      """

    def _get_mapping(self, _name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Return `None` for mapping lookup.

      Args:
          _name: Qualified source name.
          silent: Silent lookup flag.

      Returns:
          Always `None`.
      """
      _ = silent
      return None

    def _handle_variant_imports(self, _variant: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
      """No-op import handler.

      Args:
          _variant: Variant mapping.
      """

    def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
      """Return `cst.Name` for `name_str`.

      Args:
          name_str: Identifier string.

      Returns:
          `cst.Name` node.
      """
      return cst.Name(name_str)

  stmt = cst.ensure_type(cst.parse_module("a = call()").body[0], cst.SimpleStatementLine).body[0]
  assign_stmt = cst.ensure_type(stmt, cst.Assign)
  assert AttributeRewriteHandler(_NoTraitsHost()).rewrite_assign(assign_stmt, assign_stmt) is assign_stmt

  # Empty `signature_stack` in `leave_FunctionDef` and empty `scope_stack` in `_mark_stateful`
  t.context.signature_stack.clear()
  bare_fn = cst.ensure_type(cst.parse_module("def h(): pass").body[0], cst.FunctionDef)
  t.context.scope_stack.append(set())
  assert t.leave_FunctionDef(bare_fn, bare_fn) is bare_fn
  t.context.scope_stack.clear()
  t._mark_stateful("ignored_when_no_scope")

  # Functional `apply` assignment whose tuple element is not a `BaseAssignTargetExpression`
  from unittest.mock import MagicMock

  apply_orig = cst.ensure_type(
    cst.ensure_type(cst.parse_module("y, s = obj.apply(v, x)").body[0], cst.SimpleStatementLine).body[0],
    cst.Assign,
  )
  fake_elem = MagicMock(spec=cst.Element)
  fake_elem.value = cst.Integer("1")
  fake_tuple = MagicMock(spec=cst.Tuple)
  fake_tuple.elements = [fake_elem]
  fake_target = MagicMock(spec=cst.AssignTarget)
  fake_target.target = fake_tuple
  fake_updated = MagicMock(spec=cst.Assign)
  fake_updated.targets = [fake_target]
  fresh_t = ApiTransformer(
    RewriterContext(
      semantics=_build_in_memory_semantics(),
      config=RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True),
    )
  )
  assert fresh_t.attr_handler.rewrite_assign(apply_orig, fake_updated) is fake_updated

  # Strict call rewrite when `semantics` lacks `resolve_op_id`
  from ml_switcheroo.core.rewriter.passes.api_call_mixin import CallRewriteHandler

  class _NoResolveSemantics:
    """Semantics stub without `resolve_op_id` or `_key_origins`."""

    def get_definition(self, _name: str) -> None:
      """Return `None` so `_get_mapping` finds no entry.

      Args:
          _name: Qualified API name.
      """

    def get_framework_config(self, _fw: str) -> dict[str, JsonValue]:
      """Return empty framework config dictionary.

      Args:
          _fw: Framework key.

      Returns:
          Empty config dict.
      """
      return {}

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(fresh_t.context, "semantics", _NoResolveSemantics())
    unk_call = cst.ensure_type(cst.parse_expression("torch.missing_fn(x)"), cst.Call)
    assert CallRewriteHandler(fresh_t).rewrite_call(unk_call, unk_call) is unk_call

  class _WithResolveSemantics(_NoResolveSemantics):
    """Semantics stub implementing `resolve_op_id`."""

    def resolve_op_id(self, _fw: str, _api: str) -> str:
      """Return resolved op identifier `'MissingFn'`.

      Args:
          _fw: Framework key.
          _api: Source API string.

      Returns:
          Canonical op ID string.
      """
      return "MissingFn"

  with pytest.MonkeyPatch.context() as mp:
    mp.setattr(fresh_t.context, "semantics", _WithResolveSemantics())
    unk_call2 = cst.ensure_type(cst.parse_expression("torch.missing_fn(x)"), cst.Call)
    assert CallRewriteHandler(fresh_t).rewrite_call(unk_call2, unk_call2) is unk_call2
