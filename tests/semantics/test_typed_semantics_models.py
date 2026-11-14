"""Comprehensive unit and branch tests for typed SemanticsManager models and query APIs."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import libcst as cst
import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.cli.matrix import CompatibilityMatrix
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, ParameterDef, PatternDef, Rule
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.hooks_registry import AutoWireSpec
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.plugins.casting import transform_casting
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.semantics import registry_loader
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.index_builder import IndexBuilder
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import (
  TypedSemanticsStore,
  _prepare_validated_op,
  _sync_overlay_operation_model,
  _sync_typed_framework_entry,
  merge_frameworks,
  merge_overlay_data,
  merge_tier_with_models,
)
from ml_switcheroo.semantics.registry_loader import RegistryLoader
from ml_switcheroo.semantics.resolution import (
  ResolutionEngine,
  SemanticResolver,
  query_framework_traits,
  query_operation,
  query_resolve_operation,
  query_variant,
)
from ml_switcheroo.semantics.schema import FrameworkAlias, FrameworkTraits, PluginTraits, StructuralTraits
from ml_switcheroo.semantics.validator import SemanticsValidator


def test_framework_variant_legacy_alias_normalization() -> None:
  """Verify FrameworkVariant normalizes imports, output_select, rule, and replace_With_graph keys."""
  already_typed = FrameworkVariant(api="torch.abs")
  assert FrameworkVariant.model_validate(already_typed).api == "torch.abs"
  with pytest.raises(ValueError, match="Input should be a valid dictionary"):
    FrameworkVariant.model_validate("not_a_dict")

  raw_legacy: dict[str, JsonValue] = {
    "api": "jnp.abs",
    "imports": ["import jax.numpy as jnp"],
    "output_select": 1,
    "rule": [{"if_arg": "axis", "is_val": 0, "use_api": "jnp.sum"}],
    "replace_With_graph": [{"id": "n0"}],
  }
  v = FrameworkVariant.model_validate(raw_legacy)
  assert v.api == "jnp.abs"
  assert v.required_imports == ["import jax.numpy as jnp"]
  assert v.output_select_index == 1
  assert len(v.dispatch_rules) == 1
  assert isinstance(v.dispatch_rules[0], Rule)
  assert v.replace_with_graph == [{"id": "n0"}]


def test_typed_merging_and_overlay_sync_branches() -> None:
  """Verify TypedSemanticsStore, merge_tier_with_models, and _sync_overlay_operation_model branches."""
  ops: dict[str, OperationDef] = {}
  fw_traits: dict[str, FrameworkTraits] = {}
  store = TypedSemanticsStore(operations=ops, framework_traits=fw_traits)

  # _sync_typed_framework_entry: None store and invalid raw config
  _sync_typed_framework_entry(None, "torch", {})
  _sync_typed_framework_entry(fw_traits, "bad_fw", {"tiers": ["not_a_valid_tier"]})
  assert "bad_fw" not in fw_traits

  master_fw: dict[str, dict[str, JsonValue]] = {}
  merge_frameworks(
    master_fw,
    {"jax": {"alias": {"module": "jax.numpy", "name": "jnp"}}},
    typed_configs=fw_traits,
  )
  assert "jax" in fw_traits
  assert fw_traits["jax"].alias == FrameworkAlias(module="jax.numpy", name="jnp")

  # _prepare_validated_op with OperationDef (both internal and external)
  base_op = OperationDef(operation="CustomOp", variants={"torch": FrameworkVariant(api="torch.custom")})
  internal_op, internal_dict = _prepare_validated_op("CustomOp", base_op, is_internal=True)
  assert internal_op.is_internal is True
  assert internal_dict["_is_internal"] is True
  ext_op, ext_dict = _prepare_validated_op("CustomOp", base_op, is_internal=False)
  assert ext_op.is_internal is False
  assert "_is_internal" not in ext_dict

  # merge_tier_with_models: non-dict no-op, __frameworks__, __patterns__, OperationDef, duplicate op
  data: dict[str, dict[str, JsonValue]] = {}
  origins: dict[str, str] = {}
  patterns: list[PatternDef] = []
  merge_tier_with_models(store, data, origins, master_fw, "not_a_dict", SemanticTier.ARRAY_API, patterns=patterns)
  assert data == {}

  from ml_switcheroo.semantics.merging import JsonValueLike

  content: dict[str, JsonValueLike] = {
    "__frameworks__": {"mlx": {"extends": "numpy"}},
    "__patterns__": [{"name": "Pat1", "sequence": ["Add", "Relu"], "replace_with": "FusedAddRelu"}],
    "CustomOp": base_op,
  }
  merge_tier_with_models(store, data, origins, master_fw, content, SemanticTier.NEURAL, patterns=patterns)
  assert "CustomOp" in store.operations
  assert "mlx" in store.framework_traits
  assert len(patterns) == 1

  # Branch: non-dict __frameworks__ and non-list __patterns__, plus identical duplicate when op missing from store.operations
  del store.operations["CustomOp"]
  merge_tier_with_models(
    store,
    data,
    origins,
    master_fw,
    {"__frameworks__": "bad", "__patterns__": "bad", "CustomOp": base_op},
    SemanticTier.NEURAL,
    patterns=patterns,
  )
  assert "CustomOp" in store.operations

  # Merge existing op with updated variant
  updated_op = OperationDef(
    operation="CustomOp",
    variants={"jax": FrameworkVariant(api="jnp.custom")},
  )
  merge_tier_with_models(store, data, origins, master_fw, {"CustomOp": updated_op}, SemanticTier.NEURAL)
  assert "jax" in store.operations["CustomOp"].variants

  with pytest.raises(TypeError, match="Expected mapping for operation 'BadScalarOp'"):
    merge_tier_with_models(
      store,
      data,
      origins,
      master_fw,
      {"BadScalarOp": "not_a_dict"},
      SemanticTier.EXTRAS,
    )

  # _sync_overlay_operation_model branches: orphan valid, orphan invalid, None variant, FrameworkVariant, dict valid & invalid
  _sync_overlay_operation_model(ops, "OrphanValid", "torch", {"operation": "OrphanValid", "variants": {}}, None)
  assert "OrphanValid" in ops
  _sync_overlay_operation_model(ops, "OrphanBad", "torch", {"std_args": "bad"}, None)
  assert "OrphanBad" not in ops

  _sync_overlay_operation_model(ops, "CustomOp", "torch", data["CustomOp"], None)
  assert ops["CustomOp"].variants["torch"] is None
  _sync_overlay_operation_model(ops, "CustomOp", "mlx", data["CustomOp"], FrameworkVariant(api="mlx.custom"))
  assert ops["CustomOp"].variants["mlx"] is not None
  _sync_overlay_operation_model(
    ops,
    "CustomOp",
    "keras",
    {"variants": {"keras": {"api": "keras.ops.custom"}}},
    {"api": "keras.ops.custom"},
  )
  assert ops["CustomOp"].variants["keras"] is not None
  from ml_switcheroo_ir.schema.ghost import StandardMap

  _sync_overlay_operation_model(
    ops,
    "CustomOp",
    "std_map_fw",
    {"variants": "not_a_dict"},
    StandardMap(api="std_map_fw.custom"),
  )
  assert ops["CustomOp"].variants["std_map_fw"] is not None
  _sync_overlay_operation_model(
    ops,
    "CustomOp",
    "bad_fw",
    {"variants": "not_a_dict"},
    {"output_select_index": "not_an_int"},
  )
  assert "bad_fw" not in ops["CustomOp"].variants
  _sync_overlay_operation_model(ops, "CustomOp", "ignored_int", data["CustomOp"], 123)

  # merge_overlay_data with typed_store and framework block
  templates: dict[str, dict[str, str]] = {}
  merge_overlay_data(
    data=data,
    key_origins=origins,
    framework_configs=master_fw,
    test_templates=templates,
    content={
      "__framework__": "numpy",
      "framework": {"alias": {"module": "numpy", "name": "np"}},
      "mappings": {"NewOrphanOp": {"api": "np.new_op"}},
    },
    filename="numpy_v2_map.json",
    typed_store=store,
  )
  assert "numpy" in store.framework_traits
  assert "NewOrphanOp" in store.operations


def test_loaders_and_index_builder_typed_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Verify KnowledgeBaseLoader, RegistryLoader, and IndexBuilder typed store and fallback branches.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Temporary directory path fixture.
  """

  class BareMgr(SemanticsManager):
    """Minimal SemanticsManager subclass omitting operations and framework_traits attributes."""

    def __init__(self) -> None:
      """Initialize minimal dictionaries without calling super().__init__."""
      self.data = {}
      self._key_origins = {}
      self.framework_configs = {}
      self.test_templates = {}
      self._known_rng_methods = set()
      self.known_magic_args = set()

  bare = BareMgr()
  kb_bare = KnowledgeBaseLoader(bare)
  assert kb_bare._typed_store() is None
  kb_bare._load_tier_content({"Abs": {"variants": {"torch": {"api": "torch.abs"}}}}, SemanticTier.ARRAY_API)
  assert "Abs" in bare.data

  rl_bare = RegistryLoader(bare)
  assert rl_bare._typed_store() is None
  assert rl_bare._ensure_typed_framework_traits("torch") is None
  rl_bare.index_variants()

  stub_adapter = mock.MagicMock()
  stub_adapter.import_alias = ("stub.nn", "snn")
  stub_adapter.inherits_from = "jax"
  stub_adapter.structural_traits = StructuralTraits(module_base="stub.Module")
  stub_adapter.plugin_traits = PluginTraits(has_numpy_compatible_arrays=True)
  stub_adapter.supported_tiers = [SemanticTier.NEURAL, SemanticTier.ARRAY_API]
  stub_adapter.rng_seed_methods = ["seed"]
  stub_adapter.declared_magic_args = ["rngs"]
  stub_adapter.specifications = {
    "StubLayer": OperationDef(operation="StubLayer", std_args=["x"], variants={}),
    "stub_fn": OperationDef(operation="stub_fn", std_args=["x"], variants={}),
  }
  stub_adapter.definitions = {
    "StubLayer": FrameworkVariant(api="stub.nn.StubLayer"),
    "stub_fn": FrameworkVariant(api="stub.stub_fn"),
  }

  # Test both real SemanticsManager (typed_store != None) and BareMgr (typed_store == None)
  real_mgr = SemanticsManager.empty()
  real_mgr.framework_configs["stub_fw"] = {}
  rl_real = RegistryLoader(real_mgr)
  rl_real._load_adapter_traits("stub_fw", stub_adapter)
  # Call a second time so _ensure_typed_framework_traits hits existing != None branch
  rl_real._load_adapter_traits("stub_fw", stub_adapter)
  rl_real._load_adapter_specs(stub_adapter)
  rl_real._load_adapter_definitions("stub_fw", stub_adapter)
  rl_real.index_variants()

  assert "stub_fw" in real_mgr.framework_traits
  assert real_mgr.framework_traits["stub_fw"].extends == "jax"
  assert ("StubLayer", "stub_fw") in real_mgr._typed_variant_cache

  # BareMgr through _load_adapter_traits, _load_adapter_specs, _load_adapter_definitions, and _hydrate_plugins
  bare.framework_configs["stub_fw"] = {}
  rl_bare._load_adapter_traits("stub_fw", stub_adapter)
  rl_bare._load_adapter_specs(stub_adapter)
  rl_bare._load_adapter_definitions("stub_fw", stub_adapter)

  monkeypatch.setattr(
    registry_loader.hooks,
    "get_all_hook_metadata",
    lambda: {"plug": AutoWireSpec(ops={"PlugOp": {"std_args": ["x"], "variants": {"stub_fw": {"api": "stub.plug"}}}})},
  )
  rl_real._hydrate_plugins()
  rl_bare._hydrate_plugins()
  assert "PlugOp" in real_mgr.operations
  assert "PlugOp" in bare.data

  # IndexBuilder collision where op_model is absent from manager.operations (raw-only entry)
  idx_mgr = SemanticsManager.empty()
  idx_mgr.data = {
    "LowOp": {"variants": {"torch": {"api": "torch.shared"}}},
    "HighOp": {"variants": {"torch": {"api": "torch.shared"}, "jax": {"api": "jnp.shared"}}},
  }
  idx_mgr._key_origins = {"LowOp": SemanticTier.EXTRAS.value, "HighOp": SemanticTier.ARRAY_API.value}
  IndexBuilder.build_index(idx_mgr)
  assert idx_mgr._reverse_index["torch.shared"][0] == "HighOp"

  # Also test IndexBuilder._reset_caches when attributes are missing on a subclass
  class NoIndexCachesMgr(SemanticsManager):
    """Subclass omitting super().__init__ to exercise IndexBuilder._reset_caches."""

    def __init__(self) -> None:
      """Set only data, framework_configs, and _key_origins."""
      self.data = {"OnlyOp": {"variants": {"torch": {"api": "torch.only"}}}}
      self.framework_configs = {}
      self._key_origins = {"OnlyOp": SemanticTier.ARRAY_API.value}

  no_cache_mgr = NoIndexCachesMgr()
  IndexBuilder.build_index(no_cache_mgr)
  assert no_cache_mgr._reverse_index["torch.only"][0] == "OnlyOp"


def test_semantics_manager_typed_queries_and_resolution_fallbacks(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """Verify get_operation, resolve_operation, get_variant, get_framework_traits, and SemanticResolver.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Temporary directory path fixture.
  """
  # SemanticResolver with OperationDef instances directly
  op_add = OperationDef(
    operation="Add",
    std_args=[ParameterDef(name="x"), ParameterDef(name="y")],
    variants={
      "torch": FrameworkVariant(api="torch.add"),
      "jax": FrameworkVariant(api="jnp.add"),
      "none_fw": None,
    },
  )
  resolver = SemanticResolver({"Add": op_add})
  assert resolver.resolve("torch", "torch.add") == "Add"
  assert resolver.resolve("torch", "torch.sub") is None
  assert resolver.dispatch("Add", "jax") == "jnp.add"
  assert resolver.dispatch("Add", "none_fw") is None
  assert resolver.dispatch("Missing", "jax") is None

  # SemanticsManager typed queries
  mgr = SemanticsManager.empty()
  mgr.operations["Add"] = op_add
  mgr.data["Add"] = op_add.model_dump(by_alias=True, exclude_unset=True)
  mgr.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}
  mgr.framework_configs["flax_nnx"] = {"extends": "jax"}
  mgr.framework_traits["only_typed_fw"] = FrameworkTraits(extends="jax")
  mgr._build_index()

  assert mgr.get_operation("Add") is op_add
  assert mgr.get_operation("add") is op_add
  assert mgr.get_operation("torch.add") is op_add
  assert mgr.get_operation("UnknownOp") is None

  # Raw data lazy coercion in get_operation & resolve_operation
  mgr.data["RawOnly"] = {"std_args": ["a"], "variants": {"torch": {"api": "torch.raw_only"}}}
  mgr._reverse_index["torch.raw_only"] = ("RawOnly", mgr.data["RawOnly"])
  coerced_raw = mgr.get_operation("RawOnly")
  assert coerced_raw is not None
  assert coerced_raw.operation == "RawOnly"
  del mgr.operations["RawOnly"]
  via_api = mgr.get_operation("torch.raw_only")
  assert via_api is not None
  assert via_api.operation == "RawOnly"
  mgr.data["InvalidRaw"] = {"std_args": "not_a_list"}
  assert mgr.get_operation("InvalidRaw") is None
  mgr._reverse_index["torch.bad_api"] = ("InvalidRaw", {"std_args": "not_a_list"})
  assert mgr.get_operation("torch.bad_api") is None

  # resolve_operation branches: _typed_reverse_index, ops with variants, ops without variants, raw fallback, invalid raw
  resolved_pair = mgr.resolve_operation("torch.add")
  assert resolved_pair is not None
  assert resolved_pair[0] == "Add"
  assert mgr.resolve_operation("Add") is not None
  mgr.operations["EmptyVars"] = OperationDef(operation="EmptyVars", variants={})
  empty_res = mgr.resolve_operation("EmptyVars")
  assert empty_res is not None
  assert empty_res[0] == "EmptyVars"
  del mgr.operations["RawOnly"]
  raw_resolved = mgr.resolve_operation("torch.raw_only")
  assert raw_resolved is not None
  assert raw_resolved[0] == "RawOnly"
  mgr._reverse_index["torch.bad_raw"] = ("InvalidRaw", {"std_args": "bad"})
  assert mgr.resolve_operation("torch.bad_raw") is None
  assert mgr.resolve_operation("totally.missing") is None

  # get_variant branches: direct, cached, inheritance (flax_nnx -> jax), case-alt, explicit None, raw fallback, invalid raw
  v_jax = mgr.get_variant("Add", "jax")
  assert v_jax is not None
  assert v_jax.api == "jnp.add"
  # Second call hits _typed_variant_cache
  assert mgr.get_variant("Add", "jax") is v_jax
  v_inh = mgr.get_variant("Add", "flax_nnx")
  assert v_inh is not None
  assert v_inh.api == "jnp.add"
  assert mgr.get_variant("Add", "none_fw") is None

  mgr.operations["lowerop"] = OperationDef(
    operation="lowerop",
    variants={"mlx": FrameworkVariant(api="mlx.lower")},
  )
  mgr.operations["Lowerop"] = OperationDef(operation="Lowerop", variants={})
  alt_v = mgr.get_variant("Lowerop", "mlx")
  assert alt_v is not None
  assert alt_v.api == "mlx.lower"

  del mgr.operations["RawOnly"]
  raw_var = mgr.get_variant("RawOnly", "torch")
  assert raw_var is not None
  assert raw_var.api == "torch.raw_only"
  mgr.data["BadVarOp"] = {"variants": {"torch": {"output_select_index": "bad_int"}}}
  assert mgr.get_variant("BadVarOp", "torch") is None
  assert mgr.get_variant("MissingOp", "torch") is None
  mgr.framework_configs["cyc_a"] = {"extends": "cyc_b"}
  mgr.framework_configs["cyc_b"] = {"extends": "cyc_a"}
  assert mgr.get_variant("Add", "cyc_a") is None

  # get_framework_traits branches: raw_conf present, only_typed_fw (raw_conf None), unknown fw, malformed raw_conf fallback
  assert mgr.get_framework_traits("jax").alias == FrameworkAlias(module="jax.numpy", name="jnp")
  assert mgr.get_framework_traits("only_typed_fw").extends == "jax"
  assert mgr.get_framework_traits("unconfigured_fw") == FrameworkTraits()

  coerced_malformed = ResolutionEngine._coerce_framework_traits(
    {
      "tiers": ["invalid_tier"],
      "traits": StructuralTraits(module_base="m.Base"),
      "plugin_traits": PluginTraits(supports_sharding=True),
      "alias": FrameworkAlias(module="m", name="m"),
      "stateful_call": {"method": "apply"},
      "extends": "base_fw",
    }
  )
  assert coerced_malformed.extends == "base_fw"
  assert coerced_malformed.traits is not None
  assert coerced_malformed.traits.module_base == "m.Base"
  assert coerced_malformed.plugin_traits is not None
  assert coerced_malformed.plugin_traits.supports_sharding is True
  assert coerced_malformed.alias == FrameworkAlias(module="m", name="m")
  assert coerced_malformed.stateful_call == {"method": "apply"}

  coerced_dicts = ResolutionEngine._coerce_framework_traits(
    {
      "tiers": ["invalid_tier"],
      "traits": {"module_base": "d.Base"},
      "plugin_traits": {"supports_dynamic_shapes": True},
      "alias": {"module": "d", "name": "d"},
    }
  )
  assert coerced_dicts.traits is not None
  assert coerced_dicts.traits.module_base == "d.Base"
  assert coerced_dicts.plugin_traits is not None
  assert coerced_dicts.plugin_traits.supports_dynamic_shapes is True
  assert coerced_dicts.alias == FrameworkAlias(module="d", name="d")

  # Overridden methods on monkeypatched SemanticsManager
  with monkeypatch.context() as mp:
    mp.setattr(mgr, "get_definition_by_id", lambda oid: {"std_args": ["z"]} if oid == "Ov" else None)
    assert mgr.get_operation("Ov") is not None
    assert mgr.get_operation("Missing") is None

    mp.setattr(
      mgr,
      "get_definition",
      lambda api: ("Ov", {"std_args": ["z"]}) if api == "ov.fn" else None,
    )
    assert mgr.resolve_operation("ov.fn") is not None
    assert mgr.resolve_operation("none.fn") is None

    mp.setattr(
      mgr,
      "resolve_variant",
      lambda oid, fw: {"api": "ov.api"} if oid == "Ov" else ({"output_select_index": "bad"} if oid == "Bad" else None),
    )
    ov_var = mgr.get_variant("Ov", "jax")
    assert ov_var is not None
    assert ov_var.api == "ov.api"
    assert mgr.get_variant("Bad", "jax") is None
    assert mgr.get_variant("NoneOp", "jax") is None

    mp.setattr(mgr, "get_framework_config", lambda fw: {"extends": "custom_base"})
    assert mgr.get_framework_traits("jax").extends == "custom_base"

  # update_definition with invalid & valid data
  monkeypatch.setattr("ml_switcheroo.semantics.manager.resolve_semantics_dir", lambda: tmp_path)
  assert SemanticsValidator.validate_and_dump("BadUpdate", {"std_args": "not_a_list"}) is None
  dumped_valid = SemanticsValidator.validate_and_dump("GoodUpdate", {"std_args": ["x"]})
  assert dumped_valid is not None
  assert dumped_valid["operation"] == "GoodUpdate"
  mgr.update_definition("UpdatedOp", {"std_args": ["x"], "variants": {"torch": {"api": "torch.updated_op"}}})
  assert "UpdatedOp" in mgr.operations
  assert mgr._typed_reverse_index["torch.updated_op"][0] == "UpdatedOp"


def test_polymorphic_query_helpers() -> None:
  """Verify query_* helpers on None, empty stubs, and model-returning stubs."""
  assert query_operation(None, "Add") is None
  assert query_resolve_operation(None, "torch.add") is None
  assert query_variant(None, "Add", "jax") is None
  assert query_framework_traits(None, "jax") == FrameworkTraits()

  empty = mock.MagicMock(spec=[])
  assert query_operation(empty, "Add") is None
  assert query_resolve_operation(empty, "torch.add") is None
  assert query_variant(empty, "Add", "jax") is None
  assert query_framework_traits(empty, "jax") == FrameworkTraits()

  op_instance = OperationDef(operation="DirectOp", std_args=["a"], metadata={"target_type": "Float32"}, variants={})
  var_instance = FrameworkVariant(api="jnp.direct")
  ft_instance = FrameworkTraits(extends="numpy")

  class ModelReturningStub:
    """Stub returning pre-validated Pydantic models from legacy method names."""

    def get_definition_by_id(self, abstract_id: str) -> OperationDef | dict[str, JsonValue] | None:
      """Return OperationDef for DirectOp, dict for DictOp, else None.

      Args:
          abstract_id: Operation ID.

      Returns:
          Model, dict, or None.
      """
      if abstract_id == "DirectOp":
        return op_instance
      if abstract_id == "DictOp":
        return {"std_args": ["b"], "variants": {}}
      return None

    def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Return tuple for valid/invalid raw definitions or None.

      Args:
          api_name: Concrete API name.

      Returns:
          Tuple or None.
      """
      if api_name == "ok.api":
        return ("OkOp", {"std_args": ["x"], "variants": {}})
      if api_name == "bad.api":
        bad_payload: dict[str, JsonValue] = {"std_args": "bad"}
        return ("BadOp", bad_payload)
      return None

    def resolve_variant(self, abstract_id: str, target_framework: str) -> FrameworkVariant | dict[str, JsonValue] | None:
      """Return FrameworkVariant, invalid dict, or None.

      Args:
          abstract_id: Operation ID.
          target_framework: Framework key.

      Returns:
          Model, dict, or None.
      """
      if abstract_id == "DirectOp":
        return var_instance
      if abstract_id == "BadVar":
        return {"output_select_index": "bad"}
      return None

    def get_framework_config(
      self, framework: str
    ) -> FrameworkTraits | dict[str, JsonValue | StructuralTraits | PluginTraits | FrameworkAlias] | None:
      """Return FrameworkTraits or None.

      Args:
          framework: Framework key.

      Returns:
          Model or None.
      """
      if framework == "typed_fw":
        return ft_instance
      return None

  stub = ModelReturningStub()
  assert query_operation(stub, "DirectOp") is op_instance
  assert query_operation(stub, "DictOp") is not None
  assert query_operation(stub, "NoneOp") is None
  assert query_resolve_operation(stub, "ok.api") is not None
  assert query_resolve_operation(stub, "bad.api") is None
  assert query_resolve_operation(stub, "none.api") is None
  assert query_variant(stub, "DirectOp", "jax") is var_instance
  assert query_variant(stub, "BadVar", "jax") is None
  assert query_variant(stub, "NoneOp", "jax") is None
  assert query_framework_traits(stub, "typed_fw") is ft_instance
  assert query_framework_traits(stub, "none_fw") == FrameworkTraits()


def test_consumer_typed_and_fallback_edges() -> None:
  """Verify consumer edge branches across BareSubclass, CompatibilityMatrix, StableHloEmitter, and plugins."""

  # Subclass omitting super().__init__ and framework_configs
  class BareSubclass(SemanticsManager):
    """Subclass omitting super().__init__ to exercise _ensure_typed_stores."""

    def __init__(self) -> None:
      """Initialize only data."""
      self.data = {}

  bare_sub = BareSubclass()
  assert bare_sub.get_framework_traits("jax") == FrameworkTraits()

  # CompatibilityMatrix typed operations path (direct + plugin + unmapped)
  mat_mgr = SemanticsManager.empty()
  mat_mgr.operations = {
    "DirectAdd": OperationDef(operation="DirectAdd", variants={"torch": FrameworkVariant(api="torch.add")}),
    "PluginOp": OperationDef(
      operation="PluginOp",
      variants={"torch": FrameworkVariant(api="torch.plug", requires_plugin="my_plug")},
    ),
  }
  mat_mgr.data = {k: v.model_dump(by_alias=True, exclude_unset=True) for k, v in mat_mgr.operations.items()}
  matrix = CompatibilityMatrix(mat_mgr)
  with mock.patch.object(matrix, "_get_sorted_engines", return_value=["torch", "jax"]):
    rows = matrix.get_json()
  assert rows[0]["torch"] == "✅"
  assert rows[0]["jax"] == "❌"
  assert rows[1]["torch"] == "🧩"

  # StableHloEmitter fallback when semantics stub lacks resolve_operation
  legacy_emitter_sem = mock.MagicMock(spec=["get_definition"])

  def _legacy_get_def(api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Return stablehlo mapping for torch.abs.

    Args:
        api_name: Concrete API path.

    Returns:
        Definition tuple or None.
    """
    if api_name == "torch.abs":
      return ("Abs", {"variants": {"stablehlo": {"api": "stablehlo.abs"}}})
    if api_name == "torch.no_shlo":
      return ("NoShlo", {"variants": {}})
    return None

  legacy_emitter_sem.get_definition.side_effect = _legacy_get_def
  emitter = StableHloEmitter(legacy_emitter_sem)
  mod_abs = cst.parse_module("def f(x):\n    return torch.abs(x)\n")
  shlo_mod = emitter.convert(mod_abs)
  assert "stablehlo.abs" in shlo_mod.to_text()
  mod_no_shlo = cst.parse_module("def f(x):\n    return torch.no_shlo(x)\n")
  assert emitter.convert(mod_no_shlo) is not None
  mod_missing = cst.parse_module("def f(x):\n    return torch.missing(x)\n")
  assert emitter.convert(mod_missing) is not None

  # casting.py and padding.py custom _NumpyCompatibleTraitsProtocol fallback
  class CustomDuckPluginTraits:
    """Custom non-PluginTraits object exposing has_numpy_compatible_arrays."""

    has_numpy_compatible_arrays = True

  duck_sem = mock.MagicMock(spec=SemanticsManager)
  duck_sem.get_framework_config.return_value = {"plugin_traits": CustomDuckPluginTraits()}
  duck_sem.get_definition_by_id.return_value = None

  duck_ctx = HookContext(
    semantics=duck_sem,
    config=RuntimeConfig(source_framework="torch", target_framework="jax"),
  )
  duck_ctx.current_op_id = "MissingCastOp"
  call_expr = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  assert transform_casting(call_expr, duck_ctx) is call_expr
  pad_expr = cst.ensure_type(cst.parse_expression("pad(x)"), cst.Call)
  assert transform_padding(pad_expr, duck_ctx) is pad_expr
