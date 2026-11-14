"""Snapshot finding and loading utilities."""

import importlib
import importlib.resources
import json
import re
import sys
import typing
from pathlib import Path

from pydantic import JsonValue


class GroundingEngineProtocol(typing.Protocol):
  """Protocol for GroundingEngine from ml_ecosystem_snapshots."""

  def _discover_target_files(self, target: str) -> typing.Sequence[Path]:
    """Discover snapshot files for target framework.

    Args:
        target: Target framework identifier.

    Returns:
        Sequence of discovered snapshot file paths.
    """
    ...

  def has_symbol(self, target: str, symbol: str) -> bool:
    """Check if symbol exists in target snapshot.

    Args:
        target: Target framework identifier.
        symbol: Symbol path to check.

    Returns:
        True if symbol exists in snapshot.
    """
    ...

  def suggest_closest_symbol(self, target: str, symbol: str) -> str | None:
    """Suggest closest matching symbol in target snapshot.

    Args:
        target: Target framework identifier.
        symbol: Symbol path to match.

    Returns:
        Suggested symbol string or None.
    """
    ...


class GroundingEngineCtorProtocol(typing.Protocol):
  """Protocol for GroundingEngine constructor."""

  def __call__(self, base_dirs: typing.Sequence[str | Path] | None = None) -> GroundingEngineProtocol:
    """Construct a GroundingEngine instance.

    Args:
        base_dirs: Optional sequence of snapshot base directories.

    Returns:
        Constructed GroundingEngineProtocol instance.
    """
    ...


_DTYPE_PREFIXES: dict[str, str] = {
  "torch": "torch",
  "jax": "jax.numpy",
  "flax_nnx": "jax.numpy",
  "mlx": "mlx.core",
  "numpy": "numpy",
  "keras": "numpy",
  "tensorflow": "tf",
}

_STANDARD_DTYPES: tuple[str, ...] = (
  "bool",
  "bool_",
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
  "float16",
  "float32",
  "float64",
  "bfloat16",
  "complex64",
  "complex128",
  "single",
  "double",
)


def _merge_overload_params(params: list[JsonValue], overloads: list[JsonValue]) -> None:
  """Merge overload parameter definitions and mark overload-specific parameters optional.

  Args:
      params: Primary mutable parameter list on a snapshot entry.
      overloads: List of overload metadata dictionaries.
  """
  existing_names = {p.get("name") for p in params if isinstance(p, dict)}
  for ov in overloads:
    ov_params = ov.get("params") if isinstance(ov, dict) else None
    if not isinstance(ov_params, list):
      continue
    for ov_p in ov_params:
      if isinstance(ov_p, dict) and ov_p.get("name") not in existing_names:
        new_p = dict(ov_p)
        new_p["is_mandatory"] = False
        if new_p.get("default") is None:
          new_p["default"] = "None"
        params.append(new_p)
        existing_names.add(new_p.get("name"))

  for p in params:
    if not isinstance(p, dict) or p.get("default") is not None:
      continue
    pname = p.get("name")
    for ov in overloads:
      ov_params2 = ov.get("params") if isinstance(ov, dict) else None
      if isinstance(ov_params2, list) and not any(isinstance(op, dict) and op.get("name") == pname for op in ov_params2):
        p["is_mandatory"] = False
        p["default"] = "None"
        break


def _is_optional_param_from_docs(
  p: dict[str, JsonValue],
  sig_defaults: set[str],
  has_varargs: bool,
  args_block: str,
  doc_lower: str,
) -> bool:
  """Determine whether an un-defaulted C-extension parameter is optional from docstring clues.

  Args:
      p: Parameter metadata dictionary.
      sig_defaults: Parameter names showing ``=`` defaults on the docstring signature line.
      has_varargs: Whether the signature includes a ``*args`` ``VAR_POSITIONAL`` parameter.
      args_block: Extracted ``Args:`` docstring section text.
      doc_lower: Lowercased full docstring text.

  Returns:
      True if ``p`` should be marked optional with default ``"None"``.
  """
  pname = str(p.get("name", ""))
  desc_lower = str(p.get("description") or "").lower()
  if (
    pname in sig_defaults
    or (has_varargs and p.get("kind") == "KEYWORD_ONLY")
    or pname.startswith("_")
    or pname in ("rngs", "dtype", "name", "parent")
    or any(tok in desc_lower for tok in ("optional", "default", "if given"))
  ):
    return True
  if args_block:
    esc = re.escape(pname)
    if (
      (not desc_lower and not re.search(rf"(?:^|\n)\s*{esc}\b", args_block))
      or bool(re.search(rf"(?:^|\n)\s*{esc}\s*\([^)]*optional", args_block, re.IGNORECASE))
      or bool(re.search(rf"(?:^|\n)\s*{esc}:[^\n]*optional", args_block, re.IGNORECASE))
    ):
      return True
  return any(tok in doc_lower for tok in ("if not provided", "without arguments", "subclass this class"))


def _normalize_snapshot_entry(item: dict[str, JsonValue]) -> None:
  """Hydrate overload parameters and C-extension/docstring parameter defaults on a snapshot entry.

  Args:
      item: Snapshot dictionary for a single API symbol.
  """
  params = item.get("params")
  if not isinstance(params, list):
    return
  doc = str(item.get("docstring") or "")
  overloads = item.get("overloads")
  if isinstance(overloads, list) and overloads:
    _merge_overload_params(params, overloads)

  pos_kw: list[JsonValue] = [p for p in params if isinstance(p, dict) and p.get("kind") == "POSITIONAL_OR_KEYWORD"]
  pos_only: list[JsonValue] = [p for p in params if isinstance(p, dict) and p.get("kind") == "POSITIONAL_ONLY"]
  kw_only: list[JsonValue] = [p for p in params if isinstance(p, dict) and p.get("kind") == "KEYWORD_ONLY"]
  if pos_kw and pos_only and not overloads:
    item["overloads"] = [{"params": pos_kw + kw_only}, {"params": pos_only + kw_only}]

  first_line = doc.splitlines()[0] if doc else ""
  sig_defaults = set(re.findall(r"\b([a-zA-Z_]\w*)\s*(?::[^=,)]+)?\s*=", first_line))
  has_varargs = any(isinstance(p, dict) and p.get("kind") == "VAR_POSITIONAL" for p in params)
  args_match = re.search(
    r"(?:^|\n)\s*Args:\s*\n(.*?)(?:\n\s*(?:Returns|Shape|Attributes|Examples|Note|Warning|\.\. _):|\Z)",
    doc,
    re.DOTALL,
  )
  args_block = args_match.group(1) if args_match else ""
  doc_lower = doc.lower()

  for p in params:
    if (
      isinstance(p, dict)
      and p.get("default") is None
      and _is_optional_param_from_docs(p, sig_defaults, has_varargs, args_block, doc_lower)
    ):
      p["is_mandatory"] = False
      p["default"] = "None"


def _augment_dtype_and_builtin_aliases(fw: str, fw_snap: dict[str, dict[str, JsonValue]]) -> None:
  """Register scalar dtype constructors and Python primitive cast entries on ``fw_snap``.

  Args:
      fw: Canonical framework identifier.
      fw_snap: Flattened snapshot lookup dictionary for ``fw``.
  """
  if fw in _DTYPE_PREFIXES:
    pfx = _DTYPE_PREFIXES[fw]
    for dt in _STANDARD_DTYPES:
      entry: dict[str, JsonValue] = {
        "api_path": f"{pfx}.{dt}",
        "params": [
          {"name": "x", "default": "0", "is_mandatory": False},
          {"name": "value", "default": "0", "is_mandatory": False},
        ],
      }
      fw_snap.setdefault(f"{pfx}.{dt}", entry)
      if pfx == "jax.numpy":
        fw_snap.setdefault(f"jnp.{dt}", fw_snap[f"{pfx}.{dt}"])
      elif pfx == "mlx.core":
        fw_snap.setdefault(f"mx.{dt}", fw_snap[f"{pfx}.{dt}"])

  for builtin_type in ("float", "int", "bool", "str"):
    fw_snap.setdefault(
      builtin_type,
      {"api_path": builtin_type, "params": [{"name": "x", "default": "0", "is_mandatory": False}]},
    )


def _augment_jax_aliases(fw_snap: dict[str, dict[str, JsonValue]]) -> None:
  """Register standard ``jax.nn``, ``jax.numpy.linalg``, and ``jax.vjp`` entries on ``fw_snap``.

  Args:
      fw_snap: Flattened JAX or Flax NNX snapshot lookup dictionary.
  """
  if "jax.nn.gelu" in fw_snap:
    for act in (
      "relu",
      "elu",
      "selu",
      "silu",
      "sigmoid",
      "softplus",
      "hard_sigmoid",
      "hard_swish",
      "leaky_relu",
      "log_softmax",
    ):
      fw_snap.setdefault(
        f"jax.nn.{act}",
        {
          "api_path": f"jax.nn.{act}",
          "params": [
            {"name": "x"},
            {"name": "axis", "default": "-1"},
            {"name": "alpha", "default": "1.0"},
            {"name": "negative_slope", "default": "0.01"},
          ],
        },
      )
  fw_snap.setdefault(
    "jax.numpy.linalg.norm",
    {
      "api_path": "jax.numpy.linalg.norm",
      "params": [
        {"name": "x"},
        {"name": "ord", "default": "None"},
        {"name": "axis", "default": "None"},
        {"name": "keepdims", "default": "False"},
      ],
    },
  )
  fw_snap.setdefault("jax.numpy.linalg.slogdet", {"api_path": "jax.numpy.linalg.slogdet", "params": [{"name": "a"}]})
  fw_snap.setdefault(
    "jax.image.resize",
    {
      "api_path": "jax.image.resize",
      "params": [{"name": "image"}, {"name": "shape"}, {"name": "method", "default": "'linear'"}],
    },
  )
  fw_snap.setdefault("jax.scipy.special.gammaln", {"api_path": "jax.scipy.special.gammaln", "params": [{"name": "x"}]})
  fw_snap.setdefault("jax.default_backend", {"api_path": "jax.default_backend", "params": []})
  fw_snap.setdefault(
    "jax.vjp",
    {"api_path": "jax.vjp", "params": [{"name": "fun"}, {"name": "primals", "kind": "VAR_POSITIONAL"}]},
  )


def _augment_mlx_aliases(fw_snap: dict[str, dict[str, JsonValue]]) -> None:
  """Register ``mlx.core.array`` and MLX loss helper signatures on ``fw_snap``.

  Args:
      fw_snap: Flattened MLX snapshot lookup dictionary.
  """
  fw_snap.setdefault(
    "mlx.core.array",
    {"api_path": "mlx.core.array", "params": [{"name": "val"}, {"name": "dtype", "default": "None"}]},
  )
  fw_snap.setdefault("mx.array", fw_snap["mlx.core.array"])
  if "mlx.nn.losses.mse_loss" in fw_snap:
    fw_snap.setdefault(
      "mlx.nn.losses.cross_entropy",
      {
        "api_path": "mlx.nn.losses.cross_entropy",
        "params": [
          {"name": "logits"},
          {"name": "targets"},
          {"name": "weights", "default": "None"},
          {"name": "axis", "default": "-1"},
          {"name": "label_smoothing", "default": "0.0"},
          {"name": "reduction", "default": "'none'"},
        ],
      },
    )
    fw_snap.setdefault(
      "mlx.nn.losses.binary_cross_entropy",
      {
        "api_path": "mlx.nn.losses.binary_cross_entropy",
        "params": [
          {"name": "logits"},
          {"name": "targets"},
          {"name": "weights", "default": "None"},
          {"name": "with_logits", "default": "True"},
          {"name": "reduction", "default": "'mean'"},
        ],
      },
    )


def _augment_framework_aliases(fw: str, fw_snap: dict[str, dict[str, JsonValue]]) -> None:
  """Index top-level module aliases, dtype objects, and re-exported functions from snapshots.

  Args:
      fw: The canonical framework identifier.
      fw_snap: The flattened lookup dictionary for ``fw``.
  """
  if not fw_snap:
    return

  _augment_dtype_and_builtin_aliases(fw, fw_snap)

  if fw == "numpy":
    for np_sym in (
      "numpy.dtype",
      "numpy.generic",
      "numpy.ndarray",
      "numpy.random.seed",
      "numpy.random.randn",
      "numpy.random.randint",
      "numpy.random.uniform",
    ):
      fw_snap.setdefault(np_sym, {"api_path": np_sym, "params": []})
    if "numpy.multiply" in fw_snap:
      fw_snap.setdefault("numpy.ma.core.multiply", fw_snap["numpy.multiply"])
  elif fw == "torch":
    fw_snap.setdefault(
      "torch.tensor",
      {
        "api_path": "torch.tensor",
        "params": [
          {"name": "data"},
          {"name": "dtype", "default": "None"},
          {"name": "device", "default": "None"},
          {"name": "requires_grad", "default": "False"},
        ],
      },
    )
    fw_snap.setdefault("torch.from_numpy", {"api_path": "torch.from_numpy", "params": [{"name": "ndarray"}]})
  elif fw in ("jax", "flax_nnx"):
    _augment_jax_aliases(fw_snap)
  elif fw == "mlx":
    _augment_mlx_aliases(fw_snap)
  elif fw == "tensorflow":
    if "tf.convert_to_tensor" in fw_snap:
      fw_snap.setdefault("tensorflow.convert_to_tensor", fw_snap["tf.convert_to_tensor"])
    if "tf.data.Dataset" in fw_snap:
      fw_snap.setdefault(
        "tf.data.Dataset.from_tensor_slices",
        {
          "api_path": "tf.data.Dataset.from_tensor_slices",
          "params": [{"name": "tensors"}, {"name": "name", "default": "None"}],
        },
      )
  elif fw == "keras":
    if "keras.layers.Multiply" in fw_snap:
      fw_snap.setdefault(
        "keras.layers.multiply",
        {
          "api_path": "keras.layers.multiply",
          "params": [{"name": "inputs"}, {"name": "kwargs", "kind": "VAR_KEYWORD"}],
        },
      )
    fw_snap.setdefault(
      "keras.random.SeedGenerator",
      {"api_path": "keras.random.SeedGenerator", "params": [{"name": "seed", "default": "None"}]},
    )


def _is_grounding_engine_ctor(
  obj: GroundingEngineCtorProtocol | type | None,
) -> typing.TypeGuard[GroundingEngineCtorProtocol]:
  """Check if `obj` is a callable GroundingEngine constructor.

  Args:
      obj: Candidate object.

  Returns:
      True if callable.
  """
  return callable(obj)


def _resolve_grounding_engine() -> GroundingEngineCtorProtocol | None:
  """Resolve GroundingEngine class from ml_ecosystem_snapshots package.

  Returns:
      The GroundingEngine class if resolvable, otherwise None.
  """
  import inspect

  try:
    mod = importlib.import_module("ml_ecosystem_snapshots.grounding.engine")
    cls_obj = dict(inspect.getmembers(mod)).get("GroundingEngine")
    if _is_grounding_engine_ctor(cls_obj):
      return cls_obj
  except (ImportError, AttributeError):
    pass

  try:
    parent_src = Path(__file__).resolve().parent.parent.parent / "ml-ecosystem-snapshots" / "src"
    if parent_src.exists() and str(parent_src) not in sys.path:
      sys.path.insert(0, str(parent_src))
    mod = importlib.import_module("ml_ecosystem_snapshots.grounding.engine")
    cls_obj2 = dict(inspect.getmembers(mod)).get("GroundingEngine")
    if _is_grounding_engine_ctor(cls_obj2):
      return cls_obj2
  except (ImportError, AttributeError):
    pass

  return None


GroundingEngine: GroundingEngineCtorProtocol | None = _resolve_grounding_engine()


def _hydrate_ir_snapshot(flat_snapshots: dict[str, dict[str, dict[str, JsonValue]]]) -> None:
  """Hydrate the 'ir' framework snapshot from live ``ml_switcheroo_ir`` schemas and exports.

  Prioritizes sibling ``../ml-switcheroo-ir/src`` when present on disk so local IR updates
  are audited immediately without requiring a wheel reinstall.

  Args:
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  sibling_ir_src = Path(__file__).resolve().parent.parent.parent.parent / "ml-switcheroo-ir" / "src"
  if sibling_ir_src.exists() and str(sibling_ir_src) not in sys.path:
    sys.path.insert(0, str(sibling_ir_src))

  try:
    import ml_switcheroo_ir
    from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
    from ml_switcheroo_ir.schema.framework_registries import (
      CHEX_REGISTRY,
      GRAIN_REGISTRY,
      OPTAX_REGISTRY,
      ORBAX_REGISTRY,
    )
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
  except ImportError:
    return

  ir_snap = flat_snapshots.setdefault("ir", {})

  try:
    from ml_ecosystem_snapshots.frameworks.ir import collect_api
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    for ref in collect_api(SemanticTier.ARRAY_API):
      dumped: dict[str, JsonValue] = ref.model_dump()
      api_path = str(dumped["api_path"])
      name = str(dumped["name"])
      ir_snap.setdefault(api_path, dumped)
      ir_snap.setdefault(name, dumped)
      ir_snap.setdefault(f"sw_ir.{name}", dumped)
      ir_snap.setdefault(f"ir.{name}", dumped)
  except ImportError:
    pass

  for export_name in ml_switcheroo_ir.__all__:
    entry: dict[str, JsonValue] = {
      "name": str(export_name),
      "api_path": f"ml_switcheroo_ir.{export_name}",
      "params": [],
    }
    ir_snap.setdefault(str(export_name), entry)
    ir_snap.setdefault(f"ml_switcheroo_ir.{export_name}", entry)
    ir_snap.setdefault(f"sw_ir.{export_name}", entry)
    ir_snap.setdefault(f"ir.{export_name}", entry)

  for op_name, schema in {
    **ONNX_REGISTRY,
    **CUSTOM_OPS_REGISTRY,
    **CHEX_REGISTRY,
    **GRAIN_REGISTRY,
    **OPTAX_REGISTRY,
    **ORBAX_REGISTRY,
  }.items():
    params: list[JsonValue] = [{"name": inp, "kind": "POSITIONAL_OR_KEYWORD"} for inp in schema.inputs]
    params.extend(
      {
        "name": attr_name,
        "kind": "KEYWORD_ONLY",
        "default": str(attr.default) if attr.default is not None else None,
      }
      for attr_name, attr in schema.attributes.items()
    )
    item: dict[str, JsonValue] = {"name": op_name, "api_path": f"{schema.domain}.{op_name}", "params": params}
    ir_snap.setdefault(op_name, item)
    ir_snap.setdefault(f"{schema.domain}.{op_name}", item)


_API_PATH_SHORTCUTS: tuple[tuple[str, str], ...] = (
  ("jax.numpy.", "jnp."),
  ("mlx.core.", "mx."),
  ("torch.nn.functional.", "F."),
  ("flax.nnx.", "nnx."),
)

_RESERVED_SNAP_KEYS: frozenset[str] = frozenset(
  {
    "categories",
    "functions",
    "classes",
    "operations",
    "version",
    "mappings",
    "templates",
    "imports",
    "structs",
  }
)


def _index_category_item(item: dict[str, JsonValue], target: dict[str, dict[str, JsonValue]]) -> None:
  """Index a single categorized GhostRef/snapshot item under its API path, short alias, name, and aliases.

  Args:
      item: Categorized snapshot dictionary entry.
      target: Mutable flattened lookup dictionary for the framework.
  """
  raw_api = item.get("api_path")
  if isinstance(raw_api, str):
    target[raw_api] = item
    for long_pfx, short_pfx in _API_PATH_SHORTCUTS:
      if raw_api.startswith(long_pfx):
        target[short_pfx + raw_api[len(long_pfx) :]] = item
        break

  for key_field in ("name", "mnemonic"):
    val = item.get(key_field)
    if isinstance(val, str):
      target[val] = item

  raw_aliases = item.get("aliases")
  if isinstance(raw_aliases, list):
    for alias in raw_aliases:
      if isinstance(alias, str):
        target[alias] = item


def _index_categories_section(raw_cats: JsonValue, target: dict[str, dict[str, JsonValue]]) -> None:
  """Flatten the ``categories`` dictionary of a snapshot into ``target``.

  Args:
      raw_cats: Raw ``categories`` value from snapshot JSON.
      target: Mutable flattened lookup dictionary for the framework.
  """
  if not isinstance(raw_cats, dict):
    return
  for items in raw_cats.values():
    if isinstance(items, list):
      for item in items:
        if isinstance(item, dict):
          _index_category_item(item, target)
    elif isinstance(items, dict):
      target.update({k: v for k, v in items.items() if isinstance(v, dict)})


def _index_dict_section(raw_section: JsonValue, target: dict[str, dict[str, JsonValue]]) -> None:
  """Copy dictionary entries from ``functions`` or ``classes`` sections into ``target``.

  Args:
      raw_section: Raw ``functions`` or ``classes`` value from snapshot JSON.
      target: Mutable flattened lookup dictionary for the framework.
  """
  if isinstance(raw_section, dict):
    target.update({k: v for k, v in raw_section.items() if isinstance(v, dict)})


def _index_operations_section(raw_ops: JsonValue, target: dict[str, dict[str, JsonValue]]) -> None:
  """Flatten the ``operations`` list of a v0.0.6 snapshot into ``target``.

  Args:
      raw_ops: Raw ``operations`` value from snapshot JSON.
      target: Mutable flattened lookup dictionary for the framework.
  """
  if not isinstance(raw_ops, list):
    return
  for item in raw_ops:
    if not isinstance(item, dict):
      continue
    for key_field in ("api_path", "class_name", "name"):
      val = item.get(key_field)
      if isinstance(val, str):
        target[val] = item
    raw_aliases = item.get("aliases")
    if isinstance(raw_aliases, list):
      for alias in raw_aliases:
        if isinstance(alias, str):
          target[alias] = item


def _index_root_isa_entries(fw: str, snap: dict[str, JsonValue], target: dict[str, dict[str, JsonValue]]) -> None:
  """Index top-level ISA/instruction dictionaries directly under ``snap``.

  Args:
      fw: Framework/dialect identifier.
      snap: Raw snapshot dictionary.
      target: Mutable flattened lookup dictionary for ``fw``.
  """
  isa_keys = ("alu", "memory", "control_flow", "tensor_core", "inputs")
  for k, v in snap.items():
    if k in _RESERVED_SNAP_KEYS or not isinstance(v, dict):
      continue
    if fw in ("nvidia_sass", "rdna") or "args" in v or any(key in v for key in isa_keys):
      target[k] = v


def _flatten_single_framework(
  fw: str,
  snap: JsonValue,
  flat_snapshots: dict[str, dict[str, dict[str, JsonValue]]],
) -> None:
  """Flattens a raw snapshot JSON structure into a fast lookup dictionary.

  Args:
      fw: The framework name key.
      snap: The raw snapshot data (dict or list of items).
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  target = flat_snapshots.setdefault(fw, {})

  if isinstance(snap, list):
    for item in snap:
      if isinstance(item, dict) and isinstance(item.get("mnemonic"), str):
        target[str(item["mnemonic"])] = item
    return

  if not isinstance(snap, dict):
    return

  _index_categories_section(snap.get("categories", {}), target)
  _index_dict_section(snap.get("functions", {}), target)
  _index_dict_section(snap.get("classes", {}), target)

  if fw == "numpy":
    for np_builtin in (
      "numpy.dtype",
      "numpy.random.seed",
      "numpy.random.randn",
      "numpy.random.randint",
      "numpy.random.uniform",
    ):
      target.setdefault(np_builtin, {"api_path": np_builtin})

  _index_operations_section(snap.get("operations", []), target)
  _index_root_isa_entries(fw, snap, target)

  _augment_framework_aliases(fw, target)
  for entry in target.values():
    _normalize_snapshot_entry(entry)


def _canonicalize_snapshot_fw_key(file_path: Path) -> str:
  """Infer canonical framework identifier from a snapshot JSON filename.

  Args:
      file_path: Path to a candidate snapshot JSON file.

  Returns:
      Canonical framework identifier string.
  """
  if "_v" in file_path.name:
    fw = file_path.name.split("_v")[0].removesuffix("_shim")
  elif "sass" in file_path.name:
    fw = "nvidia_sass"
  elif "rdna" in file_path.name:
    fw = "rdna"
  else:
    fw = file_path.stem

  if fw == "amd_rdna" or "rdna" in fw:
    return "rdna"
  if "sass" in fw:
    return "nvidia_sass"
  return fw


def _load_directory_into_snapshots(
  snapshot_dir: Path,
  flat_snapshots: dict[str, dict[str, dict[str, JsonValue]]],
  alias_praxis: bool = False,
) -> None:
  """Scan ``snapshot_dir`` for JSON snapshot artifacts and flatten into ``flat_snapshots``.

  Args:
      snapshot_dir: Directory containing snapshot JSON files.
      flat_snapshots: Mutable multi-framework snapshot dictionary.
      alias_praxis: Whether ``praxis`` should also be aliased to ``paxml``.
  """
  candidates = (
    list(snapshot_dir.glob("*_v*.json"))
    + list(snapshot_dir.glob("*_exhaustive.json"))
    + list(snapshot_dir.glob("*_isa.json"))
  )
  pax_keys = ("pax", "praxis") if alias_praxis else ("pax",)
  for file_path in candidates:
    if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
      continue
    fw = _canonicalize_snapshot_fw_key(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      _flatten_single_framework(fw, snap, flat_snapshots)
      if fw in pax_keys:
        _flatten_single_framework("paxml", snap, flat_snapshots)
      elif fw == "stablehlo_exhaustive":
        _flatten_single_framework("stablehlo", snap, flat_snapshots)


def load_snapshots(snapshot_dir: Path) -> dict[str, dict[str, dict[str, JsonValue]]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json` or exhaustive ISA files.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx', 'nvidia_sass') to the JSON data.
  """
  flat_snapshots: dict[str, dict[str, dict[str, JsonValue]]] = {}
  _load_directory_into_snapshots(snapshot_dir, flat_snapshots, alias_praxis=False)
  return flat_snapshots


def _resolve_snapshot_search_dirs(snapshot_dirs: list[Path] | None) -> list[Path]:
  """Expand explicit or default snapshot search directories.

  Args:
      snapshot_dirs: Optional caller-provided directory list.

  Returns:
      Ordered list of directories to scan for snapshot JSON files.
  """
  dirs_to_search: list[Path] = []
  if snapshot_dirs is not None:
    for d in snapshot_dirs:
      dirs_to_search.append(d)
      if d.name == "snapshots" and (d.parent / "frameworks").exists():
        dirs_to_search.append(d.parent / "frameworks")
      dirs_to_search.extend(
        sub
        for sub in (
          d / "src" / "ml_ecosystem_snapshots" / "snapshots",
          d / "src" / "ml_ecosystem_snapshots" / "frameworks",
          d / "snapshots",
        )
        if sub.exists()
      )
    return dirs_to_search

  repo_parent = Path(__file__).resolve().parent.parent.parent
  parent_snap = repo_parent / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots"
  dirs_to_search.extend(
    candidate
    for candidate in (
      parent_snap / "snapshots",
      parent_snap / "frameworks",
      repo_parent / "ml-ecosystem-snapshots" / "snapshots",
    )
    if candidate.exists()
  )
  try:
    snap_res = importlib.resources.files("ml_ecosystem_snapshots.snapshots")
    dirs_to_search.append(Path(str(snap_res)))
  except (ImportError, ModuleNotFoundError, TypeError):
    pass
  semantics_dir = repo_parent / "src" / "ml_switcheroo" / "semantics"
  if semantics_dir.exists():
    dirs_to_search.append(semantics_dir)
  return dirs_to_search


def load_snapshots_multi(
  snapshot_dirs: list[Path] | None = None,
  strict: bool = False,
) -> dict[str, dict[str, dict[str, JsonValue]]]:
  """Load API snapshots from multiple directories.

  Args:
      snapshot_dirs: Optional list of directory paths to search for snapshot JSON files.
      strict: If True, raises FileNotFoundError when no valid snapshots are discovered.

  Returns:
      Dictionary mapping framework keys to flattened symbol dictionaries.

  Raises:
      FileNotFoundError: If ``strict`` is True and no snapshot artifacts could be loaded.
  """
  dirs_to_search = _resolve_snapshot_search_dirs(snapshot_dirs)
  flat_snapshots: dict[str, dict[str, dict[str, JsonValue]]] = {}

  for snapshot_dir in dirs_to_search:
    if snapshot_dir.exists():
      _load_directory_into_snapshots(snapshot_dir, flat_snapshots, alias_praxis=True)

  if flat_snapshots or any("ml-switcheroo-ir" in str(d) for d in dirs_to_search):
    _hydrate_ir_snapshot(flat_snapshots)

  if strict and not flat_snapshots:
    raise FileNotFoundError(
      "No valid framework snapshots found in ../ml-ecosystem-snapshots or configured snapshot directories."
    )

  return flat_snapshots


# Strict supplementary lookup tables for validated hardware ISA instructions.


def compute_snapshot_checksums(snapshot_dirs: list[Path] | None = None) -> dict[str, str]:
  """Computes SHA-256 checksums of discovered snapshot files.

  Args:
      snapshot_dirs: Optional list of directories to scan for snapshots.

  Returns:
      A dictionary mapping snapshot file names to their SHA-256 hexadecimal digests.
  """
  import hashlib

  dirs_to_search: list[Path] = []
  if snapshot_dirs is not None:
    dirs_to_search.extend(snapshot_dirs)
  else:
    dirs_to_search.extend(
      [
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
        Path("../ml-ecosystem-snapshots/snapshots"),
        Path("src/ml_switcheroo/semantics"),
      ]
    )

  checksums: dict[str, str] = {}
  for d in dirs_to_search:
    if not d.exists():
      continue
    candidates = sorted(set(list(d.glob("*_v*.json")) + list(d.glob("*_exhaustive.json")) + list(d.glob("*_isa.json"))))
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      try:
        data = file_path.read_bytes()
        if len(data) > 0:
          checksums[file_path.name] = hashlib.sha256(data).hexdigest()
      except OSError as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to read snapshot %s: %s", file_path, exc)
  return checksums
