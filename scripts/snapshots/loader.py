"""Snapshot finding and loading utilities."""

import importlib
import importlib.resources
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_snapshot_entry(item: Dict[str, Any]) -> None:
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
    existing_names = {p.get("name") for p in params if isinstance(p, dict)}
    for ov in overloads:
      if isinstance(ov, dict) and isinstance(ov.get("params"), list):
        for ov_p in ov["params"]:
          if isinstance(ov_p, dict) and ov_p.get("name") not in existing_names:
            new_p = dict(ov_p)
            new_p["is_mandatory"] = False
            if new_p.get("default") is None:
              new_p["default"] = "None"
            params.append(new_p)
            existing_names.add(new_p.get("name"))
    for p in params:
      if isinstance(p, dict) and p.get("default") is None:
        pname = p.get("name")
        if any(
          isinstance(ov, dict)
          and isinstance(ov.get("params"), list)
          and not any(isinstance(op, dict) and op.get("name") == pname for op in ov["params"])
          for ov in overloads
        ):
          p["is_mandatory"] = False
          p["default"] = "None"

  pos_kw = [p for p in params if isinstance(p, dict) and p.get("kind") == "POSITIONAL_OR_KEYWORD"]
  pos_only = [p for p in params if isinstance(p, dict) and p.get("kind") == "POSITIONAL_ONLY"]
  kw_only = [p for p in params if isinstance(p, dict) and p.get("kind") == "KEYWORD_ONLY"]
  if pos_kw and pos_only and not overloads:
    item["overloads"] = [
      {"params": pos_kw + kw_only},
      {"params": pos_only + kw_only},
    ]

  first_line = doc.splitlines()[0] if doc else ""
  sig_defaults = set(re.findall(r"\b([a-zA-Z_]\w*)\s*(?::[^=,)]+)?\s*=", first_line))
  has_varargs = any(isinstance(p, dict) and p.get("kind") == "VAR_POSITIONAL" for p in params)

  args_match = re.search(
    r"(?:^|\n)\s*Args:\s*\n(.*?)(?:\n\s*(?:Returns|Shape|Attributes|Examples|Note|Warning|\.\. _):|\Z)",
    doc,
    re.DOTALL,
  )
  args_block = args_match.group(1) if args_match else ""

  for p in params:
    if not isinstance(p, dict):
      continue
    pname = str(p.get("name", ""))
    desc = str(p.get("description") or "")
    if p.get("default") is None:
      if pname in sig_defaults:
        p["is_mandatory"] = False
        p["default"] = "None"
      elif has_varargs and p.get("kind") == "KEYWORD_ONLY":
        p["is_mandatory"] = False
        p["default"] = "None"
      elif pname.startswith("_") or pname in ("rngs", "dtype", "name", "parent"):
        p["is_mandatory"] = False
        p["default"] = "None"
      elif "optional" in desc.lower() or "default" in desc.lower() or "if given" in desc.lower():
        p["is_mandatory"] = False
        p["default"] = "None"
      elif args_block and not desc and not re.search(rf"(?:^|\n)\s*{re.escape(pname)}\b", args_block):
        p["is_mandatory"] = False
        p["default"] = "None"
      elif args_block and re.search(rf"(?:^|\n)\s*{re.escape(pname)}\s*\([^)]*optional", args_block, re.IGNORECASE):
        p["is_mandatory"] = False
        p["default"] = "None"
      elif args_block and re.search(rf"(?:^|\n)\s*{re.escape(pname)}:[^\n]*optional", args_block, re.IGNORECASE):
        p["is_mandatory"] = False
        p["default"] = "None"
      elif "if not provided" in doc.lower() or "without arguments" in doc.lower() or "subclass this class" in doc.lower():
        p["is_mandatory"] = False
        p["default"] = "None"


def _augment_framework_aliases(fw: str, fw_snap: Dict[str, Any]) -> None:
  """Index top-level module aliases, dtype objects, and re-exported functions from snapshots.

  Args:
      fw: The canonical framework identifier.
      fw_snap: The flattened lookup dictionary for ``fw``.
  """
  if not fw_snap:
    return

  dtype_prefixes = {
    "torch": "torch",
    "jax": "jax.numpy",
    "flax_nnx": "jax.numpy",
    "mlx": "mlx.core",
    "numpy": "numpy",
    "keras": "numpy",
    "tensorflow": "tf",
  }
  if fw in dtype_prefixes:
    pfx = dtype_prefixes[fw]
    for dt in (
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
    ):
      entry = {
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
    fw_snap.setdefault(
      "jax.numpy.linalg.slogdet",
      {"api_path": "jax.numpy.linalg.slogdet", "params": [{"name": "a"}]},
    )
    fw_snap.setdefault(
      "jax.image.resize",
      {
        "api_path": "jax.image.resize",
        "params": [{"name": "image"}, {"name": "shape"}, {"name": "method", "default": "'linear'"}],
      },
    )
    fw_snap.setdefault(
      "jax.scipy.special.gammaln",
      {"api_path": "jax.scipy.special.gammaln", "params": [{"name": "x"}]},
    )
    fw_snap.setdefault("jax.default_backend", {"api_path": "jax.default_backend", "params": []})
    fw_snap.setdefault(
      "jax.vjp",
      {"api_path": "jax.vjp", "params": [{"name": "fun"}, {"name": "primals", "kind": "VAR_POSITIONAL"}]},
    )
  elif fw == "mlx":
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


def _resolve_grounding_engine() -> Any:
  """Resolve GroundingEngine class from available ecosystem or fallback packages.

  Returns:
      Any: The GroundingEngine class if resolvable, otherwise None.
  """
  for mod_name in ("ml_ecosystem_snapshots.grounding.engine", "ml_framework_snapshots.grounding.engine"):
    try:
      mod = importlib.import_module(mod_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
    try:
      parent_src = Path(__file__).resolve().parent.parent.parent / repo_name / "src"
      if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
      pkg_name = repo_name.replace("-", "_") + ".grounding.engine"
      mod = importlib.import_module(pkg_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  return None


GroundingEngine: Any = _resolve_grounding_engine()


def _hydrate_ir_snapshot(flat_snapshots: Dict[str, Dict[str, Any]]) -> None:
  """Hydrate the 'ir' framework snapshot from live ``ml_switcheroo_ir`` schemas and exports.

  Prioritizes sibling ``../ml-switcheroo-ir/src`` when present on disk so local IR updates
  are audited immediately without requiring a wheel reinstall.

  Args:
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  sibling_ir_src = Path(__file__).resolve().parent.parent.parent / "ml-switcheroo-ir" / "src"
  if sibling_ir_src.exists() and str(sibling_ir_src) not in sys.path:
    sys.path.insert(0, str(sibling_ir_src))

  try:
    import ml_switcheroo_ir
    from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
  except ImportError:
    return

  ir_snap = flat_snapshots.setdefault("ir", {})

  try:
    from ml_ecosystem_snapshots.frameworks.ir import collect_api
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    for ref in collect_api(SemanticTier.ARRAY_API):
      dumped = ref.model_dump()
      api_path = str(dumped["api_path"])
      name = str(dumped["name"])
      ir_snap.setdefault(api_path, dumped)
      ir_snap.setdefault(name, dumped)
      ir_snap.setdefault(f"sw_ir.{name}", dumped)
      ir_snap.setdefault(f"ir.{name}", dumped)
  except ImportError:
    pass

  for export_name in getattr(ml_switcheroo_ir, "__all__", []):
    entry = {"name": export_name, "api_path": f"ml_switcheroo_ir.{export_name}", "params": []}
    ir_snap.setdefault(export_name, entry)
    ir_snap.setdefault(f"ml_switcheroo_ir.{export_name}", entry)
    ir_snap.setdefault(f"sw_ir.{export_name}", entry)
    ir_snap.setdefault(f"ir.{export_name}", entry)

  for op_name, schema in {**ONNX_REGISTRY, **CUSTOM_OPS_REGISTRY}.items():
    params = [{"name": inp, "kind": "POSITIONAL_OR_KEYWORD"} for inp in schema.inputs] + [
      {
        "name": attr_name,
        "kind": "KEYWORD_ONLY",
        "default": str(attr.default) if attr.default is not None else None,
      }
      for attr_name, attr in schema.attributes.items()
    ]
    item = {"name": op_name, "api_path": f"{schema.domain}.{op_name}", "params": params}
    ir_snap.setdefault(op_name, item)
    ir_snap.setdefault(f"{schema.domain}.{op_name}", item)


def _flatten_single_framework(fw: str, snap: Any, flat_snapshots: Dict[str, Dict[str, Any]]) -> None:
  """Flattens a raw snapshot JSON structure into a fast lookup dictionary.

  Args:
      fw: The framework name key.
      snap: The raw snapshot data (dict or list of items).
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  if fw not in flat_snapshots:
    flat_snapshots[fw] = {}

  if isinstance(snap, list):
    for item in snap:
      if isinstance(item, dict) and "mnemonic" in item:
        flat_snapshots[fw][item["mnemonic"]] = item
    return

  if not isinstance(snap, dict):
    return

  for cat, items in snap.get("categories", {}).items():
    if isinstance(items, list):
      for item in items:
        if "api_path" in item:
          api_path = item["api_path"]
          flat_snapshots[fw][api_path] = item
          if api_path.startswith("jax.numpy."):
            flat_snapshots[fw]["jnp." + api_path[len("jax.numpy.") :]] = item
          elif api_path.startswith("mlx.core."):
            flat_snapshots[fw]["mx." + api_path[len("mlx.core.") :]] = item
          elif api_path.startswith("torch.nn.functional."):
            flat_snapshots[fw]["F." + api_path[len("torch.nn.functional.") :]] = item
          elif api_path.startswith("flax.nnx."):
            flat_snapshots[fw]["nnx." + api_path[len("flax.nnx.") :]] = item
        if "name" in item:
          flat_snapshots[fw][item["name"]] = item
        if "mnemonic" in item:
          flat_snapshots[fw][item["mnemonic"]] = item
        if "aliases" in item and isinstance(item["aliases"], list):
          for alias in item["aliases"]:
            flat_snapshots[fw][alias] = item
    elif isinstance(items, dict):
      for k, v in items.items():
        flat_snapshots[fw][k] = v

  for k, v in snap.get("functions", {}).items():
    flat_snapshots[fw][k] = v
  for k, v in snap.get("classes", {}).items():
    flat_snapshots[fw][k] = v
  if fw == "numpy":
    for np_builtin in (
      "numpy.dtype",
      "numpy.random.seed",
      "numpy.random.randn",
      "numpy.random.randint",
      "numpy.random.uniform",
    ):
      flat_snapshots[fw].setdefault(np_builtin, {"api_path": np_builtin})
  for item in snap.get("operations", []):
    if isinstance(item, dict):
      if "api_path" in item:
        flat_snapshots[fw][item["api_path"]] = item
      if "class_name" in item:
        flat_snapshots[fw][item["class_name"]] = item
      if "name" in item:
        flat_snapshots[fw][item["name"]] = item
      if "aliases" in item and isinstance(item["aliases"], list):
        for alias in item["aliases"]:
          flat_snapshots[fw][alias] = item
  for k, v in snap.items():
    if k not in [
      "categories",
      "functions",
      "classes",
      "operations",
      "version",
      "mappings",
      "templates",
      "imports",
      "structs",
    ]:
      if isinstance(v, dict) and (
        fw in ("nvidia_sass", "rdna")
        or "args" in v
        or any(key in v for key in ("alu", "memory", "control_flow", "tensor_core", "inputs"))
      ):
        flat_snapshots[fw][k] = v

  _augment_framework_aliases(fw, flat_snapshots[fw])
  for entry in flat_snapshots[fw].values():
    if isinstance(entry, dict):
      _normalize_snapshot_entry(entry)


def load_snapshots(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json` or exhaustive ISA files.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx', 'nvidia_sass') to the JSON data.
  """
  flat_snapshots: dict[str, Dict[str, Any]] = {}
  candidates = (
    list(snapshot_dir.glob("*_v*.json"))
    + list(snapshot_dir.glob("*_exhaustive.json"))
    + list(snapshot_dir.glob("*_isa.json"))
  )
  for file_path in candidates:
    if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
      continue
    if "_v" in file_path.name:
      fw = file_path.name.split("_v")[0]
    elif "sass" in file_path.name:
      fw = "nvidia_sass"
    elif "rdna" in file_path.name:
      fw = "rdna"
    else:
      fw = file_path.stem

    if fw == "amd_rdna" or "rdna" in fw:
      fw = "rdna"
    elif "sass" in fw:
      fw = "nvidia_sass"

    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      _flatten_single_framework(fw, snap, flat_snapshots)
      if fw == "pax":
        _flatten_single_framework("paxml", snap, flat_snapshots)
      elif fw == "stablehlo_exhaustive":
        _flatten_single_framework("stablehlo", snap, flat_snapshots)

  return flat_snapshots


def load_snapshots_multi(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
  """Load API snapshots from multiple directories.

  Args:
      snapshot_dirs: Optional list of directory paths to search for snapshot JSON files.

  Returns:
      Dictionary mapping framework keys to flattened symbol dictionaries.
  """
  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    for d in snapshot_dirs:
      dirs_to_search.append(d)
      if d.name == "snapshots" and (d.parent / "frameworks").exists():
        dirs_to_search.append(d.parent / "frameworks")
  else:
    for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
      try:
        snap_res = importlib.resources.files(pkg)
        dirs_to_search.append(Path(str(snap_res)))
      except Exception:
        pass
    for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
      try:
        parent_snap = Path(__file__).resolve().parent.parent.parent / repo_name / "src" / repo_name.replace("-", "_")
        if (parent_snap / "snapshots").exists():
          dirs_to_search.append(parent_snap / "snapshots")
        if (parent_snap / "frameworks").exists():
          dirs_to_search.append(parent_snap / "frameworks")
      except Exception:
        pass
    try:
      semantics_dir = Path(__file__).resolve().parent.parent / "src" / "ml_switcheroo" / "semantics"
      if semantics_dir.exists():
        dirs_to_search.append(semantics_dir)
    except Exception:
      pass

  flat_snapshots: dict[str, Dict[str, Any]] = {}
  for snapshot_dir in dirs_to_search:
    if not snapshot_dir.exists():
      continue
    candidates = (
      list(snapshot_dir.glob("*_v*.json"))
      + list(snapshot_dir.glob("*_exhaustive.json"))
      + list(snapshot_dir.glob("*_isa.json"))
    )
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      if "_v" in file_path.name:
        fw = file_path.name.split("_v")[0]
      elif "sass" in file_path.name:
        fw = "nvidia_sass"
      elif "rdna" in file_path.name:
        fw = "rdna"
      else:
        fw = file_path.stem

      if fw == "amd_rdna" or "rdna" in fw:
        fw = "rdna"
      elif "sass" in fw:
        fw = "nvidia_sass"

      with open(file_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
        _flatten_single_framework(fw, snap, flat_snapshots)
        if fw == "optax_shim":
          _flatten_single_framework("optax", snap, flat_snapshots)
        elif fw in ("pax", "praxis", "praxis_shim", "paxml_shim"):
          _flatten_single_framework("paxml", snap, flat_snapshots)
        elif fw == "stablehlo_exhaustive":
          _flatten_single_framework("stablehlo", snap, flat_snapshots)

  if flat_snapshots or any("ml-switcheroo-ir" in str(d) for d in dirs_to_search):
    _hydrate_ir_snapshot(flat_snapshots)

  return flat_snapshots


# Strict supplementary lookup tables for validated hardware ISA instructions.


def compute_snapshot_checksums(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, str]:
  """Computes SHA-256 checksums of discovered snapshot files.

  Args:
      snapshot_dirs: Optional list of directories to scan for snapshots.

  Returns:
      A dictionary mapping snapshot file names to their SHA-256 hexadecimal digests.
  """
  import hashlib

  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    dirs_to_search.extend(snapshot_dirs)
  else:
    dirs_to_search.extend(
      [
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
        Path("src/ml_switcheroo/semantics"),
      ]
    )

  checksums: Dict[str, str] = {}
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
