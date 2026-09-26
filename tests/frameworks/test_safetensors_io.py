"""Test suite for SafeTensors IO grounding and weight conversion."""

import json
import yaml

from ml_switcheroo.frameworks.common.jax_stack import JAXStackMixin
from ml_switcheroo.frameworks.mlx_io import MlxIOMixin
from ml_switcheroo.frameworks.torch_io import TorchIOMixin
from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir


class DummyTorchIO(TorchIOMixin):
  """Dummy class implementing TorchIOMixin for testing."""


class DummyMlxIO(MlxIOMixin):
  """Dummy class implementing MlxIOMixin for testing."""


class DummyJaxIO(JAXStackMixin):
  """Dummy class implementing JAXStackMixin for testing."""


def test_torch_safetensors_io_methods() -> None:
  """Tests PyTorch SafeTensors serialization imports and code snippets."""
  helper = DummyTorchIO()
  imports = helper.get_weight_conversion_imports()
  assert any("safetensors.torch" in line for line in imports)

  load_code = helper.get_weight_load_code("ckpt_path")
  assert "safetensors.torch.load_file(ckpt_path)" in load_code
  assert "torch.load(ckpt_path" in load_code

  save_code = helper.get_weight_save_code("weights_dict", "dest_path")
  assert "safetensors.torch.save_file(converted_state, dest_path)" in save_code
  assert "torch.save(converted_state, dest_path)" in save_code


def test_mlx_safetensors_io_methods() -> None:
  """Tests MLX SafeTensors serialization imports and code snippets."""
  helper = DummyMlxIO()
  imports = helper.get_weight_conversion_imports()
  assert "import mlx.core as mx" in imports

  load_code = helper.get_weight_load_code("ckpt_path")
  assert "mx.load(ckpt_path)" in load_code

  save_code = helper.get_weight_save_code("weights_dict", "dest_path")
  assert "mx.save_safetensors(dest_path, mlx_state)" in save_code


def test_jax_safetensors_io_methods() -> None:
  """Tests JAX Stack SafeTensors serialization imports and code snippets."""
  helper = DummyJaxIO()
  imports = helper.get_weight_conversion_imports()
  assert any("safetensors.flax" in line for line in imports)

  load_code = helper.get_weight_load_code("ckpt_path")
  assert "safetensors.flax.load_file(ckpt_path)" in load_code
  assert "checkpointer.restore(ckpt_path)" in load_code

  save_code = helper.get_weight_save_code("weights_dict", "dest_path")
  assert "safetensors.flax.save_file(params_tree, dest_path)" in save_code
  assert "checkpointer.save(dest_path, final_tree)" in save_code


def test_save_safetensors_odl_grounding() -> None:
  """Validates SaveSafetensors.yaml ODL definition against static snapshot."""
  odl_path = resolve_semantics_dir() / "odl" / "SaveSafetensors.yaml"
  assert odl_path.exists(), f"Missing {odl_path}"

  with odl_path.open("r", encoding="utf-8") as f:
    spec = yaml.safe_load(f)

  assert spec["operation"] == "SaveSafetensors"
  assert "variants" in spec
  assert "torch" in spec["variants"]
  assert "flax_nnx" in spec["variants"]
  assert "paxml" in spec["variants"]
  assert "mlx" in spec["variants"]

  # Check standard arguments
  arg_names = [arg["name"] for arg in spec["std_args"]]
  assert "tensors" in arg_names
  assert "filename" in arg_names
  assert "metadata" in arg_names

  # Verify template consistency
  torch_var = spec["variants"]["torch"]
  assert "safetensors.torch" in torch_var["required_imports"][0]
  assert "{tensors}" in torch_var["macro_template"]
  assert "{filename}" in torch_var["macro_template"]

  flax_var = spec["variants"]["flax_nnx"]
  assert "safetensors.flax" in flax_var["required_imports"][0]
  assert "{tensors}" in flax_var["macro_template"]
  assert "{filename}" in flax_var["macro_template"]

  mlx_var = spec["variants"]["mlx"]
  assert "mlx.core" in mlx_var["required_imports"][0]
  assert "mx.save_safetensors" in mlx_var["macro_template"]


def test_safetensors_snapshot_signatures() -> None:
  """Verifies parameter signatures from safetensors snapshot if available."""
  snap_path = resolve_snapshots_dir() / "safetensors_v0.7.0.json"
  if not snap_path.exists():
    return

  with snap_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

  items = []
  for cat in data.get("categories", {}).values():
    if isinstance(cat, list):
      items.extend(cat)

  by_path = {item["api_path"]: item for item in items if isinstance(item, dict) and "api_path" in item}

  # Verify torch signatures
  if "safetensors.torch.load_file" in by_path:
    params = [p["name"] for p in by_path["safetensors.torch.load_file"]["params"]]
    assert "filename" in params

  if "safetensors.torch.save_file" in by_path:
    params = [p["name"] for p in by_path["safetensors.torch.save_file"]["params"]]
    assert "tensors" in params
    assert "filename" in params

  # Verify flax signatures
  if "safetensors.flax.load_file" in by_path:
    params = [p["name"] for p in by_path["safetensors.flax.load_file"]["params"]]
    assert "filename" in params

  if "safetensors.flax.save_file" in by_path:
    params = [p["name"] for p in by_path["safetensors.flax.save_file"]["params"]]
    assert "tensors" in params
    assert "filename" in params
