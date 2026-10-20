"""Tests for ODL variant parity expansion tool."""

import typing
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402
import yaml  # noqa: E402

import scripts.expand_variant_parity as expander  # noqa: E402


def test_expand_op_variants_from_snapshots() -> None:
  """Test expanding missing variants using candidate names and paxml/flax fallback."""
  odl_data = {
    "operation": "OpWithTorch",
    "variants": {
      "torch": {"api": "torch.torch_fn"},
      "paxml": {"api": "jnp.fallback_fn"},
    },
  }

  snapshots = {
    "jax": {
      "fallback_fn": {
        "api_path": "jax.numpy.fallback_fn",
        "params": [{"name": "a", "kind": "POSITIONAL_OR_KEYWORD"}],
      }
    },
    "mlx": {
      "torch_fn": {
        "api_path": "mlx.core.torch_fn",
        "params": [],
      }
    },
    "keras": {},
  }

  added = expander.expand_op_variants(odl_data, snapshots)  # type: ignore[arg-type]
  assert added == 2
  assert "jax" in odl_data["variants"]
  assert odl_data["variants"]["jax"]["api"] == "jax.numpy.fallback_fn"  # type: ignore[index]
  assert "mlx" in odl_data["variants"]
  assert odl_data["variants"]["mlx"]["api"] == "mlx.core.torch_fn"  # type: ignore[index]


def test_expand_op_variants_isa_alu_macro() -> None:
  """Test expanding RDNA and SASS variants from ALU and macro mappings."""
  odl_data = {
    "operation": "BitwiseAnd",
    "variants": {
      "torch": {"api": "torch.bitwise_and"},
      "jax": {"api": "jax.numpy.bitwise_and"},
      "mlx": {"api": "mlx.core.bitwise_and"},
      "keras": {"api": "keras.ops.bitwise_and"},
      "rdna": {"api": "existing_rdna"},
      "nvidia_sass": {"api": "existing_sass"},
    },
  }

  snapshots: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"jax": {}, "mlx": {}, "keras": {}}
  # Both rdna and sass already present
  added = expander.expand_op_variants(odl_data, snapshots)  # type: ignore[arg-type]
  assert added == 0

  # Missing rdna and sass
  odl_data_missing = {
    "operation": "BitwiseAnd",
    "variants": {
      "torch": {"api": "torch.bitwise_and"},
      "jax": {"api": "jax.numpy.bitwise_and"},
      "mlx": {"api": "mlx.core.bitwise_and"},
      "keras": {"api": "keras.ops.bitwise_and"},
    },
  }
  added = expander.expand_op_variants(odl_data_missing, snapshots)  # type: ignore[arg-type]
  assert added == 2
  assert odl_data_missing["variants"]["rdna"]["api"] == "v_and_b32"  # type: ignore[index]
  assert odl_data_missing["variants"]["nvidia_sass"]["api"] == "LOP3_LUT"  # type: ignore[index]


def test_expand_op_variants_math_entry_already_present() -> None:
  """Test expanding variants from CANONICAL_MATH_MAP when some variants already exist."""
  odl_data = {
    "operation": "Linear",
    "variants": {
      "flax_nnx": {"api": "flax.nnx.Linear", "args": {}},
    },
  }
  snapshots: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"jax": {}, "mlx": {}, "keras": {}}
  added = expander.expand_op_variants(odl_data, snapshots)  # type: ignore[arg-type]
  assert added == 5
  assert odl_data["variants"]["torch"]["api"] == "torch.nn.Linear"  # type: ignore[index]
  assert odl_data["variants"]["flax_nnx"]["api"] == "flax.nnx.Linear"  # type: ignore[index]


def test_expand_op_variants_no_torch_api_or_unmatched() -> None:
  """Test expanding variants when torch API is not present or doesn't match."""
  odl_data = {
    "operation": "CustomOp",
    "variants": {
      "torch": {"api": "torch.CustomOp"},  # fn_name in search_names
      "paxml": {"api": "jnp.unknown_jnp_fn"},  # jnp match is None
    },
  }
  snapshots: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"jax": {}, "mlx": {}, "keras": {}}
  added = expander.expand_op_variants(odl_data, snapshots, stem_name="AlternativeStem")  # type: ignore[arg-type]
  assert added == 0

  # Test torch variant with different function name
  odl_data2 = {
    "operation": "DifferentOp",
    "variants": {
      "torch": {"api": "torch.special_func"},
      "jax": {"api": "jax.existing"},
    },
  }
  added = expander.expand_op_variants(odl_data2, snapshots)  # type: ignore[arg-type]
  assert added == 0


def test_expand_parity_across_odl(tmp_path: Path) -> None:
  """Test expanding parity across a directory of ODL files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  f1 = odl_dir / "Add.yaml"
  f1.write_text("operation: Add\nvariants: {}\n")

  f2 = odl_dir / "Invalid.yaml"
  f2.write_text("broken: [yaml\n")

  f3 = odl_dir / "NonDict.yaml"
  f3.write_text("- item1\n- item2\n")

  f4 = odl_dir / "Unchanged.yaml"
  f4.write_text("operation: Unchanged\nvariants: {}\n")

  snapshots: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"jax": {}, "mlx": {}, "keras": {}}

  # Dry run
  mod_files, added = expander.expand_parity_across_odl(odl_dir, snapshots, dry_run=True)  # type: ignore[arg-type]
  assert mod_files == 1
  assert added == 6
  with open(f1, "r", encoding="utf-8") as f:
    assert "rdna" not in yaml.safe_load(f)["variants"]

  # Real run
  mod_files, added = expander.expand_parity_across_odl(odl_dir, snapshots, dry_run=False)  # type: ignore[arg-type]
  assert mod_files == 1
  assert added == 6
  with open(f1, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f)
    assert "rdna" in loaded["variants"]
    assert "nvidia_sass" in loaded["variants"]


def test_main_cli(tmp_path: Path) -> None:
  """Test main CLI entry point for expand_variant_parity.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  f = odl_dir / "Add.yaml"
  f.write_text("operation: Add\nvariants: {}\n")

  with patch("scripts.expand_variant_parity.load_snapshots_multi", return_value={"jax": {}, "mlx": {}, "keras": {}}):
    exit_code = drainer_exit = expander.main(["--odl-dir", str(odl_dir), "--dry-run"])
    assert drainer_exit == 0

    exit_code = expander.main(["--odl-dir", str(odl_dir)])
    assert exit_code == 0


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test __main__ execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
      tmp_path: Pytest temporary directory fixture.
  """
  import runpy
  import sys

  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  repo_str = str(expander.REPO_ROOT)
  orig_sys_path = list(sys.path)
  sys.path = [p for p in sys.path if p != repo_str]

  try:
    monkeypatch.setattr(
      "sys.argv",
      ["expand_variant_parity.py", "--odl-dir", str(odl_dir), "--dry-run"],
    )
    with patch("scripts.expand_variant_parity.load_snapshots_multi", return_value={"jax": {}, "mlx": {}, "keras": {}}):
      with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.expand_variant_parity", run_name="__main__")
      assert excinfo.value.code == 0
  finally:
    sys.path = orig_sys_path
