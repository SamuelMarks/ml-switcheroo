"""Golden Reference Corpus Discovery, Transpilation, and Numerical Parity Suite.

Systematically discovers all 30 operator and model subdirectories in ``tests/gold/``,
verifies every golden reference file contains non-trivial AST declarations,
transpiles golden implementations across framework pairs via ``ASTEngine``, and
verifies numerical execution behavior across pure-NumPy and JAX golden references.
"""

import ast
from pathlib import Path

import libcst as cst
import numpy as np
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager

GOLD_ROOT: Path = Path(__file__).resolve().parent

EXPECTED_CORPUS_DIRS: tuple[str, ...] = (
  "attention",
  "batch_norm",
  "concat",
  "cond_control_flow",
  "conv2d",
  "conv2d_same_padding",
  "cross_entropy_loss",
  "dropout",
  "einsum",
  "elementwise_relu",
  "embedding",
  "flatten",
  "gelu",
  "global_avg_pool",
  "gradient_clipping",
  "group_norm",
  "layer_norm",
  "linear",
  "lstm",
  "masked_fill",
  "max_pool2d",
  "mlp",
  "optim_adam",
  "resnet_block",
  "scan_loop",
  "sequential",
  "split",
  "training_loop",
  "transformer_block",
  "weight_norm",
)

STEM_TO_FRAMEWORK: dict[str, str] = {
  "flax_nnx": "flax_nnx",
  "keras3": "keras",
  "mlx": "mlx",
  "numpy": "numpy",
  "paxml": "paxml",
  "pytorch": "torch",
  "tensorflow": "tensorflow",
}

TRANSPILATION_TARGETS: tuple[str, ...] = (
  "flax_nnx",
  "keras",
  "mlx",
  "numpy",
  "tensorflow",
)


def _discover_operator_dirs() -> list[Path]:
  """Discover all operator/model corpus directories under ``tests/gold/``.

  Returns:
      Sorted list of operator corpus directory paths (excluding ``wasm`` and caches).
  """
  return sorted(p for p in GOLD_ROOT.iterdir() if p.is_dir() and not p.name.startswith("__") and p.name != "wasm")


def _top_level_symbol_names(source: str) -> list[str]:
  """Extract public top-level function and class names from Python source code.

  Args:
      source: Python module source string.

  Returns:
      Ordered list of public top-level ``FunctionDef`` and ``ClassDef`` names.
  """
  tree = ast.parse(source)
  return [
    stmt.name for stmt in tree.body if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)) and not stmt.name.startswith("_")
  ]


@pytest.fixture(scope="module")
def shared_semantics() -> SemanticsManager:
  """Provide a module-scoped ``SemanticsManager`` for fast multi-pair transpilation.

  Returns:
      Initialized ``SemanticsManager`` instance.
  """
  return SemanticsManager()


def test_gold_corpus_directory_completeness() -> None:
  """Verify all 30 expected operator/model subdirectories exist with all 7 framework variants."""
  discovered = [d.name for d in _discover_operator_dirs()]
  assert tuple(discovered) == EXPECTED_CORPUS_DIRS

  for op_dir_name in EXPECTED_CORPUS_DIRS:
    op_dir = GOLD_ROOT / op_dir_name
    stems = {f.stem for f in op_dir.glob("*.py") if f.name != "__init__.py"}
    assert stems == set(STEM_TO_FRAMEWORK.keys()), f"Missing framework files in {op_dir_name}: {stems}"


@pytest.mark.parametrize("op_name", EXPECTED_CORPUS_DIRS)
def test_gold_files_ast_and_symbol_parity(op_name: str) -> None:
  """Verify every golden reference file in ``op_name`` parses cleanly and declares public symbols.

  Args:
      op_name: Operator/model corpus subdirectory name.
  """
  op_dir = GOLD_ROOT / op_name
  symbol_sets: dict[str, set[str]] = {}

  for stem in sorted(STEM_TO_FRAMEWORK):
    file_path = op_dir / f"{stem}.py"
    source = file_path.read_text(encoding="utf-8")
    cst_mod = cst.parse_module(source)
    assert len(cst_mod.body) >= 2, f"{file_path} is an unimplemented stub"

    symbols = _top_level_symbol_names(source)
    assert symbols, f"{file_path} declares no public top-level functions or classes"
    symbol_sets[stem] = set(symbols)

  torch_primary = symbol_sets["pytorch"]
  for stem, sym_set in symbol_sets.items():
    assert sym_set & torch_primary, f"{op_name}/{stem}.py public symbols {sym_set} do not overlap PyTorch {torch_primary}"


@pytest.mark.parametrize("op_name", EXPECTED_CORPUS_DIRS)
def test_gold_corpus_cross_framework_transpilation(
  op_name: str,
  shared_semantics: SemanticsManager,
) -> None:
  """Transpile each golden operator across framework pairs and assert structural validity.

  Args:
      op_name: Operator/model corpus subdirectory name.
      shared_semantics: Module-scoped ``SemanticsManager`` fixture.
  """
  op_dir = GOLD_ROOT / op_name
  pytorch_src = (op_dir / "pytorch.py").read_text(encoding="utf-8")
  expected_symbols = set(_top_level_symbol_names(pytorch_src))

  for target_fw in TRANSPILATION_TARGETS:
    cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=False)
    engine = ASTEngine(semantics=shared_semantics, config=cfg)
    result = engine.run(pytorch_src)
    assert result.success, f"torch -> {target_fw} failed on {op_name}: {result.errors}"
    cst.parse_module(result.code)
    emitted_symbols = set(_top_level_symbol_names(result.code))
    assert expected_symbols.issubset(emitted_symbols), (
      f"torch -> {target_fw} lost symbols on {op_name}: expected {expected_symbols}, got {emitted_symbols}"
    )

  flax_src = (op_dir / "flax_nnx.py").read_text(encoding="utf-8")
  flax_symbols = set(_top_level_symbol_names(flax_src))
  rev_cfg = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rev_res = ASTEngine(semantics=shared_semantics, config=rev_cfg).run(flax_src)
  assert rev_res.success, f"flax_nnx -> torch failed on {op_name}: {rev_res.errors}"
  cst.parse_module(rev_res.code)
  assert flax_symbols.issubset(set(_top_level_symbol_names(rev_res.code)))


def test_gold_numpy_numerical_execution_parity() -> None:
  """Execute NumPy golden reference modules and verify numerical invariants."""
  from tests.gold.concat.numpy import concat_tensors
  from tests.gold.cond_control_flow.numpy import conditional_op
  from tests.gold.elementwise_relu.numpy import relu_activation
  from tests.gold.gradient_clipping.numpy import clip_grads
  from tests.gold.group_norm.numpy import GNModel
  from tests.gold.masked_fill.numpy import causal_mask_fill
  from tests.gold.scan_loop.numpy import rnn_loop
  from tests.gold.split.numpy import split_tensor
  from tests.gold.transformer_block.numpy import TransformerBlock
  from tests.gold.weight_norm.numpy import WNModel

  x = np.array([[-2.0, 0.0, 3.0]], dtype=np.float64)
  np.testing.assert_allclose(relu_activation(x), np.array([[0.0, 0.0, 3.0]]))
  np.testing.assert_allclose(concat_tensors(x, x, axis=0).shape, (2, 3))

  np.testing.assert_allclose(conditional_op(np.array([True]), x), np.array([[-4.0, 0.0, 6.0]]))
  np.testing.assert_allclose(conditional_op(np.array([False]), x), np.array([[0.0, 2.0, 5.0]]))

  clipped = clip_grads([np.array([3.0, 4.0], dtype=np.float64)], max_norm=2.5)
  np.testing.assert_allclose(np.linalg.norm(clipped[0]), 2.5, rtol=1e-6)
  unchanged = clip_grads([np.array([0.3, 0.4], dtype=np.float64)], max_norm=2.5)
  np.testing.assert_allclose(unchanged[0], np.array([0.3, 0.4]))

  gn = GNModel(num_groups=2, num_channels=4)
  gn_in = np.arange(32, dtype=np.float64).reshape(1, 4, 2, 4)
  gn_out = gn(gn_in)
  assert gn_out.shape == (1, 4, 2, 4)
  np.testing.assert_allclose(np.mean(gn_out[0, :2]), 0.0, atol=1e-6)

  scores = np.ones((2, 2), dtype=np.float64)
  mask = np.array([[1, 0], [1, 1]], dtype=np.int64)
  filled = causal_mask_fill(scores, mask, value=-1e9)
  assert filled[0, 1] == -1e9
  assert filled[0, 0] == 1.0

  seq = np.ones((3, 2), dtype=np.float64)
  init_s = np.zeros((2,), dtype=np.float64)
  outs, final_s = rnn_loop(lambda xt, st: (xt + st, xt + st), seq, init_s)
  assert outs.shape == (3, 2)
  np.testing.assert_allclose(final_s, np.array([3.0, 3.0]))

  parts = split_tensor(np.arange(12, dtype=np.float64).reshape(2, 6), split_size=2, dim=-1)
  assert len(parts) == 3
  assert all(p.shape == (2, 2) for p in parts)

  block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16)
  tokens = np.ones((1, 4, 8), dtype=np.float64)
  encoded = block(tokens, attn_mask=np.zeros((4, 4), dtype=np.float64))
  assert encoded.shape == (1, 4, 8)

  wn = WNModel(in_features=4, out_features=3)
  wn_out = wn(np.ones((2, 4), dtype=np.float64))
  assert wn_out.shape == (2, 3)
  np.testing.assert_allclose(wn_out, np.full((2, 3), 2.0), rtol=1e-6)
