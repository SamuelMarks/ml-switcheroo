"""
Tests for the Consensus Engine.

Verifies:
1. Normalization Logic (Stripping suffixes, handling cases).
2. Clustering Logic (Grouping disparate frameworks correctly).
3. Signature Alignment (Identifying standard arguments).
"""

import pytest
from ml_switcheroo.discovery.consensus import ConsensusEngine, CandidateStandard
from ml_switcheroo.core.ghost import GhostRef, GhostParam

# --- Fixtures ---


@pytest.fixture
def engine():
  return ConsensusEngine()


# --- Normalization Tests ---


def test_normalize_snake_vs_camel(engine):
  """Verify snake_case and CamelCase resolve to same key."""
  assert engine.normalize_name("CrossEntropy") == "crossentropy"
  assert engine.normalize_name("cross_entropy") == "crossentropy"


def test_normalize_strip_suffixes(engine):
  """Verify noise words are stripped."""
  assert engine.normalize_name("HuberLoss") == "huber"
  assert engine.normalize_name("huber_loss") == "huber"
  assert engine.normalize_name("L1Loss") == "l1"


def test_arg_normalization(engine):
  """Verify argument canonicalization aliases."""
  assert engine.normalize_arg("learning_rate") == "lr"
  assert engine.normalize_arg("lr") == "lr"
  assert engine.normalize_arg("epsilon") == "eps"
  assert engine.normalize_arg("unknown_arg") == "unknown_arg"


# --- Clustering Tests ---


def test_cluster_huber_consensus(engine):
  """Verify clustering groups variants correctly."""
  inputs = {
    "torch": [GhostRef(name="HuberLoss", api_path="torch.nn.HuberLoss", kind="class")],
    "keras": [GhostRef(name="Huber", api_path="keras.losses.Huber", kind="class")],
    "jax": [GhostRef(name="huber_loss", api_path="optax.huber_loss", kind="function")],
  }
  results = engine.cluster(inputs)
  assert len(results) == 1
  assert results[0].score == 3.0


def test_filtering_orphans(engine):
  """Verify filtering based on support."""
  dummy = GhostRef(name="d", api_path="d", kind="c")
  candidates = [
    CandidateStandard(name="Common", variants={"a": dummy, "b": dummy}, score=2.0),
    CandidateStandard(name="Unique", variants={"a": dummy}, score=1.0),
  ]
  filtered = engine.filter_common(candidates, min_support=2)
  assert len(filtered) == 1
  assert filtered[0].name == "Common"


# --- Signature Alignment Tests ---


def test_signature_alignment_consensus(engine):
  """
  Scenario:
  - Torch: Adam(lr=0.001, betas=(0.9, 0.99))
  - JAX: adam(learning_rate=0.001, b1=0.9, b2=0.999)
  - Keras: Adam(learning_rate=0.001)

  Expectation:
  - Consensus Args: 'lr' (3/3), 'betas' (No consensus, structure mismatch or low count?
    Wait, Torch: betas / Jax: b1, b2. No direct mapping yet without complex split logic.
    Let's test simpler args first).

  Let's try 'epsilon' / 'eps'.
  """

  # Create GhostRefs with Params
  p_lr = GhostParam(name="lr", kind="kw")
  p_eps = GhostParam(name="eps", kind="kw")

  p_learn = GhostParam(name="learning_rate", kind="kw")
  p_epsilon = GhostParam(name="epsilon", kind="kw")

  # Variant A: Standard shorter names
  ref_a = GhostRef(name="Adam", api_path="A", kind="c", params=[p_lr, p_eps])

  # Variant B: Verbose names
  ref_b = GhostRef(name="adam", api_path="B", kind="f", params=[p_learn, p_epsilon])

  # Variant C: Mixed
  ref_c = GhostRef(name="Adam", api_path="C", kind="c", params=[p_learn])
  # C missing epsilon

  cand = CandidateStandard(name="Adam", variants={"a": ref_a, "b": ref_b, "c": ref_c})

  engine.align_signatures([cand], consensus_threshold=0.5)

  # 'lr': Present in A(lr), B(learning_rate->lr), C(learning_rate->lr). 3/3 Support.
  assert "lr" in cand.std_args

  # 'eps': Present in A(eps), B(epsilon->eps). Missing in C. 2/3 Support. (>0.66)
  assert "eps" in cand.std_args

  # Mappings
  # A uses 'lr', B uses 'learning_rate', C uses 'learning_rate'
  assert cand.arg_mappings["a"]["lr"] == "lr"
  assert cand.arg_mappings["b"]["lr"] == "learning_rate"

  # A uses 'eps', B uses 'epsilon'. C does not map.
  assert cand.arg_mappings["a"]["eps"] == "eps"
  assert cand.arg_mappings["b"]["eps"] == "epsilon"
  assert "eps" not in cand.arg_mappings["c"]


def test_signature_alignment_threshold(engine):
  """
  Scenario: An argument exists only in 1 out of 3 frameworks.
  Expectation: It should NOT be promoted to standard arg (threshold 0.5).
  """
  p_amsgrad = GhostParam(name="amsgrad", kind="kw")

  ref_a = GhostRef(name="Adam", api_path="A", kind="c", params=[p_amsgrad])
  ref_b = GhostRef(name="Adam", api_path="B", kind="c", params=[])
  ref_c = GhostRef(name="Adam", api_path="C", kind="c", params=[])

  cand = CandidateStandard(name="Adam", variants={"a": ref_a, "b": ref_b, "c": ref_c})

  engine.align_signatures([cand], consensus_threshold=0.5)

  # Support 1/3 (0.33) < 0.5
  assert "amsgrad" not in cand.std_args
