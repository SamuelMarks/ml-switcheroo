"""Test suite for the Verified Pipeline module."""

from ml_switcheroo.ingestion import verified_pipeline


def test_verified_pipeline_dummy():
  """Verifies the behavior of verified pipeline dummy."""
  assert hasattr(verified_pipeline, "run_verified_pipeline")
