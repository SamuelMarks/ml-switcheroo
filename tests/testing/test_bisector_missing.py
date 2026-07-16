"""Auto-generated doc."""


def test_bisector_extract_params():
  """Auto-generated doc."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)

  op_def = {"std_args": [{"name": "a", "type": "int", "min": 0}, ["b", "float"], ["c"], "d"], "test_rtol": 1e-10}

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.bisector.EquivalenceRunner.verify", return_value=(True, "OK")
  ):

    class MockRunner:
      """Auto-generated doc."""

      def verify(self, *args, **kwargs):
        """Auto-generated doc."""
        # Verify hints are extracted correctly to cover those lines
        assert kwargs.get("hints") == {"a": "int", "b": "float"}
        return True, "OK"

    bisector.runner = MockRunner()
    res = bisector.propose_fix("foo", op_def)
    assert res is not None


def test_bisector_runner_exception():
  """Auto-generated doc."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  class MockRunnerEx:
    """Auto-generated doc."""

    def verify(self, *args, **kwargs):
      """Auto-generated doc."""
      raise ValueError("Runner died")

  bisector.runner = MockRunnerEx()
  res = bisector.propose_fix("foo", op_def)
  assert res is None


def test_bisector_no_fix_needed():
  """Auto-generated doc."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)
  op_def = {"std_args": ["a"]}

  # We want it to pass on the very first try (1e-3, 1e-4) which matches defaults
  class MockRunnerOk:
    """Auto-generated doc."""

    def verify(self, *args, **kwargs):
      """Auto-generated doc."""
      return True, "OK"

  bisector.runner = MockRunnerOk()
  res = bisector.propose_fix("foo", op_def)
  # Because it passed on default config, no fix needed
  assert res is None


def test_bisector_fix_found():
  """Auto-generated doc."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)

  op_def = {"std_args": ["a"]}

  class MockRunner:
    """Auto-generated doc."""

    def verify(self, *args, **kwargs):
      """Auto-generated doc."""
      return True, "OK"

  bisector.runner = MockRunner()

  # original is 1e-3, we need it to be 1e-3, 1e-4 and match.
  # The first item in relaxations is (1e-3, 1e-4) which matches default.
  # We want it to be DIFFERENT from current config.
  op_def["test_rtol"] = 1e-9  # Different
  res = bisector.propose_fix("foo", op_def)
  assert res is not None
  assert res["test_rtol"] == 1e-3


def test_bisector_exception():
  """Auto-generated doc."""
  from ml_switcheroo.testing.bisector import SemanticsBisector

  bisector = SemanticsBisector(None)

  op_def = {"std_args": ["a"]}

  class MockRunner:
    """Auto-generated doc."""

    def verify(self, *args, **kwargs):
      """Auto-generated doc."""
      raise Exception("fail")

  bisector.runner = MockRunner()
  res = bisector.propose_fix("foo", op_def)
  assert res is None
