def test_bisector_extract_params():
  from ml_switcheroo.testing.bisector import SemanticsBisector
  import logging

  bisector = SemanticsBisector(None)

  op_def = {"std_args": [{"name": "a", "type": "int", "min": 0}, ["b", "float"], ["c"], "d"]}

  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.testing.bisector.EquivalenceRunner.verify", return_value=(True, "OK")
  ):
    import ml_switcheroo.testing.bisector

    old_runner = ml_switcheroo.testing.bisector.EquivalenceRunner

    class MockRunner:
      def verify(self, *args, **kwargs):
        return True, "OK"

    bisector.runner = MockRunner()
    bisector.propose_fix("foo", op_def)


def test_bisector_fix_found():
  from ml_switcheroo.testing.bisector import SemanticsBisector
  import logging

  bisector = SemanticsBisector(None)

  op_def = {"std_args": ["a"]}

  class MockRunner:
    def verify(self, *args, **kwargs):
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
  from ml_switcheroo.testing.bisector import SemanticsBisector
  import logging

  bisector = SemanticsBisector(None)

  op_def = {"std_args": ["a"]}

  class MockRunner:
    def verify(self, *args, **kwargs):
      raise Exception("fail")

  bisector.runner = MockRunner()
  res = bisector.propose_fix("foo", op_def)
  assert res is None
