def test_batch_runner_verbose():
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  sm = SemanticsManager()

  # We just want to mock run_all iterating over op names with verbose=True
  # To avoid the execution loop, we can just mock out get_known_apis to return an empty dict
  with __import__("unittest.mock").mock.patch.object(sm, "get_known_apis", return_value={}):
    validator = BatchValidator(sm)
    validator.run_all(verbose=True)


def test_batch_runner_unpack_args_dict():
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator = BatchValidator(SemanticsManager())
  raw_args = [
    {},  # 114-115
    {"name": "a", "type": "int", "min": 0, "max": 10},  # 118-129
    {"name": "b"},  # Missing type
  ]
  p, h, c = validator._unpack_args(raw_args)
  assert "a" in p
  assert "b" in p
  assert h["a"] == "int"
  assert "min" in c["a"]


def test_batch_runner_scan_manual_tests_not_exist(tmp_path):
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator = BatchValidator(SemanticsManager())
  # Dir doesn't exist
  assert validator._scan_manual_tests(tmp_path / "fake_dir") == set()


def test_batch_runner_scan_manual_tests_parse_error(tmp_path):
  from ml_switcheroo.testing.batch_runner import BatchValidator
  from ml_switcheroo.semantics.manager import SemanticsManager

  validator = BatchValidator(SemanticsManager())
  d = tmp_path / "test_dir"
  d.mkdir()
  f = d / "test_bad.py"
  f.write_text("def test_foo():\n    this is a syntax error\n")

  assert validator._scan_manual_tests(d) == set()
