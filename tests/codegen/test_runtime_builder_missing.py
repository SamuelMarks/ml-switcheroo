from ml_switcheroo.generated_tests.runtime_builder import get_required_packages, ensure_runtime_module


def test_get_required_packages_syntax_error():
  assert get_required_packages("import from invalid syntax") == []


def test_ensure_runtime_module_no_req_pkgs(tmp_path):
  class MockSemantics:
    def get_test_template(self, fw):
      return {"import": "import"}  # Syntax error

    def get_framework_config(self, fw):
      return {}

  ensure_runtime_module(tmp_path, ["dummy"], MockSemantics())
  runtime_py = tmp_path / "runtime.py"
  assert runtime_py.exists()
  content = runtime_py.read_text()
  assert "DUMMY_AVAILABLE = True" in content
