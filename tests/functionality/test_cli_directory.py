from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  def get_definition(self, api_name):
    if api_name == "torch.abs":
      return "abs", {"std_args": ["x"], "variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}}
    elif api_name == "torch.sum":
      return "sum", {"std_args": ["x"], "variants": {"torch": {"api": "torch.sum"}, "jax": {"api": "jax.numpy.sum"}}}
    return None

  def resolve_variant(self, abstract_id, target_fw):
    if target_fw == "jax":
      if abstract_id == "abs":
        return {"api": "jax.numpy.abs"}
      if abstract_id == "sum":
        return {"api": "jax.numpy.sum"}
    return None

  def is_verified(self, _id):
    return True


def test_recursive_directory_mirroring(tmp_path, capsys, monkeypatch):
  monkeypatch.setattr("ml_switcheroo.cli.handlers.convert.SemanticsManager", MockSemantics)

  in_root = tmp_path / "src"
  in_root.mkdir()
  (in_root / "main.py").write_text("import torch\nx = torch.abs(y)")

  out_root = tmp_path / "dst"

  # Run
  try:
    main(["convert", str(in_root), "--out", str(out_root)])
  except SystemExit:
    pass

  assert (out_root / "main.py").exists()
  content = (out_root / "main.py").read_text()
  assert "jax.numpy.abs" in content


def test_directory_fails_without_out_arg(tmp_path, capsys):
  in_root = tmp_path / "bad"
  in_root.mkdir()
  try:
    main(["convert", str(in_root)])
  except SystemExit as e:
    assert e.code == 1
  assert "requires --out" in capsys.readouterr().out


def test_empty_directory_handling(tmp_path, capsys):
  in_root = tmp_path / "empty"
  in_root.mkdir()
  try:
    main(["convert", str(in_root), "--out", str(tmp_path / "out")])
  except SystemExit:
    pass
  assert "No .py files found" in capsys.readouterr().out
