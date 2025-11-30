from ml_switcheroo.cli.__main__ import main


def test_cli_convert_file(tmp_path, capsys):
  """Test full file conversion flow."""
  infile = tmp_path / "model.py"
  infile.write_text("y = torch.unknown_func(x)")  # Use unknown to ensure pass-through
  outfile = tmp_path / "converted.py"

  args = ["convert", str(infile), "--out", str(outfile), "--source", "torch", "--target", "jax"]

  try:
    main(args)
  except SystemExit:
    pass

  assert outfile.exists()
  content = outfile.read_text()

  # Expect passthrough because 'unknown_func' is not in our clean JSONs
  assert "torch.unknown_func(x)" in content
