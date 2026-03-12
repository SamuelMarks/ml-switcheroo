def test_wizard_missing_lines():
  from ml_switcheroo.cli.wizard import MappingWizard
  from ml_switcheroo.semantics.manager import SemanticsManager
  from unittest.mock import patch, MagicMock

  with patch("ml_switcheroo.cli.wizard.RuntimeConfig", side_effect=Exception("error")):
    wiz = MappingWizard(SemanticsManager())
    assert wiz.default_target == "jax"  # 55-56

  wiz = MappingWizard(SemanticsManager())
  wiz.console = MagicMock()

  # 128-129 KeyboardInterrupt in start
  with patch.object(wiz, "_render_card", side_effect=KeyboardInterrupt):
    with patch.object(wiz, "_find_unmapped_apis", return_value={"test": {}}):
      with patch("ml_switcheroo.cli.wizard.ApiInspector") as mock_insp:
        wiz.start("test")
        wiz.console.print.assert_called_with("\n[yellow]Wizard interrupted by user.[/yellow]")

  # 167 Empty doc
  wiz._render_card("test", {"doc_summary": ""}, 1, 1)


from unittest.mock import patch, MagicMock, mock_open


def test_wizard_normalize_args():
  from ml_switcheroo.cli.wizard import MappingWizard
  from ml_switcheroo.semantics.manager import SemanticsManager

  wiz = MappingWizard(SemanticsManager())
  wiz.console = MagicMock()

  # 211
  assert wiz._prompt_arg_normalization([], "test") == ([], {})

  # 217
  with patch("ml_switcheroo.cli.wizard.Prompt.ask", return_value="test"):
    std, mapp = wiz._prompt_arg_normalization(["self"], "test")
    assert std == []


def test_wizard_write_json_exceptions():
  from ml_switcheroo.cli.wizard import MappingWizard
  from ml_switcheroo.semantics.manager import SemanticsManager
  from pathlib import Path

  wiz = MappingWizard(SemanticsManager())

  # 360-361
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("builtins.open", mock_open()) as m_open,
    patch("json.load", side_effect=Exception("err")),
  ):
    wiz._write_to_file(Path("fake.json"), "key", {})

  # 388-389
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("builtins.open", mock_open()) as m_open,
    patch("json.load", side_effect=Exception("err")),
  ):
    wiz._write_to_snapshot(Path("snap"), "fw", "key", {})


def test_wizard_save_pop_args():
  from ml_switcheroo.cli.wizard import MappingWizard
  from ml_switcheroo.semantics.manager import SemanticsManager

  wiz = MappingWizard(SemanticsManager())

  # 338, 364
  with patch.object(wiz, "_write_to_snapshot"):
    with patch.object(wiz, "_write_to_file"):
      with patch("ml_switcheroo.cli.wizard.resolve_semantics_dir"):
        with patch("ml_switcheroo.cli.wizard.resolve_snapshots_dir"):
          # 338: args is None
          wiz._save_complex_entry(
            "f",
            "src",
            "test",
            [],
            "src_fw",
            {},
            {"framework": "tgt", "data": {"api": "b", "args": None, "requires_plugin": None}},
          )


def test_wizard_write_json_update():
  from ml_switcheroo.cli.wizard import MappingWizard
  from ml_switcheroo.semantics.manager import SemanticsManager
  from pathlib import Path

  wiz = MappingWizard(SemanticsManager())

  # 364
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("builtins.open", mock_open()) as m_open,
    patch("json.load", return_value={"key": {"a": 1}}),
    patch("json.dump"),
  ):
    wiz._write_to_file(Path("fake.json"), "key", {"b": 2})

  # 392
  with (
    patch("pathlib.Path.exists", return_value=True),
    patch("builtins.open", mock_open()) as m_open,
    patch("json.load", return_value={}),
    patch("json.dump"),
  ):
    wiz._write_to_snapshot(Path("snap"), "fw", "key", {})
