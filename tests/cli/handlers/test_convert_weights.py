"""Docstring."""

from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator


def test_weight_script_generator_init() -> None:
  """Docstring."""
  generator: WeightScriptGenerator = WeightScriptGenerator(MagicMock(), MagicMock())
  assert generator is not None


def test_weight_script_generator_unsupported_direction() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.target_framework = "torch"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  assert not generator.generate(Path("a"), Path("b"))


def test_weight_script_generator_no_adapters() -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=None):
    assert not generator.generate(Path("a"), Path("b"))


def test_weight_script_generator_read_fail(tmp_path: Path) -> None:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=MagicMock()):
    assert not generator.generate(tmp_path / "does_not_exist.py", Path("b"))


def test_weight_script_generator_parse_fail(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("invalid python code {{")
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=MagicMock()):
    assert not generator.generate(source, Path("b"))


def test_weight_script_generator_no_layers(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("def foo(): pass")
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=MagicMock()):
    assert not generator.generate(source, Path("b"))


def test_weight_script_generator_success(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out: Path = tmp_path / "script.py"
  semantics: MagicMock = MagicMock()

  from typing import Optional

  # Mock lookup so rules are generated
  def mock_lookup(aid: str) -> Optional[Dict[str, str]]:
    """Docstring."""
    return {"api": "jax.numpy.conv"} if aid == "Conv2d" else None

  semantics.get_definition.return_value = ("Conv2d", {"variants": {"jax": {"api": "jax.numpy.conv"}}})
  semantics.resolve_variant.return_value = {"api": "jax.numpy.conv"}
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    source_adapter: MagicMock = MagicMock()
    source_adapter.get_weight_load_code.return_value = "load"
    source_adapter.get_tensor_to_numpy_expr.return_value = "to_numpy"
    target_adapter: MagicMock = MagicMock()
    target_adapter.get_weight_conversion_imports.return_value = ["import a"]
    target_adapter.get_weight_save_code.return_value = "save"

    def adapter_side_effect(fw: str) -> MagicMock:
      """Docstring."""
      if fw == "torch":
        return source_adapter
      return target_adapter

    mock_get_adapter.side_effect = adapter_side_effect

    assert generator.generate(source, out)
    assert out.exists()


def test_weight_script_generator_write_fail(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out: Path = tmp_path / "script.py"
  pass
  semantics: MagicMock = MagicMock()
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  semantics.get_definition.return_value = ("Conv2d", {"variants": {"jax": {"api": "jax.numpy.conv"}}})
  semantics.resolve_variant.return_value = {"api": "jax.numpy.conv"}
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter"):
    with patch("pathlib.Path.write_text", side_effect=Exception("write error")):
      assert not generator.generate(source, out)


def test_weight_script_generator_rules(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out: Path = tmp_path / "script.py"
  semantics: MagicMock = MagicMock()

  # Mock lookup so rules are generated
  def mock_lookup(aid: str) -> Optional[Tuple[str, dict]]:
    """Docstring."""
    if aid == "Conv2d":
      return (
        "Conv2d",
        {
          "variants": {
            "jax": {"api": "jax.numpy.conv", "layout_map": {"weight": "OIHW->HWIO"}},
            "torch": {"api": "torch.nn.Conv2d", "layout_map": {"weight": "HWIO->OIHW"}},
          }
        },
      )
    return None

  semantics.get_definition.side_effect = mock_lookup
  semantics.resolve_variant.return_value = {"api": "jax.numpy.conv"}
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    source_adapter: MagicMock = MagicMock()
    target_adapter: MagicMock = MagicMock()

    def adapter_side_effect(fw: str) -> MagicMock:
      """Docstring."""
      if fw == "torch":
        return source_adapter
      return target_adapter

    mock_get_adapter.side_effect = adapter_side_effect
    assert generator.generate(source, out)


def test_weight_script_generator_rules_inverse(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out: Path = tmp_path / "script.py"
  semantics: MagicMock = MagicMock()

  # Mock lookup so rules are generated
  def mock_lookup(aid: str) -> Optional[Tuple[str, dict]]:
    """Docstring."""
    if aid == "Conv2d":
      return (
        "Conv2d",
        {
          "variants": {
            "jax": {"api": "jax.numpy.conv", "layout_map": {"weight": "OIHW->HWIO"}},
            "torch": {"api": "torch.nn.Conv2d", "layout_map": {"weight": "HWIO->OIHW"}},
          }
        },
      )
    return None

  semantics.get_definition.side_effect = mock_lookup
  semantics.resolve_variant.return_value = {"api": "jax.numpy.conv"}
  config: MagicMock = MagicMock()
  config.effective_source = "jax"
  config.effective_target = "torch"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    source_adapter: MagicMock = MagicMock()
    target_adapter: MagicMock = MagicMock()

    def adapter_side_effect(fw: str) -> MagicMock:
      """Docstring."""
      if fw == "jax":
        return source_adapter
      return target_adapter

    mock_get_adapter.side_effect = adapter_side_effect
    assert generator.generate(source, out)


def test_weight_script_generator_rules_missing(tmp_path: Path) -> None:
  """Docstring."""
  source: Path = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = MissingOp()")
  out: Path = tmp_path / "script.py"
  semantics: MagicMock = MagicMock()

  # Mock lookup so rules are generated
  def mock_lookup(aid: str) -> Optional[Tuple[str, dict]]:
    """Docstring."""
    if aid == "MissingOp":
      return None
    return None

  semantics.get_definition.side_effect = mock_lookup
  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator: WeightScriptGenerator = WeightScriptGenerator(semantics, config)
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter"):
    assert generator.generate(source, out)


def test_weight_script_generator_rules_no_arrow(tmp_path):
  """Docstring."""
  from unittest.mock import MagicMock, patch
  from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator

  source = tmp_path / "model.py"
  source.write_text("class Model:\n  def __init__(self):\n    self.conv = Conv2d()")
  out = tmp_path / "script.py"
  semantics = MagicMock()

  def mock_lookup(aid: str):
    """Docstring."""
    if aid == "Conv2d":
      return (
        "Conv2d",
        {
          "variants": {
            "jax": {"api": "jax.numpy.conv", "layout_map": {"weight": "FLATTEN"}},
            "torch": {"api": "torch.nn.Conv2d"},
          }
        },
      )
    return None

  semantics.get_definition.side_effect = mock_lookup

  config = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"
  generator = WeightScriptGenerator(semantics, config)

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    source_adapter = MagicMock()
    target_adapter = MagicMock()

    def adapter_side_effect(fw: str):
      """Docstring."""
      if fw == "torch":
        return source_adapter
      return target_adapter

    mock_get_adapter.side_effect = adapter_side_effect
    assert generator.generate(source, out)
