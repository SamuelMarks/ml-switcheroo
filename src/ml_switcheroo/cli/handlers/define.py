"""
Handler for the 'define' command.

This module orchestrates the "Code Injection" workflow, allowing an LLM or
developer to define a new operation via a YAML specification and automatically
inject the corresponding Python code into the `ml-switcheroo` source tree.

Workflow:
1.  **Parse YAML**: Validates input against the `OperationDef` schema.
2.  **Auto-Inference**: Resolves `api: "infer"` placeholders.
3.  **Inject Hub**: Updates `standards_internal.py` (Abstract Standard).
4.  **Inject Spokes**: Updates framework adapter files (Mappings).
5.  **Scaffold Plugins**: Generates boilerplate Python files for hooks.
6.  **Generate Tests**: Creates physical test file for verification.
"""

import fileinput
import inspect
from pathlib import Path
from typing import Optional, Any

import libcst as cst

try:
  import yaml
except ImportError:
  yaml = None  # type: ignore

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.core.discovery import SimulatedReflection
from ml_switcheroo.tools.injector_spec import StandardsInjector
from ml_switcheroo.tools.injector_fw import FrameworkInjector
from ml_switcheroo.tools.injector_plugin import PluginGenerator
from ml_switcheroo.utils.console import log_success, log_error, log_info, log_warning
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.generated_tests.generator import TestGenerator
import ml_switcheroo.semantics.standards_internal as internal_standards_module
import ml_switcheroo.plugins


def handle_define(yaml_file: Path) -> int:
  """
  Main entry point for the `define` CLI command.

  Args:
      yaml_file: Path to the ODL definition YAML file.

  Returns:
      int: Exit code (0 for success, 1 for failure).
  """
  if yaml is None:
    log_error("PyYAML is not installed. Please run `pip install PyYAML`.")
    return 1

  if not yaml_file.exists() and not yaml_file.name == "-":
    log_error(f"File not found: {yaml_file}")
    return 1

  try:
    with fileinput.FileInput(str(yaml_file), mode="r", encoding="utf-8") as content:
      data = yaml.safe_load("".join(content))

    # Handle both Single Object and List of Objects
    if isinstance(data, list):
      ops = [OperationDef(**d) for d in data]
    else:
      ops = [OperationDef(**data)]

  except Exception as e:
    log_error(f"Failed to valid ODL YAML: {e}")
    return 1

  # Load Semantics Manager once (for test generation templates)
  # Note: We rely on the manager to load templates, but we manually
  # construct the op definition dict for the generator to ensure it sees the NEW op.
  semantics_mgr = SemanticsManager()

  for op_def in ops:
    log_info(f"Defining Operation: {op_def.operation}...")

    # 1. Auto-Inference Resolution
    _resolve_inferred_apis(op_def)

    # 2. Hub Injection (Abstract Standard)
    if not _inject_hub(op_def):
      return 1

    # 3. Spoke Injection (Framework Adapters)
    _inject_spokes(op_def)

    # 4. Plugin Scaffolding (New Hooks)
    _scaffold_plugins(op_def)

    # 5. Test Generation (New Feature)
    _generate_test_file(op_def, semantics_mgr)

    log_success(f"Operation '{op_def.operation}' defined & tested.")

  return 0


def _resolve_inferred_apis(op_def: OperationDef) -> None:
  """
  Iterates through variants and attempts to infer APIs marked with 'infer'.
  """
  for fw_key, variant in op_def.variants.items():
    if variant and variant.api and variant.api.lower() == "infer":
      log_info(f"Inferring API for {op_def.operation} in {fw_key}...")
      reflector = SimulatedReflection(fw_key)
      discovered = reflector.discover(op_def.operation)
      if discovered:
        variant.api = discovered
        log_success(f"  Result: {discovered}")
      else:
        log_warning(f"  Failed: Could not infer API for {op_def.operation} in {fw_key}. Keeping 'infer'.")


def _inject_hub(op_def: OperationDef) -> bool:
  """
  Injects the abstract definition into `standards_internal.py`.
  """
  try:
    spec_file = Path(inspect.getfile(internal_standards_module))
    if not spec_file.exists():
      log_error(f"Could not locate standards source file: {spec_file}")
      return False
    source_code = spec_file.read_text("utf-8")
    tree = cst.parse_module(source_code)
    transformer = StandardsInjector(op_def)
    new_tree = tree.visit(transformer)
    if not transformer.found:
      log_warning("Could not find `INTERNAL_OPS` dictionary in source. Skipping Hub injection.")
      return False
    if new_tree.code != source_code:
      spec_file.write_text(new_tree.code, "utf-8")
      log_success(f"Updated Hub: {spec_file.name}")
    else:
      log_info(f"Hub unchanged: {spec_file.name}")
    return True
  except Exception as e:
    log_error(f"Hub Injection failed: {e}")
    return False


def _inject_spokes(op_def: OperationDef) -> None:
  """
  Iterates variants and injects mappings into framework adapter files.
  """
  for fw_key, variant in op_def.variants.items():
    if not variant:
      continue

    if variant.api and variant.api.lower() == "infer":
      log_warning(f"Skipping '{fw_key}' spoke injection: API is unresolved ('infer').")
      continue
    adapter = get_adapter(fw_key)
    if not adapter:
      continue
    try:
      adapter_cls = type(adapter)
      adapter_file = Path(inspect.getfile(adapter_cls))
      if not adapter_file.exists() or not adapter_file.name.endswith(".py"):
        log_warning(f"Skipping '{fw_key}': Cannot verify source file ({adapter_file}).")
        continue
      source_code = adapter_file.read_text("utf-8")
      tree = cst.parse_module(source_code)
      injector = FrameworkInjector(target_fw=fw_key, op_name=op_def.operation, variant=variant)
      new_tree = tree.visit(injector)
      if not injector.found:
        log_warning(f"Skipping '{fw_key}': Could not locate `definitions` property in {adapter_cls.__name__}.")
        continue
      if new_tree.code != source_code:
        adapter_file.write_text(new_tree.code, "utf-8")
        log_success(f"Updated Spoke ({fw_key}): {adapter_file.name}")
      else:
        log_info(f"Spoke unchanged ({fw_key}): {adapter_file.name}")
    except Exception as e:
      log_warning(f"Failed to update {fw_key}: {e}")


def _scaffold_plugins(op_def: OperationDef) -> None:
  """
  Generates new Python files for required plugins.
  """
  if not op_def.scaffold_plugins:
    return
  try:
    if ml_switcheroo.plugins.__file__:
      plugins_pkg_dir = Path(ml_switcheroo.plugins.__file__).parent
    else:
      # Fallback path logic
      plugins_pkg_dir = Path(__file__).resolve().parents[2] / "plugins"

    generator = PluginGenerator(plugins_pkg_dir)
    for plug_def in op_def.scaffold_plugins:
      try:
        created = generator.generate(plug_def)
        if created:
          log_success(f"Generated Plugin: plugins/{plug_def.name}.py")
        else:
          log_info(f"Plugin already exists (skipped): plugins/{plug_def.name}.py")
      except Exception as e:
        log_error(f"Failed to generate plugin {plug_def.name}: {e}")
  except Exception as e:
    log_error(f"Plugin scaffolding process error: {e}")


def _generate_test_file(op_def: OperationDef, mgr: SemanticsManager) -> None:
  """
  Generates a physical test file for the operation.
  Location: tests/generated/test_odl_{opname}.py
  """
  # 1. Construct Dictionary representation of the Op for the Generator
  # TestGenerator expects { "OpName": { "std_args": ..., "variants": ... } }
  # We must convert the pydantic model to dict
  op_data_dict = op_def.model_dump(exclude_unset=True)

  # Wrap in single-key dictionary
  semantics_input = {op_def.operation: op_data_dict}

  # 2. Determine Output Path
  # Resolve 'tests' directory relative to package
  try:
    # Assuming typical structure: src/ml_switcheroo/cli/handlers/define.py -> root/
    root_dir = Path(__file__).parents[4]
    if not (root_dir / "pyproject.toml").exists():
      # Fallback if structure confusing (e.g. installed pkg vs source)
      root_dir = Path.cwd()
  except Exception:
    root_dir = Path.cwd()

  test_dir = root_dir / "tests" / "generated"
  test_file = test_dir / f"test_odl_{op_def.operation.lower()}.py"

  # 3. Run Generator
  try:
    gen = TestGenerator(semantics_mgr=mgr)
    gen.generate(semantics_input, test_file)
    # Log path usually relative to cwd for readability
    try:
      rel_path = test_file.relative_to(Path.cwd())
      log_success(f"Generated Verification Test: {rel_path}")
    except ValueError:
      log_success(f"Generated Verification Test: {test_file}")
  except Exception as e:
    log_warning(f"Failed to generate tests for {op_def.operation}: {e}")
