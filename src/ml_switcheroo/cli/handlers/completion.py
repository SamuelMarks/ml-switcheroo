"""Shell Completion Script Generator for ml-switcheroo CLI.

This module provides generators for Bash, Zsh, and Fish shell completion
scripts supporting all CLI subcommands, flags, and framework choices.
"""

import sys
from typing import List

_COMMANDS: List[str] = [
  "convert",
  "convert-weights",
  "define",
  "import-onnx",
  "gen-weight-script",
  "matrix",
  "schema",
  "suggest",
  "scaffold",
  "harvest",
  "verified-pipeline",
  "ci",
  "gen-docs",
  "gen-tests",
  "completion",
]

_FRAMEWORKS: List[str] = [
  "torch",
  "jax",
  "flax_nnx",
  "mlx",
  "keras",
  "tensorflow",
  "numpy",
  "nvidia_sass",
  "rdna",
  "mlir",
  "stablehlo",
  "ir",
  "wasm",
  "cpp",
]


def generate_bash_completion() -> str:
  """Generate Bash completion script for ml-switcheroo.

  Returns:
      A string containing Bash completion shell script.
  """
  cmds = " ".join(_COMMANDS)
  fws = " ".join(_FRAMEWORKS)
  template = [
    "# bash completion for ml_switcheroo",
    "_ml_switcheroo_completion() {",
    "    local cur prev opts commands frameworks",
    "    COMPREPLY=()",
    '    cur="${COMP_WORDS[COMP_CWORD]}"',
    '    prev="${COMP_WORDS[COMP_CWORD-1]}"',
    f'    commands="{cmds}"',
    f'    frameworks="{fws}"',
    "",
    '    case "$prev" in',
    "        --source|--target|-s|-t)",
    '            COMPREPLY=( $(compgen -W "$frameworks" -- "$cur") )',
    "            return 0",
    "            ;;",
    "        --intermediate)",
    '            COMPREPLY=( $(compgen -W "ir ml_switcheroo_ir mlir tikz" -- "$cur") )',
    "            return 0",
    "            ;;",
    "        completion)",
    '            COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") )',
    "            return 0",
    "            ;;",
    "    esac",
    "",
    '    if [[ "$COMP_CWORD" -eq 1 ]] ; then',
    '        COMPREPLY=( $(compgen -W "$commands --help --version" -- "$cur") )',
    "        return 0",
    "    fi",
    "}",
    "",
    "complete -F _ml_switcheroo_completion ml_switcheroo",
    'complete -F _ml_switcheroo_completion "🔄🦘"',
    "",
  ]
  return chr(10).join(template)


def generate_zsh_completion() -> str:
  """Generate Zsh completion script for ml-switcheroo.

  Returns:
      A string containing Zsh completion shell script.
  """
  cmds = " ".join(f"'{c}:{c} command'" for c in _COMMANDS)
  fws = " ".join(f"'{f}:{f} framework'" for f in _FRAMEWORKS)
  template = [
    "#compdef ml_switcheroo 🔄🦘",
    "",
    "_ml_switcheroo() {",
    "    local -a commands frameworks",
    f"    commands=({cmds})",
    f"    frameworks=({fws})",
    "",
    "    _arguments -C ",
    "        '1: :->command' ",
    "        '*: :->args'",
    "",
    "    case $state in",
    "        command)",
    "            _describe -t commands 'ml_switcheroo command' commands",
    "            ;;",
    "        args)",
    "            case $words[2] in",
    "                convert)",
    "                    _arguments ",
    "                        '--source[Source framework]:framework:($frameworks)' ",
    "                        '--target[Target framework]:framework:($frameworks)' ",
    "                        '--intermediate[Intermediate dialect]:dialect:(ir ml_switcheroo_ir mlir tikz)'",
    "                    ;;",
    "                completion)",
    "                    _values 'shell' bash zsh fish",
    "                    ;;",
    "            esac",
    "            ;;",
    "    esac",
    "}",
    "",
    '_ml_switcheroo "$@"',
    "",
  ]
  return chr(10).join(template)


def generate_fish_completion() -> str:
  """Generate Fish completion script for ml-switcheroo.

  Returns:
      A string containing Fish completion shell script.
  """
  lines = [
    "# Fish completion for ml_switcheroo",
    "complete -c ml_switcheroo -f",
    "complete -c '🔄🦘' -f",
  ]
  for cmd in _COMMANDS:
    lines.append(f"complete -c ml_switcheroo -n '__fish_use_subcommand' -a '{cmd}' -d '{cmd} subcommand'")
    lines.append(f"complete -c '🔄🦘' -n '__fish_use_subcommand' -a '{cmd}' -d '{cmd} subcommand'")

  for fw in _FRAMEWORKS:
    lines.append(f"complete -c ml_switcheroo -l source -a '{fw}' -d '{fw} framework'")
    lines.append(f"complete -c ml_switcheroo -l target -a '{fw}' -d '{fw} framework'")

  lines.append("complete -c ml_switcheroo -n '__fish_seen_subcommand_from completion' -a 'bash zsh fish'")
  lines.append("")
  return chr(10).join(lines)


def handle_completion(shell: str) -> int:
  """Handle the 'completion' CLI command.

  Args:
      shell: Target shell ('bash', 'zsh', 'fish').

  Returns:
      int: 0 on success, 1 on invalid shell choice.
  """
  clean_shell = shell.lower().strip()
  if clean_shell == "bash":
    sys.stdout.write(generate_bash_completion())
    return 0
  elif clean_shell == "zsh":
    sys.stdout.write(generate_zsh_completion())
    return 0
  elif clean_shell == "fish":
    sys.stdout.write(generate_fish_completion())
    return 0
  else:
    sys.stderr.write(f"Unsupported shell '{shell}'. Supported shells: bash, zsh, fish" + chr(10))
    return 1
