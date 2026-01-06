#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
#
# Features:
# - Auto-installs missing ML libraries (torch, jax, keras, etc.)
# - Fallback restoration of snapshots from git if libraries are missing.
# - Special handling for PaxML via Py3.10 venvs.
# - Dynamic detection of required libraries via Python introspection.
# - ANSI color output via printf.
# - Uses `uv` for installation if available.
# ==============================================================================

set -e

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# Directory Resolution (POSIX compliant replacement for realpath)
# ------------------------------------------------------------------------------
get_abs_script_path() {
    # Run in subshell to keep current shell context
    (
        CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P
    )
}

SCRIPT_LOCATION=$(get_abs_script_path)
# Go up two levels from the script location to find ROOT_DIR:
# Equivalent to: dirname(dirname(script_path))
ROOT_DIR=$(dirname -- "$(dirname -- "$SCRIPT_LOCATION")")

VENV_DIR="${VENV_DIR:-$ROOT_DIR}"
SNAPSHOTS_DIR="$ROOT_DIR/src/ml_switcheroo/snapshots"
SEMANTICS_DIR="$ROOT_DIR/src/ml_switcheroo/semantics"
TEMP_DIR="_bootstrap_temp"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

pip_install() {
  if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install "$@"
  else
    uv pip install "$@"
  fi
}

pip_() {
  if python3 -m pip --version >/dev/null 2>&1; then
    # shellcheck disable=SC2068
    python3 -m pip $@
  else
    # shellcheck disable=SC2068
    uv pip $@
  fi
}

printf '%b🔥 Initiating Knowledge Base Bootstrap (Dynamic)%b\n' "${BOLD}" "${NC}"
printf '%s\n' "----------------------------------------"

# ------------------------------------------------------------------------------
# 0. SETUP & CLEANUP
# ------------------------------------------------------------------------------
printf '%b[0/6] Preparing Workspace...%b\n' "${BLUE}" "${NC}"

# Clean existing JSON artifacts to force regeneration.
rm -f "$SEMANTICS_DIR"/k_*.json
rm -f "$SNAPSHOTS_DIR"/*_map.json
rm -f "$SNAPSHOTS_DIR"/*_v*.json

# Prepare temp dir
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Ensure ml-switcheroo is installed in editable mode.
printf '   📦 Ensuring local package is installed...\n'
pip_install -e . > /dev/null 2>&1 || true

# ------------------------------------------------------------------------------
# 1. DEPENDENCY CHECK & INSTALL
# ------------------------------------------------------------------------------
printf '\n%b[1/6] Checking Environment & Dependencies...%b\n' "${BLUE}" "${NC}"

# Dynamically determine supported frameworks via Python.
# Returns a space-separated string of package names.
REQUIRED_LIBS=$(python3 -c "
try:
    from ml_switcheroo.frameworks import available_frameworks
    print(' '.join(sorted(set(
      {'flax_nnx': 'flax'}.get(fw, fw)
      for fw in available_frameworks()
      if fw not in {'html', 'latex_dsl', 'tikz', 'mlir', 'stablehlo'}
    ))))
except ImportError:
    print('torch jax flax tensorflow keras mlx numpy')
")

printf '   🔍 Audited requirements: %b%s%b\n' "${YELLOW}" "$REQUIRED_LIBS" "${NC}"

MISSING_LIBS=""

# POSIX Loop Strategy:
# 1. printf the list.
# 2. tr converts spaces to newlines.
# 3. read processes line-by-line.
# 4. We capture stdout because the loop runs in a subshell.
DETECTED_MISSING=$(printf '%s' "$REQUIRED_LIBS" | tr ' ' '\n' | while read -r lib; do
    # Skip empty lines
    [ -z "$lib" ] && continue

    # Skip paxml in main env check
    [ "$lib" = "paxml" ] && continue

    # Check if importable. If NOT importable, echo the lib name to stdout.
    if python3 -c 'import importlib.util; exit(int(importlib.util.find_spec("'"$lib"'") is not None))' > /dev/null 2>&1; then
        printf '%s\n' "$lib"
    fi
done)

# Normalize detected missing libs from newlines back to spaces for pip
MISSING_LIBS=$(printf '%s' "$DETECTED_MISSING" | tr '\n' ' ')

if [ -n "$MISSING_LIBS" ]; then
    printf '   ⚠️  Missing libraries:%b %s%b\n' "${YELLOW}" "$MISSING_LIBS" "${NC}"
    printf '   📦 Attempting to install missing libraries for live scanning...\n'

    # Try install. Don't exit on fail (Ghost Mode fallbacks).
    # shellcheck disable=SC2086
    pip_install $MISSING_LIBS || true
else
    printf '   ✅ All core libraries detected.\n'
fi

# ------------------------------------------------------------------------------
# 2. DYNAMIC DISCOVERY
# ------------------------------------------------------------------------------
# Query internal registry for loaded adapters, sorted by priority.
REGISTERED_FWS=$(python3 -c "from ml_switcheroo.frameworks import available_frameworks, get_adapter; print(' '.join(sorted(available_frameworks(), key=lambda x: (getattr(get_adapter(x), 'ui_priority', 999) if get_adapter(x) else 999, x))))")

if [ -z "$REGISTERED_FWS" ]; then
    printf '%b❌ No frameworks found locally! Check src/ml_switcheroo/frameworks/__init__.py%b\n' "${RED}" "${NC}"
    exit 1
fi

printf '   🔍 Detected adapters: %b%s%b\n' "${YELLOW}" "$REGISTERED_FWS" "${NC}"

# ------------------------------------------------------------------------------
# 3. IMPORT UPSTREAM SPECS (Hub Population)
# ------------------------------------------------------------------------------
printf '\n%b[3/6] Importing Upstream Specifications...%b\n' "${BLUE}" "${NC}"

# Tier A: Array API
if [ ! -d "$TEMP_DIR/array-api" ]; then
    printf '   ⬇️  Cloning Array API Standard...\n'
    git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1
fi
printf '   Parsings Array API Stubs...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# Tier B: ONNX Neural
if [ ! -d "$TEMP_DIR/onnx" ]; then
    printf '   ⬇️  Cloning ONNX Operators...\n'
    git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1
fi
printf '   Parsings ONNX Operators...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# Tier C: StableHLO
if [ ! -d "$TEMP_DIR/stablehlo" ]; then
    printf '   ⬇️  Cloning StableHLO Spec...\n'
    git clone --depth 1 https://github.com/openxla/stablehlo "$TEMP_DIR/stablehlo" > /dev/null 2>&1
fi
printf '   Parsing StableHLO Spec...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/stablehlo/docs/spec.md"

# Tier D: Internals
printf '   Importing Internal Golden Sets...\n'
python3 -m ml_switcheroo import-spec internal

# ------------------------------------------------------------------------------
# 4. CONSENSUS & SCAFFOLDING (Spoke Population)
# ------------------------------------------------------------------------------
printf '\n%b[4/6] Discovering & Scaffolding Standards...%b\n' "${BLUE}" "${NC}"

# Filter out virtual frameworks from consensus to avoid errors
REAL_FWS=$(echo "$REGISTERED_FWS" | sed 's/\bhtml\b//g' | sed 's/\blatex_dsl\b//g' | sed 's/\btikz\b//g' | sed 's/\bmlir\b//g' | sed 's/\bstablehlo\b//g')

# Consensus
printf '   🤝 Running Consensus Engine (on %s)...\n' "$REAL_FWS"
# We deliberately rely on word splitting here to pass frameworks as separate args
# shellcheck disable=SC2086
python3 -m ml_switcheroo sync-standards --categories layer activation loss optimizer --frameworks $REAL_FWS

# Scaffolding
printf '   🏗️  Applying Heuristics...\n'
python3 -m ml_switcheroo scaffold --frameworks "${REGISTERED_FWS}"

# ------------------------------------------------------------------------------
# 5. GHOST SNAPSHOT & SYNC
# ------------------------------------------------------------------------------
printf '\n%b[5/6] Syncing Implementations & Capturing Snapshots...%b\n' "${BLUE}" "${NC}"

python3 -m ml_switcheroo snapshot --out-dir "$SNAPSHOTS_DIR"

# Loop through registered frameworks using POSIX read loop
printf '%s' "$REGISTERED_FWS" | tr ' ' '\n' | while read -r fw; do
    [ -z "$fw" ] && continue

    # Skip virtual frameworks that have no runtime implementation to sync
    case "$fw" in
        html|latex_dsl|tikz|mlir|stablehlo)
            # printf '   ⏭️  Skipping virtual framework %s\n' "$fw"
            continue
            ;;
    esac

    printf '   👉 Syncing %s... ' "$fw"
    if python3 -m ml_switcheroo sync "$fw" > /dev/null 2>&1; then
        printf '%b✔ Done%b\n' "${GREEN}" "${NC}"
    else
        printf '%bFailed (Checking fallback)%b\n' "${RED}" "${NC}"

        # FALLBACK: Restore from Git if generation failed
        if [ -d "$ROOT_DIR/.git" ]; then
             printf '      ↪ Attempting restore from git... '
             if git checkout -- "$SNAPSHOTS_DIR/${fw}_v"* 2>/dev/null; then
                  printf '%b✔ Restored%b\n' "${GREEN}" "${NC}"
             else
                  printf '%bNo snapshot in git%b\n' "${YELLOW}" "${NC}"
             fi
        fi
    fi
done

# ------------------------------------------------------------------------------
# 6. OPTIONAL EXTRAS (PaxML handling)
# ------------------------------------------------------------------------------
# Check if PaxML is in the string list
if printf '%s\n' "$REGISTERED_FWS" | grep -q "paxml"; then
    if [ -d "$VENV_DIR/.venv-pyenv-3-10" ] || [ -d "$VENV_DIR/.venv-uv-3-10" ]; then

        printf '\n%b[Optional] Syncing PaxML via Py3.10 venv...%b\n' "${BLUE}" "${NC}"

        ACTIVATE_SCRIPT=""
        if [ -d "$VENV_DIR/.venv-pyenv-3-10" ]; then
            ACTIVATE_SCRIPT="$VENV_DIR/.venv-pyenv-3-10/bin/activate"
        else
            ACTIVATE_SCRIPT="$VENV_DIR/.venv-uv-3-10/bin/activate"
        fi

        # Run in a subshell (paren) to isolate environment activation
        if (
            # POSIX sourcing uses '.'
            # shellcheck disable=SC1090
            . "$ACTIVATE_SCRIPT"
            printf '   📦 Ensuring PaxML dependencies in venv...\n'
            python3 -m pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1 || true
            ml_switcheroo sync paxml
        ); then
            printf '   %b✔ PaxML Synced (from venv)%b\n' "${GREEN}" "${NC}"
        else
            printf '   %b❌ PaxML Sync Failed (check venv)%b\n' "${RED}" "${NC}"

            # Git Restore Fallback
            if [ -d "$ROOT_DIR/.git" ]; then
                printf '      ↪ Attempting restore from git... '
                if git checkout -- "$SNAPSHOTS_DIR/paxml_v"* 2>/dev/null; then
                     printf '%b✔ Restored%b\n' "${GREEN}" "${NC}"
                else
                     printf '%bNo snapshot in git%b\n' "${YELLOW}" "${NC}"
                fi
            fi
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 7. CLEANUP
# ------------------------------------------------------------------------------
printf '\n%b[6/6] Cleanup...%b\n' "${BLUE}" "${NC}"
rm -rf -- "$TEMP_DIR"

printf '%s\n' "----------------------------------------"
printf '%b✅ Bootstrap Complete.%b\n' "${GREEN}" "${NC}"
printf '   Specs: %s\n' "$SEMANTICS_DIR"
printf '   Maps:  %s\n' "$SNAPSHOTS_DIR"
