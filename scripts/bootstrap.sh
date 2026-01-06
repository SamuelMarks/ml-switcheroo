#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
#
# Description:
#   This script automates the hydration of the Semantic Knowledge Base (The Hub)
#   and Implementation Snapshots (The Spokes).
#
#   Workflow:
#   1.  **Environment Audit**: Checks for installed ML libraries.
#   2.  **Auto-Installation**: Installs missing dependencies (Live Mode).
#       - Intelligently handles 'uv' vs 'pip'.
#       - Falls back to specific Python 3.10 venvs for PaxML compatibility.
#       - Ensures 'torchvision' is installed alongside 'torch'.
#   3.  **Ingestion**: Downloads and parses upstream standards.
#   4.  **Discovery**: Runs Consensus to find standard neural operations.
#   5.  **Scaffolding**: Maps APIs via heuristics (Skipping virtual frameworks).
#   6.  **Ghost Mode**: Captures snapshots and syncs framework implementations.
#
# Usage:
#   ./scripts/bootstrap.sh
# ==============================================================================

set -e

# Visual helpers (ANSI Colors)
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# Directory Resolution
# ------------------------------------------------------------------------------
# Function: get_abs_script_path
# Returns the absolute path of the directory containing this script.
get_abs_script_path() {
    (
        CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P
    )
}

SCRIPT_LOCATION=$(get_abs_script_path)
ROOT_DIR=$(dirname -- "$SCRIPT_LOCATION")

VENV_DIR="${VENV_DIR:-$ROOT_DIR}"
SNAPSHOTS_DIR="$ROOT_DIR/src/ml_switcheroo/snapshots"
SEMANTICS_DIR="$ROOT_DIR/src/ml_switcheroo/semantics"
TEMP_DIR="_bootstrap_temp"

printf '%b🔥 Initiating Knowledge Base Bootstrap%b\n' "${BOLD}" "${NC}"
printf '%s\n' "----------------------------------------"

# ------------------------------------------------------------------------------
# 0. SETUP & CLEANUP
# ------------------------------------------------------------------------------
printf '%b[0/6] Preparing Workspace...%b\n' "${BLUE}" "${NC}"

# Clean existing JSON artifacts to force a fresh regeneration.
printf '   🧹 Cleaning old mappings and descriptors...\n'
rm -f "$SEMANTICS_DIR"/k_*.json
rm -f "$SNAPSHOTS_DIR"/*_map.json
rm -f "$SNAPSHOTS_DIR"/*_v*.json

# Prepare temporary directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Ensure the package is importable
# Returns 0 if import success, 1 otherwise
if ! python3 -c "import ml_switcheroo" >/dev/null 2>&1; then
    printf '   ⚠️  Package not found in current env. Attempting editable install...\n'
    if [ -n "$VIRTUAL_ENV" ] && command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    else
        python3 -m pip install -e .
    fi
fi

# ------------------------------------------------------------------------------
# 1. DEPENDENCY AUDIT & INSTALLATION
# ------------------------------------------------------------------------------
printf '\n%b[1/6] Auditing & Installing Dependencies...%b\n' "${BLUE}" "${NC}"

# Function: get_install_cmd
# Determines whether to use `uv pip install` or `python3 -m pip install`
# based on the presence of the `uv` tool and active virtual environment.
get_install_cmd() {
    if [ -n "$VIRTUAL_ENV" ] && command -v uv >/dev/null 2>&1; then
        echo "uv pip install"
    else
        echo "python3 -m pip install"
    fi
}

# Function: list_required_libs
# Introspects the python package to find all registered frameworks.
# Filters to ensure virtual/internal frameworks are skipped.
# Logic explicitly enforces filtering set behavior.
list_required_libs() {
    python3 -c "
try:
    from ml_switcheroo.frameworks import available_frameworks
    # Explicit ignore list for virtual/internal frameworks that are not pip packages
    # Must match keys in register_framework calls
    ignored = {'html', 'latex_dsl', 'tikz', 'mlir', 'stablehlo', 'sass'}

    all_fws = set(available_frameworks())

    # Filter
    libs = {fw for fw in all_fws if fw not in ignored}

    # Normalize aliases: flax_nnx is provided by 'flax' package
    if 'flax_nnx' in libs:
        libs.discard('flax_nnx')
        libs.add('flax')

    # Hard dependency additions: Torch needs TorchVision for some extras
    if 'torch' in libs:
        libs.add('torchvision')

    print(' '.join(sorted(libs)))
except ImportError:
    # Safe fallback
    print('torch torchvision jax flax tensorflow keras mlx numpy')
"
}

REQUIRED_LIBS=$(list_required_libs)
INSTALL_CMD=$(get_install_cmd)

printf '   🔍 Required Libraries: %b%s%b\n' "${YELLOW}" "$REQUIRED_LIBS" "${NC}"

# 1a. Check for Basic Dependencies (Numpy check)
# If the current environment is bare (missing numpy), we attempt to switch context
# to a dedicated Py3.10 venv as requested.
if ! python3 -c "import numpy" >/dev/null 2>&1; then
    printf '   ⚠️  Current environment lacks basic dependencies. Checking for pre-configured venvs...\n'

    FOUND_VENV=""
    for venv_name in ".venv-pyenv-3-10" ".venv-uv-3-10"; do
        if [ -d "$VENV_DIR/$venv_name" ]; then
            printf '   🔄 Activating fallback environment: %s\n' "$venv_name"
            # We source the activate script in the current shell context (if running interactively)
            # or just mark it for usage. Since we are in a script, we source it here
            # but note this only affects this script process.
            . "$VENV_DIR/$venv_name/bin/activate"
            FOUND_VENV="$venv_name"
            # Recalculate install command inside new venv
            INSTALL_CMD=$(get_install_cmd)
            break
        fi
    done

    if [ -z "$FOUND_VENV" ]; then
        printf '   No fallback venv found. Proceeding with installation in current environment.\n'
    fi
fi

# 1b. Install Missing Core Libraries
# We check each library individually using importlib
MISSING_LIBS=""
DETECTED_MISSING=$(printf '%s' "$REQUIRED_LIBS" | tr ' ' '\n' | while read -r lib; do
    [ -z "$lib" ] && continue
    # Skip paxml in main loop, handled specifically later
    [ "$lib" = "paxml" ] && continue

    if ! python3 -c 'import importlib.util; import sys; exit(0 if importlib.util.find_spec("'"$lib"'") else 1)' > /dev/null 2>&1; then
        printf '%s\n' "$lib"
    fi
done)

# Capture output into variable
MISSING_LIBS=$(printf '%s' "$DETECTED_MISSING" | tr '\n' ' ')

if [ -n "$MISSING_LIBS" ]; then
    printf '   📦 Installing missing core libraries: %b%s%b\n' "${YELLOW}" "$MISSING_LIBS" "${NC}"
    # shellcheck disable=SC2086
    $INSTALL_CMD $MISSING_LIBS || printf '   ⚠️  Installation failed. Will attempt Ghost Mode.\n'
else
    printf '   ✅ Core libraries satisfied.\n'
fi

# 1c. Special Handling: PaxML (Requires Py3.10)
# PaxML often conflicts with newer JAX/Python versions. We isolate logic here.
if printf '%s' "$REQUIRED_LIBS" | grep -q "paxml"; then
    CURRENT_PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    PAXML_INSTALLED=$(python3 -c 'import importlib.util; print(1 if importlib.util.find_spec("paxml") else 0)')

    if [ "$PAXML_INSTALLED" -eq 1 ]; then
         printf '   ✅ PaxML is already installed.\n'
    else
        printf '   🔍 Configuring PaxML (Requires Python 3.10). Current: %s\n' "$CURRENT_PY_VER"

        if [ "$CURRENT_PY_VER" = "3.10" ]; then
             printf '   📦 Environment matches. Installing PaxML...\n'
             $INSTALL_CMD paxml jaxlib==0.4.26 || true
        else
             # Try to find a compatible venv
             PAX_VENV_FOUND=0
             for venv_name in ".venv-pyenv-3-10" ".venv-uv-3-10"; do
                 if [ -d "$VENV_DIR/$venv_name" ]; then
                     printf '   🔄 Utilizing %s for PaxML operations...\n' "$venv_name"

                     # Execute checking/install/sync in a subshell with the venv activated
                     (
                         . "$VENV_DIR/$venv_name/bin/activate"
                         # Check presence via pip freeze
                         if ! pip freeze | grep -q "paxml"; then
                             printf '      📦 Installing PaxML in venv...\n'
                             CMD=$(get_install_cmd)
                             # Note: Install command needs to be recalculated inside subshell or passed explicitly
                             if [ -n "$VIRTUAL_ENV" ] && command -v uv >/dev/null 2>&1; then
                                 CMD="uv pip install"
                             else
                                 CMD="python3 -m pip install"
                             fi
                             $CMD paxml jaxlib==0.4.26 > /dev/null 2>&1 || true
                         fi
                         # Run Sync immediately in context
                         printf '      👉 Syncing paxml (in venv)... '
                         python3 -m ml_switcheroo sync paxml > /dev/null 2>&1 && printf '%b✔ Done%b\n' "${GREEN}" "${NC}" || printf '%bFailed%b\n' "${RED}" "${NC}"
                     )
                     PAX_VENV_FOUND=1
                     break
                 fi
             done

             if [ "$PAX_VENV_FOUND" -eq 0 ]; then
                 printf '   ⚠️  Skipping PaxML (No Py3.10 env found).\n'
             fi
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 2. DYNAMIC DISCOVERY
# ------------------------------------------------------------------------------
# Re-query registry after potential installations to get updated list/priorities
REGISTERED_FWS=$(python3 -c "from ml_switcheroo.frameworks import available_frameworks, get_adapter; print(' '.join(sorted(available_frameworks(), key=lambda x: (getattr(get_adapter(x), 'ui_priority', 999) if get_adapter(x) else 999, x))))")

if [ -z "$REGISTERED_FWS" ]; then
    printf '%b❌ No frameworks found locally! Check src/ml_switcheroo/frameworks/__init__.py%b\n' "${RED}" "${NC}"
    exit 1
fi

printf '   🔍 Active Adapters: %b%s%b\n' "${YELLOW}" "$REGISTERED_FWS" "${NC}"

# ------------------------------------------------------------------------------
# 3. IMPORT UPSTREAM SPECS (Hub Population)
# ------------------------------------------------------------------------------
printf '\n%b[3/6] Importing Upstream Specifications (The Hub)...%b\n' "${BLUE}" "${NC}"

# 3.1. Tier A: Array API Standard (Math)
if [ ! -d "$TEMP_DIR/array-api" ]; then
    printf '   ⬇️  Cloning Array API Standard (2024.12)...\n'
    git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1
fi
printf '   📜 Parsing Array API Stubs...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# 3.2. Tier B: ONNX Operators (Neural)
if [ ! -d "$TEMP_DIR/onnx" ]; then
    printf '   ⬇️  Cloning ONNX Operators (v1.20.0)...\n'
    git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1
fi
printf '   📜 Parsing ONNX Operators.md...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# 3.3. Tier C: StableHLO (Intermediate Representation)
if [ ! -d "$TEMP_DIR/stablehlo" ]; then
    printf '   ⬇️  Cloning StableHLO Spec...\n'
    git clone --depth 1 https://github.com/openxla/stablehlo "$TEMP_DIR/stablehlo" > /dev/null 2>&1
fi
printf '   📜 Parsing StableHLO Spec...\n'
python3 -m ml_switcheroo import-spec "$TEMP_DIR/stablehlo/docs/spec.md"

# 3.4. Tier D: Internal Defaults
printf '   🔄 Hydrating Internal Golden Sets...\n'
python3 -m ml_switcheroo import-spec internal

# 3.5. Tier E: NVIDIA SASS (Binary Utilities Spec)
if [ ! -f "$TEMP_DIR/cuda_binary_utils.html" ]; then
    printf '   ⬇️  Downloading NVIDIA SASS Documentation...\n'
    if command -v curl >/dev/null 2>&1; then
      curl -sL -o "$TEMP_DIR/cuda_binary_utils.html" \
          "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html" || true
    else
      printf '      ⚠️ curl not found, skipping SASS download.\n'
    fi
fi

if [ -f "$TEMP_DIR/cuda_binary_utils.html" ]; then
    printf '   📜 Parsing SASS Instruction Sets from HTML...\n'
    python3 -m ml_switcheroo import-spec "$TEMP_DIR/cuda_binary_utils.html"
else
    printf '   ⚠️  Skipping SASS import (file missing)\n'
fi

# ------------------------------------------------------------------------------
# 4. CONSENSUS & SCAFFOLDING (The Spokes)
# ------------------------------------------------------------------------------
printf '\n%b[4/6] Disovering & Scaffolding Implementations...%b\n' "${BLUE}" "${NC}"

# Filter out virtual frameworks from consensus to avoid noise.
# We strip 'sass' here to ensure consensus doesn't try to introspect a non-existent python module.
# shellcheck disable=SC2046
REAL_FWS=$(echo "$REGISTERED_FWS" | sed 's/\bhtml\b//g' | sed 's/\blatex_dsl\b//g' | sed 's/\btikz\b//g' | sed 's/\bmlir\b//g' | sed 's/\bstablehlo\b//g' | sed 's/\bsass\b//g')

# 4.1. Consensus Engine
# Finds "De Facto" standards for operations not covered by official bodies.
printf '   🤝 Running Consensus Engine (Targets: %s)...\n' "$REAL_FWS"
# shellcheck disable=SC2086
python3 -m ml_switcheroo sync-standards --categories layer activation loss optimizer --frameworks $REAL_FWS

# 4.2. Scaffolding
# Uses regex heuristics to map remaining APIs.
printf '   🏗️  Applying Heuristic Scaffolding...\n'
# We loop over frameworks individually.
for fw in $REGISTERED_FWS; do
    # Skip virtual frameworks or those handled strictly by venv logic.
    # This prevents the inspector from crashing on missing packages (like mlir, stablehlo, paxml).
    case "$fw" in
        html|latex_dsl|tikz|mlir|stablehlo|sass|paxml)
            continue
            ;;
    esac

    printf '      -> Scaffolding %s\n' "$fw"
    # Allow scaffolding to fail (e.g. TF Abort trap) without stopping bootstrap
    python3 -m ml_switcheroo scaffold --frameworks "$fw" || printf '      ⚠️  Scaffolding failed for %s (Skipping)\n' "$fw"
done

# ------------------------------------------------------------------------------
# 5. GHOST SNAPSHOT & SYNC
# ------------------------------------------------------------------------------
printf '\n%b[5/6] Capturing Snapshots & Syncing...%b\n' "${BLUE}" "${NC}"

# 5.1. Capture Snapshots (Raw API signatures)
printf '   📸 Capturing API Snapshots...\n'
python3 -m ml_switcheroo snapshot --out-dir "$SNAPSHOTS_DIR"

# 5.2. Sync Mappings (Link Specs to Implementation)
# shellcheck disable=SC2086
for fw in $REGISTERED_FWS; do
    # Skip virtual frameworks that have no runtime implementation to sync
    case "$fw" in
        html|latex_dsl|tikz|mlir|stablehlo)
            continue
            ;;
        paxml)
            # PaxML is handled in the specialized block above/below if environment permits
            continue
            ;;
    esac

    printf '   👉 Syncing %-15s ... ' "$fw"
    if python3 -m ml_switcheroo sync "$fw" > /dev/null 2>&1; then
        printf '%b✔ Done%b\n' "${GREEN}" "${NC}"
    else
        printf '%bFailed (Checking fallback)%b\n' "${RED}" "${NC}"

        # Git Fallback Logic
        if [ -d "$ROOT_DIR/.git" ]; then
             if git checkout -- "$SNAPSHOTS_DIR/${fw}_v"* > /dev/null 2>&1; then
                  printf '      ↪ Restored snapshot from git %b✔%b\n' "${GREEN}" "${NC}"
             else
                  printf '      ↪ No snapshot in git %b(Ignored)%b\n' "${YELLOW}" "${NC}"
             fi
        fi
    fi
done

# ------------------------------------------------------------------------------
# 6. CLEANUP
# ------------------------------------------------------------------------------
printf '\n%b[6/6] Cleaning Up...%b\n' "${BLUE}" "${NC}"
rm -rf -- "$TEMP_DIR"

printf '%s\n' "----------------------------------------"
printf '%b✅ Bootstrap Complete.%b\n' "${GREEN}" "${NC}"
printf '   Specs Path:    %s\n' "$SEMANTICS_DIR"
printf '   Mappings Path: %s\n' "$SNAPSHOTS_DIR"
