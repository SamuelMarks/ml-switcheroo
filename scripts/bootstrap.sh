#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
#
# This script hydrates the Semantic Knowledge Base (Hub) and Framework
# Implementation Maps (Spokes) from upstream sources and the local environment.
#
# It dynamically queries the 'ml_switcheroo' package to discover all supported
# frameworks (including new plugins added to src/ml_switcheroo/frameworks/).
#
# Stages:
# 1. Ingestion: Import external specs (Array API, ONNX).
# 2. Discovery: Find consensus standards across ALL registered frameworks.
# 3. Scaffolding: Heuristic mapping for ALL registered frameworks.
# 4. Snapshots: Capture API signatures for Ghost Mode.
# 5. Sync: Link implementations to abstract standards.
# ==============================================================================

set -e

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ROOT_DIR="$(dirname -- "$(dirname -- "$(realpath -- "$0")")")"
VENV_DIR="${VENV_DIR:-$ROOT_DIR}"
TEMP_DIR="_bootstrap_temp"
SEMANTICS_DIR="$ROOT_DIR"'/src/ml_switcheroo/semantics'
SNAPSHOTS_DIR="$ROOT_DIR"'/src/ml_switcheroo/snapshots'

echo "${BOLD}🔥 Initiating Knowledge Base Bootstrap (Dynamic)${NC}"
echo "----------------------------------------"

# ------------------------------------------------------------------------------
# 0. SETUP & CLEANUP
# ------------------------------------------------------------------------------
echo "${BLUE}[0/6] Preparing Workspace...${NC}"

# Clean existing JSON artifacts to force regeneration
rm -f "$SEMANTICS_DIR"/k_*.json
rm -f "$SNAPSHOTS_DIR"/*_map.json
rm -f "$SNAPSHOTS_DIR"/*_v*.json

# Prepare temp dir
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Install ml-switcheroo in customizable mode to ensure we can query it
if command -v uv >/dev/null 2>&1; then
    uv pip install -e . > /dev/null 2>&1
else
    python3 -m pip install -e . > /dev/null 2>&1
fi

# ------------------------------------------------------------------------------
# 1. DYNAMIC DISCOVERY OF FRAMEWORKS
# ------------------------------------------------------------------------------
echo "\n${BLUE}[1/6] querying registered frameworks...${NC}"

# We use python to inspect the registry. This picks up any file added to
# src/ml_switcheroo/frameworks/ provided it has the @register_framework decorator.
# We set PYTHONPATH to src to ensure we load the local version.
REGISTERED_FWS=$(PYTHONPATH=src python3 -c "from ml_switcheroo.frameworks import available_frameworks; print(' '.join(available_frameworks()))")

if [ -z "$REGISTERED_FWS" ]; then
    echo "${RED}❌ No frameworks found! Check src/ml_switcheroo/frameworks/__init__.py${NC}"
    exit 1
fi

echo "   🔍 Detected adapters: ${YELLOW}$REGISTERED_FWS${NC}"

# Attempt to install corresponding libraries for live scanning
echo "   📦 Attempting to install libraries for live scanning..."
# Note: failure here is tolerabe (Ghost Mode fallback), so we use || true
if command -v uv >/dev/null 2>&1; then
    uv pip install torch jax flax tensorflow keras mlx numpy || true
else
    python3 -m pip install torch jax flax tensorflow keras mlx numpy || true
fi

# ------------------------------------------------------------------------------
# 2. IMPORT UPSTREAM SPECS (Hub Population)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[2/6] Importing Upstream Specifications...${NC}"

# Tier A: Array API
if [ ! -d "$TEMP_DIR/array-api" ]; then
    git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1
fi
echo "   Parsing Array API Stubs..."
ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# Tier B: ONNX Neural
if [ ! -d "$TEMP_DIR/onnx" ]; then
    git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1
fi
echo "   Parsing ONNX Operators..."
ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# Tier C: Internals
echo "   Importing Internals..."
ml_switcheroo import-spec internal

# ------------------------------------------------------------------------------
# 3. CONSENSUS & SCAFFOLDING (Spoke Population)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[3/6] Discovering & Scaffolding Standards...${NC}"

# Consensus: Find intersection of APIs across ALL registered frameworks
# Explicitly expand the python list to space-separated args
echo "   🤝 Running Consensus on: $REGISTERED_FWS"
# shellcheck disable=SC2086
ml_switcheroo sync-standards --categories layer activation loss optimizer --frameworks $REGISTERED_FWS

# Scaffolding: Apply heuristics for ALL registered frameworks
echo "   🏗️  Scaffolding Heuristics..."
# shellcheck disable=SC2086
ml_switcheroo scaffold --frameworks $REGISTERED_FWS

# ------------------------------------------------------------------------------
# 4. GHOST SNAPSHOT CAPTURE
# ------------------------------------------------------------------------------
echo "\n${BLUE}[4/6] Capturing API Snapshots (Ghost Mode)...${NC}"
# Dumps raw API signatures for client-side usage (WASM)
ml_switcheroo snapshot --out-dir "$SNAPSHOTS_DIR"

# ------------------------------------------------------------------------------
# 5. IMPLEMENTATION SYNC
# ------------------------------------------------------------------------------
echo "\n${BLUE}[5/6] Syncing Maps...${NC}"

# Iterate dynamically discovered list
for fw in $REGISTERED_FWS; do
    printf '   👉 Syncing %s... ' "$fw"
    if ml_switcheroo sync "$fw" > /dev/null 2>&1; then
        echo "${GREEN}✔ Done${NC}"
    else
        echo "${RED}Skipped (Sync Failed)${NC}"
    fi
done

# ------------------------------------------------------------------------------
# 6. OPTIONAL EXTRAS (PaxML handling)
# ------------------------------------------------------------------------------
# PaxML often requires specific python versions (3.10).
# If available in the environment, it is already handled by the loop above.
# If not, we try specific venvs only if they exist.
if echo "$REGISTERED_FWS" | grep -q "paxml"; then
    if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ] || [ -d "$VENV_DIR"'/.venv-uv-3-10' ]; then
        echo "\n${BLUE}[Optional] Syncing PaxML via Py3.10 venv...${NC}"
        # Temporarily switch context
        if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ]; then
            . "$VENV_DIR"'/.venv-pyenv-3-10/bin/activate'
        else
            . "$VENV_DIR"'/.venv-uv-3-10/bin/activate'
        fi

        # Install deps if needed
        python3 -m pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1 || true

        ml_switcheroo sync paxml
        echo "   ${GREEN}✔ PaxML Synced${NC}"
    fi
fi

# ------------------------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------------------------
echo "\n${BOLD}🧹 Cleanup...${NC}"
rm -rf "$TEMP_DIR"

echo "----------------------------------------"
echo "${GREEN}✅ Bootstrap Complete.${NC}"
echo "   Specs: $SEMANTICS_DIR"
echo "   Maps:  $SNAPSHOTS_DIR"
