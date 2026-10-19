#!/usr/bin/env bash

# Greps for hardcoded framework routing logic in python files

FAILED=0
PATTERN="(if|elif|return)[[:space:]]+(self\.|ctx\.|engine\.)?(framework|target|source|target_fw|source_fw)[[:space:]]+(==|!=|in)[[:space:]]*(\{|\[|\(|str\()?[[:space:]]*['\"](jax|torch|mlx|tensorflow|keras|numpy|flax|flax_nnx|paxml)['\"]"

for file in "$@"; do
    if [[ "$file" == *.py ]]; then
        if grep -q -E "$PATTERN" "$file"; then
            echo "Error: Hardcoded framework routing found in $file"
            grep -n -E "$PATTERN" "$file"
            echo "Please use JSON variants or traits for dispatch logic."
            FAILED=1
        fi
    fi
done

exit $FAILED
