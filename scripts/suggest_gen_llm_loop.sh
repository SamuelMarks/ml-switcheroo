#!/bin/sh

set -e

suggest_gen_llm_loop() {
    BASE_DIR="${BASE_DIR:-tmp/mlx_ops}"
    BASE_FILE="${BASE_FILE:-suggest_mlx_core_}"
    BASE_EXT='.md'
    while true; do
        printf 'Enter next NUMBER for "%s/%sNUMBER%s":\n\t' "$BASE_DIR" "$BASE_FILE" "$BASE_EXT"
        read -r user_input
        if [ -n "$user_input" ]; then
          case "$user_input" in
              ''|*[!0-9]*)  echo "Invalid input. Please enter a valid number." ;;
              *)
                n="$(printf "%03d" "$user_input")"
                f="${BASE_DIR}"'/'"${BASE_FILE}${n}""${BASE_EXT}"
                printf '%s\n---\n%s\n---\n%s\n' \
                  "$(cat -- "$f")" \
                  "$(code2prompt -e '*.lock,*.json,*.yml,*.yaml,test_*,sphinx*' --token-map .)" \
                  'I expect YAML for 30 operations for: PyTorch, MLX, Keras, TensorFlow (if different from Keras), Flax NNX (if different from JAX), and Pax (if different from JAX).' \
                  | pbcopy

                out='src/ml_switcheroo/frameworks/definitions/'"${BASE_FILE}${n}"'.yml'
                printf 'f=%s\n' "$f"
                printf 'out=%s\n' "$out"
                printf 'Everything is in your clipboard, provide it to your LLM. Then provide the file, e.g.:\n%s%s%s\n\n' \
                       "pbpaste > '" \
                       "$out" \
                       "'"
                printf 'Press ENTER once you'"'"'ve pasted for a sanity check\n\t'
                read -r _

                printf 'Requested: %d\nGenerated: %d\n\n' \
                        "$(grep -F 'Name:' "$f" | sort -u | wc -l)" \
                        "$(grep -Fc 'operation:' "$out")"

                ;;
          esac
        else
            # Input is not a valid number
            echo "Invalid input. Please enter a valid number."
        fi
    done
}

suggest_gen_llm_loop
