#!/usr/bin/env bash
set -euo pipefail

# Downloads every Hugging Face model listed in a models file (one repo id per
# line, e.g. "Qwen/Qwen3.5-9B"; blank lines and "#" comments are skipped)
# into ./models/<model-name>.
#
# Usage:
#   ./download_hf_models.sh [models_file]
#
# Default models_file: config/huggingface_models.txt

MODELS_FILE="${1:-config/huggingface_models.txt}"
DOWNLOAD_DIR="models"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "models file not found: $MODELS_FILE" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

ok=()
failed=()

while IFS= read -r line || [[ -n "$line" ]]; do
  repo_id="$(echo "$line" | sed 's/#.*$//' | xargs)"
  [[ -z "$repo_id" ]] && continue

  model_name="${repo_id##*/}"
  target_dir="${DOWNLOAD_DIR}/${model_name}"

  echo "=== downloading ${repo_id} -> ${target_dir} ==="
  if hf download "$repo_id" --local-dir "$target_dir"; then
    ok+=("$repo_id")
  else
    echo "!!! failed: ${repo_id}" >&2
    failed+=("$repo_id")
  fi
done < "$MODELS_FILE"

echo
echo "=== summary ==="
echo "succeeded (${#ok[@]}):"
printf '  %s\n' "${ok[@]:-}"
echo "failed (${#failed[@]}):"
printf '  %s\n' "${failed[@]:-}"

[[ ${#failed[@]} -eq 0 ]]
