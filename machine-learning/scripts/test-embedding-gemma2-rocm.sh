#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://127.0.0.1:3003}"
MODEL="${EMBEDDING_GEMMA2_ALIAS:-ViT-B-16-SigLIP-256__webli}"
IMAGE="${1:-}"

text_dim="$({
  curl -fsS \
    -F "entries={\"clip\":{\"textual\":{\"modelName\":\"${MODEL}\"}}}" \
    -F 'text=cat sleeping on a sofa' \
    "${ML_URL}/predict"
} | python -c 'import json,sys; r=json.load(sys.stdin); print(len(json.loads(r["clip"])))')"

if [[ "${text_dim}" != "768" ]]; then
  echo "text embedding dimension: ${text_dim} (expected 768)" >&2
  exit 1
fi

echo "text embedding dimension: 768"

if [[ -n "${IMAGE}" ]]; then
  image_result="$({
    curl -fsS \
      -F "entries={\"clip\":{\"visual\":{\"modelName\":\"${MODEL}\"}}}" \
      -F "image=@${IMAGE}" \
      "${ML_URL}/predict"
  } | python -c 'import json,sys; r=json.load(sys.stdin); print(len(json.loads(r["clip"])), r["imageWidth"], r["imageHeight"])')"

  read -r image_dim image_width image_height <<<"${image_result}"
  if [[ "${image_dim}" != "768" ]]; then
    echo "image embedding dimension: ${image_dim} (expected 768)" >&2
    exit 1
  fi

  echo "image embedding dimension: 768 (${image_width}x${image_height})"
fi

echo "API smoke test passed. Confirm container logs list MIGraphXExecutionProvider first."
