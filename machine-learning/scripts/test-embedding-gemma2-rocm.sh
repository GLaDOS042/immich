#!/usr/bin/env bash
set -euo pipefail

ML_URL="${ML_URL:-http://127.0.0.1:3003}"
MODEL="${EMBEDDING_GEMMA2_ALIAS:-ViT-B-16-SigLIP-256__webli}"
IMAGE="${1:-}"

request_json() {
  local output rc
  if output="$(curl -sS --fail-with-body "$@")"; then
    printf '%s' "${output}"
    return 0
  else
    rc=$?
    echo "ML request failed (curl exit ${rc})." >&2
    if [[ -n "${output}" ]]; then
      echo "Response body:" >&2
      printf '%s\n' "${output}" >&2
    fi
    echo "Check the ML worker logs with:" >&2
    echo "  podman logs --tail=200 immich-ml-test" >&2
    return "${rc}"
  fi
}

if ! curl -fsS "${ML_URL}/ping" >/dev/null; then
  echo "ML service is not healthy at ${ML_URL}." >&2
  echo "Check: podman ps -a --filter name=immich-ml-test && podman logs --tail=200 immich-ml-test" >&2
  exit 1
fi

text_json="$(request_json \
  -F "entries={\"clip\":{\"textual\":{\"modelName\":\"${MODEL}\"}}}" \
  -F 'text=cat sleeping on a sofa' \
  "${ML_URL}/predict")"

text_dim="$(printf '%s' "${text_json}" | python -c 'import json,sys; r=json.load(sys.stdin); print(len(json.loads(r["clip"])))')"

if [[ "${text_dim}" != "768" ]]; then
  echo "text embedding dimension: ${text_dim} (expected 768)" >&2
  exit 1
fi

echo "text embedding dimension: 768"

if [[ -n "${IMAGE}" ]]; then
  image_json="$(request_json \
    -F "entries={\"clip\":{\"visual\":{\"modelName\":\"${MODEL}\"}}}" \
    -F "image=@${IMAGE}" \
    "${ML_URL}/predict")"

  image_result="$(printf '%s' "${image_json}" | python -c 'import json,sys; r=json.load(sys.stdin); print(len(json.loads(r["clip"])), r["imageWidth"], r["imageHeight"])')"

  read -r image_dim image_width image_height <<<"${image_result}"
  if [[ "${image_dim}" != "768" ]]; then
    echo "image embedding dimension: ${image_dim} (expected 768)" >&2
    exit 1
  fi

  echo "image embedding dimension: 768 (${image_width}x${image_height})"
fi

echo "API smoke test passed. Confirm container logs list MIGraphXExecutionProvider first."
