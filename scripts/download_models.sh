#!/usr/bin/env bash
# Download Aura's trained GPU models (full mode) from the GitHub release into artifacts/.
#
#   scripts/download_models.sh            # all models
#   scripts/download_models.sh ece llm    # only some (ece, ser, fer, llm)
#
# Uses the GitHub CLI when it is logged in (works for private repos), plain HTTPS otherwise.
set -euo pipefail

REPO="${AURA_REPO:-krish1809/aura-ai}"
TAG="${AURA_MODELS_TAG:-models-v1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/artifacts"
declare -A ASSET=([ece]=aura-ece-v1.tar [ser]=aura-ser-v1.tar [fer]=aura-fer-v1.tar [llm]=aura-llm-lora-v1.tar)

wanted=("$@")
[ ${#wanted[@]} -eq 0 ] && wanted=(ece ser fer llm)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fetch() {  # fetch <asset name>
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" -R "$REPO" -p "$1" -D "$tmp" --clobber
  else
    curl -fL --retry 3 -o "$tmp/$1" "https://github.com/$REPO/releases/download/$TAG/$1"
  fi
}

fetch SHA256SUMS
for name in "${wanted[@]}"; do
  asset="${ASSET[$name]:-}"
  [ -z "$asset" ] && { echo "unknown model '$name' (choose from: ${!ASSET[*]})" >&2; exit 1; }
  echo "downloading $asset ..."
  fetch "$asset"
  (cd "$tmp" && grep " $asset\$" SHA256SUMS | sha256sum -c --quiet -)
  tar -xf "$tmp/$asset" -C "$DEST"
  echo "  -> $DEST ($name ready)"
done
echo "Done. Start full mode with: AURA_ML_MODE=full AURA_LLM_PROVIDER=local make api"
