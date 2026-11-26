#!/usr/bin/env bash
# Downloads a run of the local models into public/models/stac-zap-<run>/, then builds the vectors of the known
# collections (scripts/build-vectors.ts). public/models/ is not in git.
#
# Usage: scripts/fetch-local-models.sh [run, default v5]
# The model repo is private for now: log in first with `hf auth login`.
set -euo pipefail
RUN=${1:-v5}
REPO=developmentseed/stac-zap-models
OUT=public/models/stac-zap-$RUN
TMP=$(mktemp -d)
for part in retriever decider tagger; do
  hf download "$REPO" --include "$part/$RUN/*" --exclude "*.safetensors" --exclude "*/onnx/model.onnx" --local-dir "$TMP" >/dev/null
  mkdir -p "$OUT/$part"
  cp -r "$TMP/$part/$RUN/." "$OUT/$part/"
done
rm -rf "$TMP"
npx tsx scripts/build-vectors.ts "$OUT"
du -sh "$OUT"
