#!/usr/bin/env bash
# Downloads a run of the local model (one encoder, three heads) into public/models/stac-zap-<run>/, then builds the
# vectors of the known collections (scripts/build-vectors.ts). public/models/ is not in git.
#
# Usage: scripts/fetch-local-models.sh [run, default v7]
# With UPLOAD=1, the vectors file also goes to the model repo (shared/<run>/vectors.json), so a deployed site can
# load everything from the Hub (VITE_LOCAL_MODEL_URL=https://huggingface.co/developmentseed/stac-zap-models/resolve/main/shared/<run>/).
set -euo pipefail
RUN=${1:-v7}
REPO=developmentseed/stac-zap-models
OUT=public/models/stac-zap-$RUN
TMP=$(mktemp -d)
hf download "$REPO" --include "shared/$RUN/*" --exclude "*.pt" --exclude "*/onnx/model.onnx" --exclude "*/onnx/model_q4.onnx*" --local-dir "$TMP" >/dev/null
mkdir -p "$OUT"
cp -r "$TMP/shared/$RUN/." "$OUT/"
rm -rf "$TMP"
npx tsx scripts/build-vectors.ts "$OUT"
if [ "${UPLOAD:-0}" = 1 ]; then
  hf upload "$REPO" "$OUT/vectors.json" "shared/$RUN/vectors.json" --commit-message "vectors of the curated collections ($RUN)"
fi
du -sh "$OUT"
