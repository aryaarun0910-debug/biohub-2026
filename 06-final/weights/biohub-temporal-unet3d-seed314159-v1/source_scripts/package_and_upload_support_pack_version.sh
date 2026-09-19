#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${BIOHUB_PROJECT_ROOT:-$(cd "$SIDECAR_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"

"$SIDECAR_DIR/scripts/bootstrap_project_payload.sh"

VENV_DIR="${BIOHUB_VENV_DIR:-$PROJECT_ROOT/.venv-biohub-gpu}"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [[ -z "${BIOHUB_TRAIN_METHOD:-}" ]]; then
  echo "BIOHUB_TRAIN_METHOD is required so the correct checkpoint is packaged." >&2
  echo "Example: BIOHUB_TRAIN_METHOD=unet_transformer_alltrain_seed314159_v1" >&2
  exit 1
fi

PYTHON_BIN="${BIOHUB_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Keep the local artifact name informative while uploading it as a Kaggle
# Dataset version. By default this versions the existing support-pack dataset,
# so submission notebooks do not need their /kaggle/input path changed.
ARTIFACT_TAG="${BIOHUB_ARTIFACT_TAG:-106ep-v1}"
ARTIFACT_SLUG="${BIOHUB_ARTIFACT_SLUG:-biohub-tracking-support-pack-${ARTIFACT_TAG}}"
ARTIFACT_OUT="${BIOHUB_ARTIFACT_OUT:-assets/${ARTIFACT_SLUG}}"
WEIGHTS_SOURCE="${BIOHUB_WEIGHTS_SOURCE:-weights/${BIOHUB_TRAIN_METHOD}}"
FULL_FRAME_CENTER_SOURCE="${BIOHUB_FULL_FRAME_CENTER_SOURCE:-}"
DATASET_ID="${BIOHUB_KAGGLE_DATASET_ID:-pilkwang/biohub-tracking-support-pack-50ep-v1}"
DATASET_TITLE="${BIOHUB_KAGGLE_DATASET_TITLE:-Biohub Tracking Support Pack V1}"
VERSION_MESSAGE="${BIOHUB_KAGGLE_VERSION_MESSAGE:-Upload ${ARTIFACT_TAG} best checkpoint.}"
UPLOAD_MODE="${BIOHUB_KAGGLE_UPLOAD_MODE:-version}"

BEST_WEIGHT="$WEIGHTS_SOURCE/split_0/edge_predictor_best.pth"
LAST_CKPT="$WEIGHTS_SOURCE/split_0/checkpoint_last.pth"
CONFIG_JSON="$WEIGHTS_SOURCE/split_0/config.json"

echo "== Biohub support pack package + version upload =="
echo "project:       $PROJECT_ROOT"
echo "method:        $BIOHUB_TRAIN_METHOD"
echo "weights:       $WEIGHTS_SOURCE"
if [[ -n "$FULL_FRAME_CENTER_SOURCE" ]]; then
  echo "center model:  $FULL_FRAME_CENTER_SOURCE"
fi
echo "artifact out:  $ARTIFACT_OUT"
echo "dataset id:    $DATASET_ID"
echo "upload mode:   $UPLOAD_MODE"
echo

if [[ ! -f "$BEST_WEIGHT" ]]; then
  echo "Missing best weight: $BEST_WEIGHT" >&2
  find weights -maxdepth 5 -type f -name edge_predictor_best.pth 2>/dev/null || true
  exit 1
fi
if [[ ! -f "$LAST_CKPT" ]]; then
  echo "Missing resume checkpoint: $LAST_CKPT" >&2
  echo "Refusing to package an ambiguous/incomplete training run." >&2
  exit 1
fi
if [[ ! -f "$CONFIG_JSON" ]]; then
  METHOD_CONFIG="weights/${BIOHUB_TRAIN_METHOD}/split_0/config.json"
  if [[ "$WEIGHTS_SOURCE" != "weights/${BIOHUB_TRAIN_METHOD}" && -f "$METHOD_CONFIG" ]]; then
    echo "Snapshot is missing config.json; copying from active method config:"
    echo "  $METHOD_CONFIG -> $CONFIG_JSON"
    mkdir -p "$(dirname "$CONFIG_JSON")"
    cp -f "$METHOD_CONFIG" "$CONFIG_JSON"
  else
    echo "Missing model config: $CONFIG_JSON" >&2
    echo "The support-pack manifest requires split_0/config.json." >&2
    find weights -maxdepth 5 -type f -path '*/split_0/config.json' 2>/dev/null || true
    exit 1
  fi
fi

ls -lh "$BEST_WEIGHT"
ls -lh "$LAST_CKPT"
ls -lh "$CONFIG_JSON"

if [[ "${BIOHUB_SKIP_PACKAGE:-0}" != "1" ]]; then
  BUILD_ARGS=(
    scripts/build_biohub_support_pack.py
    --clean \
    --zip \
    --include-wheels \
    --weights-source "$WEIGHTS_SOURCE" \
    --output "$ARTIFACT_OUT" \
    --kaggle-id "$DATASET_ID" \
    --title "$DATASET_TITLE"
  )
  if [[ -n "$FULL_FRAME_CENTER_SOURCE" ]]; then
    if [[ ! -f "$FULL_FRAME_CENTER_SOURCE/best.pt" ]]; then
      echo "Missing full-frame center detector best.pt: $FULL_FRAME_CENTER_SOURCE/best.pt" >&2
      exit 1
    fi
    BUILD_ARGS+=(--full-frame-center-source "$FULL_FRAME_CENTER_SOURCE")
  fi
  "$PYTHON_BIN" "${BUILD_ARGS[@]}"
fi

if [[ ! -f "$ARTIFACT_OUT/ARTIFACT_MANIFEST.json" ]]; then
  echo "Packaged artifact manifest is missing: $ARTIFACT_OUT/ARTIFACT_MANIFEST.json" >&2
  exit 1
fi
if [[ ! -d "$ARTIFACT_OUT/wheels" ]]; then
  echo "Packaged artifact is missing wheels/: $ARTIFACT_OUT/wheels" >&2
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found; installing pinned CLI in the active Python environment."
  "$PYTHON_BIN" -m pip install --upgrade "kaggle==1.7.4.5"
  hash -r
fi

UPLOAD_ARGS=(
  scripts/upload_kaggle_support_pack.py
  --artifact-dir "$ARTIFACT_OUT"
  --id "$DATASET_ID"
  --title "$DATASET_TITLE"
  --message "$VERSION_MESSAGE"
  --mode "$UPLOAD_MODE"
)

if [[ "${BIOHUB_KAGGLE_DELETE_OLD_VERSIONS:-0}" == "1" ]]; then
  UPLOAD_ARGS+=(--delete-old-versions)
fi
if [[ "${BIOHUB_KAGGLE_SKIP_AUTH_CHECK:-0}" == "1" ]]; then
  UPLOAD_ARGS+=(--skip-auth-check)
fi
if [[ "${BIOHUB_DRY_RUN:-0}" == "1" ]]; then
  UPLOAD_ARGS+=(--dry-run)
fi

"$PYTHON_BIN" "${UPLOAD_ARGS[@]}"

echo
echo "Done."
echo "Artifact directory: $PROJECT_ROOT/$ARTIFACT_OUT"
echo "Artifact zip:       $PROJECT_ROOT/${ARTIFACT_OUT}.zip"
echo "Kaggle dataset id:  $DATASET_ID"
