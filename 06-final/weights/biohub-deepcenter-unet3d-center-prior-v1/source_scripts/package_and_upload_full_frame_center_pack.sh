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

PYTHON_BIN="${BIOHUB_PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

METHOD="${BIOHUB_FULL_FRAME_METHOD:-full_frame_center_v1}"
SOURCE_DIR="${BIOHUB_FULL_FRAME_CENTER_SOURCE:-weights/$METHOD}"
ARTIFACT_TAG="${BIOHUB_FULL_FRAME_ARTIFACT_TAG:-${METHOD}}"
ARTIFACT_SLUG="${BIOHUB_FULL_FRAME_ARTIFACT_SLUG:-biohub-full-frame-center-pack-${ARTIFACT_TAG}}"
ARTIFACT_OUT="${BIOHUB_FULL_FRAME_ARTIFACT_OUT:-assets/${ARTIFACT_SLUG}}"
DATASET_ID="${BIOHUB_FULL_FRAME_KAGGLE_DATASET_ID:-pilkwang/biohub-full-frame-center-pack-v1}"
DATASET_TITLE="${BIOHUB_FULL_FRAME_KAGGLE_DATASET_TITLE:-Biohub Full-Frame Center Detector Pack V1}"
VERSION_MESSAGE="${BIOHUB_FULL_FRAME_KAGGLE_VERSION_MESSAGE:-Upload ${ARTIFACT_TAG} full-frame center detector.}"
UPLOAD_MODE="${BIOHUB_FULL_FRAME_KAGGLE_UPLOAD_MODE:-auto}"
DIR_MODE="${BIOHUB_KAGGLE_DIR_MODE:-zip}"

echo "== Biohub full-frame center pack package + upload =="
echo "project:      $PROJECT_ROOT"
echo "method:       $METHOD"
echo "source:       $SOURCE_DIR"
echo "artifact out: $ARTIFACT_OUT"
echo "dataset id:   $DATASET_ID"
echo "upload mode:  $UPLOAD_MODE"
echo

for required in \
  "$SOURCE_DIR/best.pt" \
  "$SOURCE_DIR/checkpoint_last.pt" \
  "$SOURCE_DIR/config.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required full-frame artifact: $required" >&2
    find weights -maxdepth 4 -type f \( -name best.pt -o -name checkpoint_last.pt -o -name config.json \) 2>/dev/null || true
    exit 1
  fi
done

ls -lh "$SOURCE_DIR/best.pt"
ls -lh "$SOURCE_DIR/checkpoint_last.pt"
ls -lh "$SOURCE_DIR/config.json"
for optional in \
  "$SOURCE_DIR/history.csv" \
  "$SOURCE_DIR/split_manifest.json" \
  "$SOURCE_DIR/gate_summary.json" \
  "$SOURCE_DIR/gate_threshold_metrics.csv" \
  "$SOURCE_DIR/gate_frame_metrics.csv" \
  "$SOURCE_DIR/gate_peak_samples.csv"; do
  if [[ -f "$optional" ]]; then
    ls -lh "$optional"
  else
    echo "Warning: optional diagnostic is missing: $optional" >&2
  fi
done

if [[ "${BIOHUB_SKIP_PACKAGE:-0}" != "1" ]]; then
  "$PYTHON_BIN" scripts/build_full_frame_center_pack.py \
    --clean \
    --zip \
    --source "$SOURCE_DIR" \
    --output "$ARTIFACT_OUT" \
    --kaggle-id "$DATASET_ID" \
    --title "$DATASET_TITLE"
fi

if [[ ! -f "$ARTIFACT_OUT/ARTIFACT_MANIFEST.json" ]]; then
  echo "Packaged manifest is missing: $ARTIFACT_OUT/ARTIFACT_MANIFEST.json" >&2
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found; installing pinned CLI in the active Python environment."
  "$PYTHON_BIN" -m pip install --upgrade "kaggle==1.7.4.5"
  hash -r
fi

if [[ "${BIOHUB_KAGGLE_SKIP_AUTH_CHECK:-0}" != "1" ]]; then
  kaggle datasets list --mine -p 1 >/dev/null
fi

upload_exists=0
case "$UPLOAD_MODE" in
  version)
    upload_exists=1
    ;;
  create)
    upload_exists=0
    ;;
  auto)
    if kaggle datasets status "$DATASET_ID" --format json >/tmp/biohub_full_frame_dataset_status.json 2>/tmp/biohub_full_frame_dataset_status.err; then
      upload_exists=1
      cat /tmp/biohub_full_frame_dataset_status.json || true
    else
      upload_exists=0
      cat /tmp/biohub_full_frame_dataset_status.err || true
    fi
    ;;
  *)
    echo "Unsupported BIOHUB_FULL_FRAME_KAGGLE_UPLOAD_MODE=$UPLOAD_MODE; use version/create/auto." >&2
    exit 1
    ;;
esac

if [[ "$upload_exists" == "1" ]]; then
  UPLOAD_CMD=(
    kaggle datasets version
    -p "$ARTIFACT_OUT"
    -m "$VERSION_MESSAGE"
    -r "$DIR_MODE"
  )
  if [[ "${BIOHUB_KAGGLE_DELETE_OLD_VERSIONS:-0}" == "1" ]]; then
    UPLOAD_CMD+=(--delete-old-versions)
  fi
else
  UPLOAD_CMD=(
    kaggle datasets create
    -p "$ARTIFACT_OUT"
    -r "$DIR_MODE"
  )
  if [[ "${BIOHUB_KAGGLE_PUBLIC:-0}" == "1" ]]; then
    UPLOAD_CMD+=(--public)
  fi
fi

if [[ "${BIOHUB_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run only; not uploading."
  printf 'Would run:'
  printf ' %q' "${UPLOAD_CMD[@]}"
  printf '\n'
else
  "${UPLOAD_CMD[@]}"
fi

echo
echo "Done."
echo "Artifact directory: $PROJECT_ROOT/$ARTIFACT_OUT"
echo "Artifact zip:       $PROJECT_ROOT/${ARTIFACT_OUT}.zip"
echo "Kaggle dataset id:  $DATASET_ID"
