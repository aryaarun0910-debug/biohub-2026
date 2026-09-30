#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${BIOHUB_PROJECT_ROOT:-$(cd "$SIDECAR_DIR/.." && pwd)}"
CONFIG_FILE="${BIOHUB_RUNPOD_CONFIG:-$SIDECAR_DIR/configs/cloud_50epoch.env}"

cd "$PROJECT_ROOT"

mkdir -p logs weights
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${BIOHUB_FULL_FRAME_LOG:-$PROJECT_ROOT/logs/full_frame_center_${TS}.log}"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "== Biohub full-frame center detector training =="
echo "time:    $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "project: $PROJECT_ROOT"
echo "config:  $CONFIG_FILE"
echo "log:     $RUN_LOG"
echo

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  source "$CONFIG_FILE"
  set +a
fi

"$SIDECAR_DIR/scripts/bootstrap_project_payload.sh"

VENV_DIR="${BIOHUB_VENV_DIR:-$PROJECT_ROOT/.venv-biohub-gpu}"
PYTHON_BOOTSTRAP="${BIOHUB_BOOTSTRAP_PYTHON:-python3}"
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"

if [[ -e "$VENV_DIR" && ! -f "$ACTIVATE_SCRIPT" ]]; then
  echo "Existing virtualenv is incomplete; recreating: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv --system-site-packages "$VENV_DIR"
fi
if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
  echo "Virtualenv activation script was not created: $ACTIVATE_SCRIPT" >&2
  exit 1
fi
source "$ACTIVATE_SCRIPT"
python -m pip install --upgrade pip setuptools wheel

REQ_TMP="$(mktemp)"
REQ_SOURCE="$PROJECT_ROOT/requirements-unet-ilp.txt"
if [[ -f "$REQ_SOURCE" ]]; then
  python - "$REQ_SOURCE" "$REQ_TMP" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip = {"torch"}
lines = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name = line.split("==")[0].split(">=")[0].split("<")[0].strip().lower()
    if name in skip:
        continue
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
print(dst.read_text())
PY
else
  cat > "$REQ_TMP" <<'REQ'
numpy>=2
pandas>=2
polars>=1.36
zarr>=3.0.10,<4
numcodecs>=0.13
blosc2
scikit-image>=0.24
tqdm
tracksdata
geff>=1.1.3.1.1
pyarrow
REQ
fi
python -m pip install -r "$REQ_TMP"
rm -f "$REQ_TMP"

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not visible to PyTorch.")
PY

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PROJECT_ROOT/$1" ;;
  esac
}

DATA_DIR_DEFAULT="data/dense_channel"
if [[ -n "${BIOHUB_SCRATCH_ROOT:-}" ]]; then
  DATA_DIR_DEFAULT="$(resolve_path "$BIOHUB_SCRATCH_ROOT")/data/dense_channel"
fi
DATA_DIR="$(resolve_path "${BIOHUB_DATA_DIR:-$DATA_DIR_DEFAULT}")"
METHOD="${BIOHUB_FULL_FRAME_METHOD:-full_frame_center_v1}"
OUTPUT_DIR="$(resolve_path "${BIOHUB_FULL_FRAME_OUTPUT_DIR:-weights/$METHOD}")"
EPOCHS="${BIOHUB_FULL_FRAME_EPOCHS:-50}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Training data directory does not exist: $DATA_DIR" >&2
  echo "Prepare data first with run_main_training_oneshot.sh, or set BIOHUB_DATA_DIR." >&2
  exit 1
fi
if ! find "$DATA_DIR" -maxdepth 1 -name '*.zarr' -print -quit | grep -q .; then
  echo "No .zarr samples found in $DATA_DIR" >&2
  exit 1
fi
if ! find "$DATA_DIR" -maxdepth 1 -name '*.geff' -print -quit | grep -q .; then
  echo "No .geff labels found in $DATA_DIR" >&2
  exit 1
fi

case "${BIOHUB_RUN_MODE:-}" in
  continue)
    if [[ ! -f "$OUTPUT_DIR/checkpoint_last.pt" ]]; then
      echo "BIOHUB_RUN_MODE=continue requires:" >&2
      echo "  $OUTPUT_DIR/checkpoint_last.pt" >&2
      exit 1
    fi
    RESUME_ARGS=(--resume)
    ;;
  fresh)
    if [[ "${BIOHUB_ALLOW_FRESH_START:-0}" != "1" ]]; then
      echo "BIOHUB_RUN_MODE=fresh is blocked by default." >&2
      echo "Set BIOHUB_ALLOW_FRESH_START=1 only when intentionally starting a new detector." >&2
      exit 1
    fi
    if [[ -e "$OUTPUT_DIR" ]]; then
      echo "Fresh full-frame run: removing existing output directory:"
      echo "  $OUTPUT_DIR"
      rm -rf "$OUTPUT_DIR"
    fi
    RESUME_ARGS=(--overwrite)
    ;;
  "")
    echo "BIOHUB_RUN_MODE is required: use continue or fresh." >&2
    exit 1
    ;;
  *)
    echo "Unsupported BIOHUB_RUN_MODE=$BIOHUB_RUN_MODE; use continue or fresh." >&2
    exit 1
    ;;
esac

echo
echo "== Effective full-frame training config =="
echo "data_dir:   $DATA_DIR"
echo "output_dir: $OUTPUT_DIR"
echo "epochs:     $EPOCHS"
env | sort | grep '^BIOHUB_' || true
echo

SNAPSHOT_INTERVAL="${BIOHUB_FULL_FRAME_SNAPSHOT_INTERVAL:-100}"
SNAPSHOT_FINAL="${BIOHUB_FULL_FRAME_SNAPSHOT_FINAL:-1}"
if [[ "$SNAPSHOT_INTERVAL" != "0" || "$SNAPSHOT_FINAL" == "1" ]]; then
  if ! python scripts/train_full_frame_center_detector.py --help 2>&1 | grep -q -- "--snapshot-interval"; then
    echo "The local scripts/train_full_frame_center_detector.py does not support snapshot arguments." >&2
    echo "This usually means the RunPod project has an old training script but a new wrapper." >&2
    echo "Rebuild/upload the cloud_training payload and run bootstrap_project_payload.sh." >&2
    exit 1
  fi
fi

TRAIN_ARGS=(
  scripts/train_full_frame_center_detector.py
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "${BIOHUB_FULL_FRAME_BATCH_SIZE:-8}"
  --num-workers "${BIOHUB_FULL_FRAME_NUM_WORKERS:-4}"
  --frames-per-movie "${BIOHUB_FULL_FRAME_FRAMES_PER_MOVIE:-0}"
  --pool-factor "${BIOHUB_FULL_FRAME_POOL_FACTOR:-4}"
  --base-channels "${BIOHUB_FULL_FRAME_BASE_CHANNELS:-24}"
  --gauss-sigma "${BIOHUB_FULL_FRAME_GAUSS_SIGMA:-1.0}"
  --pos-thresh "${BIOHUB_FULL_FRAME_POS_THRESH:-0.05}"
  --bg-quantile "${BIOHUB_FULL_FRAME_BG_QUANTILE:-0.40}"
  --w-pos "${BIOHUB_FULL_FRAME_W_POS:-12.0}"
  --w-bg "${BIOHUB_FULL_FRAME_W_BG:-1.0}"
  --w-ignore "${BIOHUB_FULL_FRAME_W_IGNORE:-0.05}"
  --norm-lo-pct "${BIOHUB_FULL_FRAME_NORM_LO_PCT:-50.0}"
  --norm-hi-pct "${BIOHUB_FULL_FRAME_NORM_HI_PCT:-99.5}"
  --norm-clip-lo "${BIOHUB_FULL_FRAME_NORM_CLIP_LO:--0.5}"
  --norm-clip-hi "${BIOHUB_FULL_FRAME_NORM_CLIP_HI:-6.0}"
  --learning-rate "${BIOHUB_FULL_FRAME_LR:-0.001}"
  --weight-decay "${BIOHUB_FULL_FRAME_WEIGHT_DECAY:-0.0}"
  --val-fraction "${BIOHUB_FULL_FRAME_VAL_FRACTION:-0.10}"
  --progress-interval "${BIOHUB_FULL_FRAME_PROGRESS_INTERVAL:-50}"
  --val-batches "${BIOHUB_FULL_FRAME_VAL_BATCHES:-24}"
  --gate-eval-frames "${BIOHUB_FULL_FRAME_GATE_EVAL_FRAMES:-240}"
  --gate-thresholds "${BIOHUB_FULL_FRAME_GATE_THRESHOLDS:-0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80}"
  --gate-peak-min-distance "${BIOHUB_FULL_FRAME_GATE_PEAK_MIN_DISTANCE:-1}"
  --gate-match-radius-um "${BIOHUB_FULL_FRAME_GATE_MATCH_RADIUS_UM:-7.0}"
  --gate-peak-sample-limit "${BIOHUB_FULL_FRAME_GATE_PEAK_SAMPLE_LIMIT:-50000}"
  --snapshot-interval "$SNAPSHOT_INTERVAL"
  --snapshot-prefix "${BIOHUB_FULL_FRAME_SNAPSHOT_PREFIX:-ep}"
  "${RESUME_ARGS[@]}"
)
if [[ -n "${BIOHUB_FULL_FRAME_SNAPSHOT_DIR:-}" ]]; then
  TRAIN_ARGS+=(--snapshot-dir "$BIOHUB_FULL_FRAME_SNAPSHOT_DIR")
fi
if [[ -n "${BIOHUB_FULL_FRAME_SNAPSHOT_GATE_EVAL_FRAMES:-}" ]]; then
  TRAIN_ARGS+=(--snapshot-gate-eval-frames "$BIOHUB_FULL_FRAME_SNAPSHOT_GATE_EVAL_FRAMES")
fi
if [[ "$SNAPSHOT_FINAL" == "1" ]]; then
  TRAIN_ARGS+=(--snapshot-final)
fi
if [[ -n "${BIOHUB_FULL_FRAME_MOVIE_LIMIT:-}" ]]; then
  TRAIN_ARGS+=(--movie-limit "$BIOHUB_FULL_FRAME_MOVIE_LIMIT")
fi
if [[ -n "${BIOHUB_FULL_FRAME_GRAD_CLIP_NORM:-}" ]]; then
  TRAIN_ARGS+=(--grad-clip-norm "$BIOHUB_FULL_FRAME_GRAD_CLIP_NORM")
fi
if [[ -n "${BIOHUB_FULL_FRAME_BRIGHTNESS_JITTER:-}" ]]; then
  TRAIN_ARGS+=(--brightness-jitter "$BIOHUB_FULL_FRAME_BRIGHTNESS_JITTER")
fi
if [[ "${BIOHUB_FULL_FRAME_NO_RANDOM_FLIP:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--no-random-flip)
fi

python "${TRAIN_ARGS[@]}"

echo
echo "== Full-frame detector artifacts =="
for artifact in \
  "$OUTPUT_DIR"/best.pt \
  "$OUTPUT_DIR"/checkpoint_last.pt \
  "$OUTPUT_DIR"/config.json \
  "$OUTPUT_DIR"/history.csv \
  "$OUTPUT_DIR"/split_manifest.json \
  "$OUTPUT_DIR"/gate_summary.json \
  "$OUTPUT_DIR"/gate_threshold_metrics.csv \
  "$OUTPUT_DIR"/gate_frame_metrics.csv \
  "$OUTPUT_DIR"/gate_peak_samples.csv; do
  if [[ -e "$artifact" ]]; then
    ls -lh "$artifact"
  fi
done

SNAPSHOT_DIR="${BIOHUB_FULL_FRAME_SNAPSHOT_DIR:-$(dirname "$OUTPUT_DIR")/$(basename "$OUTPUT_DIR")_snapshots}"
if [[ -d "$SNAPSHOT_DIR" ]]; then
  echo
  echo "== Full-frame interval snapshots =="
  find "$SNAPSHOT_DIR" -maxdepth 2 -type f \( \
    -name best.pt -o \
    -name checkpoint_last.pt -o \
    -name SNAPSHOT_MANIFEST.json -o \
    -name gate_summary.json -o \
    -name gate_threshold_metrics.csv \
  \) -print | sort | sed -n '1,200p'
fi

if [[ "${BIOHUB_PACKAGE_AFTER_FULL_FRAME:-0}" == "1" ]]; then
  echo
  echo "== Packaging support pack with full-frame detector =="
  export BIOHUB_FULL_FRAME_CENTER_SOURCE="$OUTPUT_DIR"
  bash "$SIDECAR_DIR/scripts/package_and_upload_support_pack_version.sh"
fi

if [[ "${BIOHUB_PACKAGE_FULL_FRAME_SEPARATE:-0}" == "1" ]]; then
  echo
  echo "== Packaging standalone full-frame center detector pack =="
  export BIOHUB_FULL_FRAME_CENTER_SOURCE="$OUTPUT_DIR"
  export BIOHUB_FULL_FRAME_METHOD="$METHOD"
  bash "$SIDECAR_DIR/scripts/package_and_upload_full_frame_center_pack.sh"
fi

echo
echo "Done."
echo "Run log: $RUN_LOG"
