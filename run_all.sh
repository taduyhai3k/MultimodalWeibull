#!/usr/bin/env bash
set -Eeuo pipefail
AUTO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$AUTO_DIR/pipeline.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing config: $ENV_FILE. Copy pipeline.env.example to pipeline.env and edit it."
    exit 2
fi
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${PROJECT_ROOT:?Set PROJECT_ROOT in pipeline.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COHORTS="${COHORTS:-coadread}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXPERIMENT="${EXPERIMENT:-factory_v1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
ENABLE_DRIVE_BACKUP="${ENABLE_DRIVE_BACKUP:-0}"
BACKUP_INTERVAL="${BACKUP_INTERVAL:-300}"
BACKUP_TIMEOUT="${BACKUP_TIMEOUT:-900}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/$EXPERIMENT}"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
LOG_DIR="$OUTPUT_DIR/pipeline_logs"
STATE_DIR="$OUTPUT_DIR/pipeline_state"
mkdir -p -- "$LOG_DIR" "$STATE_DIR"
command -v flock >/dev/null || { echo 'flock is required (util-linux).'; exit 2; }
exec 9>"$OUTPUT_DIR/.pipeline.lock"
flock -n 9 || { echo "A pipeline is already using $OUTPUT_DIR"; exit 2; }
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
say() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; }
read -r -a cohort_list <<< "$COHORTS"
read -r -a fold_list <<< "$FOLDS"
(( ${#cohort_list[@]} && ${#fold_list[@]} )) || { echo 'Empty COHORTS/FOLDS'; exit 2; }
for fold in "${fold_list[@]}"; do [[ "$fold" =~ ^[0-9]+$ ]] || { echo 'Fold must be an integer'; exit 2; }; done
for flag in INSTALL_DEPS DOWNLOAD_DATA RUN_PREFLIGHT REQUIRE_CUDA START_TENSORBOARD ENABLE_DRIVE_BACKUP; do
    [[ "${!flag}" =~ ^[01]$ ]] || { echo "$flag must be 0 or 1"; exit 2; }
done
for num in BACKUP_INTERVAL BACKUP_TIMEOUT TENSORBOARD_PORT; do
    [[ "${!num}" =~ ^[0-9]+$ ]] && (( ${!num} > 0 )) || { echo "Invalid $num"; exit 2; }
done
backup_pid="";tb_pid="";train_pid="";backup_ready=0
REMOTE_DEST=""
backup_once() {
    timeout --signal=TERM --kill-after=15 "$BACKUP_TIMEOUT" \
      rclone copy "$OUTPUT_DIR" "$REMOTE_DEST" \
      --drive-root-folder-id "$DRIVE_FOLDER_ID" \
      --exclude '*.tmp' --exclude '*.lock' \
      --exclude 'pipeline_logs/**' --exclude 'pipeline_state/**' \
      --transfers 2 --retries 3 --stats 30s
}
cleanup() {
    rc=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "$train_pid" ]]; then
        kill -TERM "$train_pid" 2>/dev/null
        wait "$train_pid" 2>/dev/null
    fi
    if [[ -n "$backup_pid" ]]; then
        kill -TERM "$backup_pid" 2>/dev/null
        wait "$backup_pid" 2>/dev/null
    fi
    if [[ -n "$tb_pid" ]]; then
        kill -TERM "$tb_pid" 2>/dev/null
        wait "$tb_pid" 2>/dev/null
    fi
    if (( backup_ready )); then
        say 'Final Drive backup (bounded by BACKUP_TIMEOUT)'
        if ! backup_once; then
            say 'Final backup FAILED; local results remain. Read pipeline.log.'
            (( rc == 0 )) && rc=30
        fi
    fi
    say "Pipeline exit code: $rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Run bundled trainer, while importing model/block/dataset from PROJECT_ROOT.
cd -- "$AUTO_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
for needed in model.py dataset/DatasetFactory.py dataset/DatasetSurvival.py; do
    [[ -f "$PROJECT_ROOT/$needed" ]] || { echo "Missing source: $PROJECT_ROOT/$needed"; exit 2; }
done
say "Project: $PROJECT_ROOT; output: $OUTPUT_DIR"
say "Cohorts: $COHORTS; folds: $FOLDS"
"$PYTHON_BIN" -c 'import torch; print("PyTorch",torch.__version__,"CUDA available:",torch.cuda.is_available())'
if [[ "$REQUIRE_CUDA" == 1 ]]; then
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "PyTorch CUDA unavailable; choose/fix GPU environment first"; print(torch.cuda.get_device_name(0))'
fi
if [[ "$INSTALL_DEPS" == 1 ]]; then
    say 'Install helper dependencies in selected Python; do not replace PyTorch'
    "$PYTHON_BIN" -m pip install -r "$AUTO_DIR/factory_trainer/requirements.txt" 'gdown==5.2.0'
fi
# Validate auth before spending time downloading/training.
if [[ "$ENABLE_DRIVE_BACKUP" == 1 ]]; then
    : "${DRIVE_REMOTE:?Set DRIVE_REMOTE}" "${DRIVE_FOLDER_ID:?Set DRIVE_FOLDER_ID}"
    command -v rclone >/dev/null || { echo 'Install/configure rclone before enabling Drive backup.'; exit 2; }
    command -v timeout >/dev/null || { echo 'timeout (coreutils) required'; exit 2; }
    REMOTE_DEST="${DRIVE_REMOTE}:$EXPERIMENT"
    say 'Validate/create result folder on Drive'
    timeout --signal=TERM --kill-after=15 90 rclone mkdir "$REMOTE_DEST" --drive-root-folder-id "$DRIVE_FOLDER_ID"
    backup_ready=1
fi
# Detect source contract before any large download.
"$PYTHON_BIN" - <<'PY'
import inspect
from model import MultiModelWeibull
from block.loss import PIBDWeibullLoss
from dataset.DatasetFactory import SurvivalDatasetFactory
assert 'return_aux' in inspect.signature(MultiModelWeibull.forward).parameters, 'Import your working notebook model version: forward(return_aux=True) required'
assert hasattr(MultiModelWeibull,'predict'), 'Model needs label-free predict()'
assert hasattr(PIBDWeibullLoss,'club_parameters'), 'Import your working PIBDWeibullLoss'
print('Model:',inspect.getfile(MultiModelWeibull))
print('Factory:',inspect.getfile(SurvivalDatasetFactory))
PY
manifest="${DRIVE_MANIFEST:-$AUTO_DIR/drive_setup/drive_manifest.json}"
base_config="${TRAINER_CONFIG:-$AUTO_DIR/factory_trainer/config.json}"
if [[ "$DOWNLOAD_DATA" == 1 ]]; then
    say 'Download Google Drive ZIPs and extract .pt features'
    "$PYTHON_BIN" "$AUTO_DIR/drive_setup/prepare_data.py" \
      --root "$PROJECT_ROOT" --manifest "$manifest" \
      --cohorts "${cohort_list[@]}" --folds "${fold_list[@]}"
fi
runtime_config="$STATE_DIR/runtime_config.json"
say 'Resolve WSI parent directory for the existing DatasetFactory'
"$PYTHON_BIN" "$AUTO_DIR/make_runtime.py" \
  --root "$PROJECT_ROOT" --config "$base_config" --output "$runtime_config" \
  --cohorts "${cohort_list[@]}" --folds "${fold_list[@]}" --require-cuda "$REQUIRE_CUDA"
if [[ "$START_TENSORBOARD" == 1 ]]; then
    say "Start TensorBoard on 127.0.0.1:$TENSORBOARD_PORT"
    "$PYTHON_BIN" -m tensorboard.main --logdir "$OUTPUT_DIR" --host 127.0.0.1 --port "$TENSORBOARD_PORT" \
      >"$LOG_DIR/tensorboard.log" 2>&1 9>&- &
    tb_pid=$!
    sleep 2
    kill -0 "$tb_pid" 2>/dev/null || { echo "TensorBoard failed; see $LOG_DIR/tensorboard.log"; exit 2; }
fi
if (( backup_ready )); then
    bash "$AUTO_DIR/backup_worker.sh" "$OUTPUT_DIR" "$REMOTE_DEST" "$DRIVE_FOLDER_ID" "$BACKUP_INTERVAL" "$BACKUP_TIMEOUT" \
      >"$LOG_DIR/drive_backup.log" 2>&1 9>&- &
    backup_pid=$!
fi
for cohort in "${cohort_list[@]}"; do
    for fold in "${fold_list[@]}"; do
        fold_dir="$OUTPUT_DIR/$cohort/fold_$fold"
        marker="$STATE_DIR/${cohort}_fold_${fold}.done"
        if [[ -f "$marker" && -f "$fold_dir/best.pt" && -f "$fold_dir/val_best_metrics.json" ]]; then
            say "Already completed: $cohort fold $fold; skip"
            continue
        fi
        if [[ "$RUN_PREFLIGHT" == 1 && ! -f "$fold_dir/last.pt" ]]; then
            say "Preflight: $cohort fold $fold"
            "$PYTHON_BIN" -m factory_trainer.run --config "$runtime_config" --root "$PROJECT_ROOT" \
              --output "$OUTPUT_DIR/preflight" --cohorts "$cohort" --folds "$fold" --preflight 9>&- &
            train_pid=$!
            wait "$train_pid"
            train_pid=""
        fi
        resume_args=()
        if [[ -f "$fold_dir/last.pt" ]]; then resume_args=(--resume); fi
        say "Train: $cohort fold $fold; resume=${#resume_args[@]}"
        "$PYTHON_BIN" -m factory_trainer.run --config "$runtime_config" --root "$PROJECT_ROOT" \
          --output "$OUTPUT_DIR" --cohorts "$cohort" --folds "$fold" "${resume_args[@]}" 9>&- &
        train_pid=$!
        wait "$train_pid"
        train_pid=""
        date -Is >"$marker"
        say "Completed $cohort fold $fold"
    done
done
say 'All selected folds completed. Final backup follows if configured.'
