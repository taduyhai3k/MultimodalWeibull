#!/usr/bin/env bash
set -u
# Internal worker. Arguments are paths/remote, passed without eval.
source_dir=$1
remote_dest=$2
folder_id=$3
interval=$4
timeout_sec=$5
transfer_pid=""
stop_worker() {
    trap - TERM INT
    if [[ -n "$transfer_pid" ]]; then
        kill -TERM "$transfer_pid" 2>/dev/null || true
        wait "$transfer_pid" 2>/dev/null || true
    fi
    exit 0
}
trap stop_worker TERM INT
while true; do
    printf '\n[%s] Copy results to Drive\n' "$(date -Is)"
    timeout --signal=TERM --kill-after=15 "$timeout_sec" \
      rclone copy "$source_dir" "$remote_dest" \
      --drive-root-folder-id "$folder_id" \
      --exclude '*.tmp' --exclude '*.lock' \
      --exclude 'pipeline_logs/**' --exclude 'pipeline_state/**' \
      --transfers 2 --retries 3 --stats 30s &
    transfer_pid=$!
    if ! wait "$transfer_pid"; then
        printf '[%s] Backup failed/timed out; next cycle will retry.\n' "$(date -Is)"
    fi
    transfer_pid=""
    sleep "$interval" &
    transfer_pid=$!
    wait "$transfer_pid" || true
    transfer_pid=""
done
