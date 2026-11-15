#!/bin/bash
###################################### dependencies
set -euo pipefail
MP_ROOT="${MP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ENVS="${ENVS:-/opt/envs}"
###################################### constants
export SEED=42
MP_T0=$SECONDS
MP_PARAMS="${MP_OUT:-}/inputs/params.yaml"
###################################### container guard
if [ -z "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ] && [ ! -d /.singularity.d ]; then echo "[SLURM-INFO] refusing to run outside the container image (run through python3 src/pipeline.py <analysis>)"; exit 1; fi
[ -x "$ENVS/esm-evo/bin/python" ] && [ -x "$ENVS/esm-fair/bin/python" ] || { echo "[SLURM-INFO] $ENVS lacks the esm-evo/esm-fair venvs: wrong image"; exit 1; }
[ -n "${MP_OUT:-}" ] || { echo "[SLURM-INFO] MP_OUT (results/<analysis_name>) is not set: run through python3 src/pipeline.py <analysis> or the console"; exit 1; }
###################################### helpers
mp_log() { echo "[SLURM-INFO] $*"; }
mp_python() { case "$1" in mp) echo /opt/conda/envs/mp/bin/python ;; esm-evo|esm-fair) echo "$ENVS/$1/bin/python" ;; *) mp_log "unknown env $1"; exit 1 ;; esac; }
mp_meta() { local id="$1" step="$2"; shift 2; python3 -I "$MP_ROOT/src/sample_meta.py" "$id" "$step" "$@"; }
mp_peak() { local rel f; rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null || true); for f in "/sys/fs/cgroup$rel/memory.peak" /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do [ -r "$f" ] && { awk '{printf "%.2f GB", $1/1073741824}' "$f"; return; }; done; echo "NA (see sacct)"; }
mp_footer() { mp_log "$(date '+%F %T') | elapsed $(date -u -d @$((SECONDS-MP_T0)) +%H:%M:%S) | max RSS $(mp_peak)"; }
mp_log "host $(hostname) | job ${SLURM_JOB_ID:-local} | image ${MP_IMAGE:-none} | gpu ${MP_GPU:-0}"
