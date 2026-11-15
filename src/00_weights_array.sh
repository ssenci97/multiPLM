#!/bin/bash
#SBATCH -J mp_weights
#SBATCH -p THIN
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 02:00:00
#SBATCH --array=0-9%4
#SBATCH -o weights_%A_%a.log
###################################### dependencies
set -euo pipefail
command -v singularity >/dev/null || module load singularity
###################################### parameters
backends=(esm2_t6_8M esm2_t12_35M esm2_t30_150M esm2_t33_650M esm2_t36_3B esmc_300m esmc_600m esm3_sm_open_v1 prott5_xl_half prostt5)
MP_ROOT="$(cd "${MP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" && pwd -P)"
sif="$MP_ROOT/containers/multiplm.sif"
b="${backends[${SLURM_ARRAY_TASK_ID:?run through sbatch}]}"
T0=$SECONDS
log() { echo "[SLURM-INFO] $*"; }
peak() { local rel f; rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null || true); for f in "/sys/fs/cgroup$rel/memory.peak" /sys/fs/cgroup/memory.peak; do [ -r "$f" ] && { awk '{printf "%.2f GB", $1/1073741824}' "$f"; return; }; done; echo "NA (see sacct)"; }
###################################### weights
cd "$MP_ROOT"
[ -s "$sif" ] || { log "$sif missing: run containers/build.sh pull first"; exit 1; }
log "host $(hostname) | job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID} | backend $b"
singularity exec --cleanenv --bind "$MP_ROOT" --pwd "$MP_ROOT" "$sif" /opt/conda/envs/mp/bin/python src/weights.py --backend "$b" --models_dir data/models --checksums_dir configs/checksums
log "$(date '+%F %T') | elapsed $(date -u -d @$((SECONDS-T0)) +%H:%M:%S) | max RSS $(peak)"
