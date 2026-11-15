#!/bin/bash
###################################### dependencies
set -euo pipefail
###################################### parameters
script="$1"; shift
cd "$MP_ROOT"
source "$MP_ROOT/src/find_app.sh"; app="$(find_app)"
MP_OUT="$MP_ROOT/results/$MP_ANALYSIS"
T_START="$(date '+%F %T')"; S0=$SECONDS
nv=(); [ "${MP_GPU:-0}" = 1 ] && nv=(--nv)
binds="$MP_ROOT${MP_BINDS:+,$MP_BINDS}"
offline=1; [ "${MP_ONLINE:-0}" = 1 ] && offline=0
token_env=""
if [ "${MP_ONLINE:-0}" = 1 ] && [ -n "${MP_HF_TOKEN_FILE:-}" ] && [ -r "$MP_HF_TOKEN_FILE" ]; then binds+=",$MP_HF_TOKEN_FILE:/run/mp_hf_token:ro"; token_env=" HF_TOKEN_PATH=/run/mp_hf_token"; fi
###################################### run
prefix="set -euo pipefail; export PATH=/opt/conda/envs/mp/bin:\$PATH ENVS=/opt/envs"
prefix+=" MP_ROOT=$(printf %q "$MP_ROOT") MP_OUT=$(printf %q "$MP_OUT") MP_IMAGE=$(printf %q "$MP_IMAGE") MP_GPU=${MP_GPU:-0} SLURM_JOB_ID=$(printf %q "${SLURM_JOB_ID:-local}")"
prefix+=" HF_HUB_OFFLINE=$offline TRANSFORMERS_OFFLINE=$offline HF_HOME=$(printf %q "$MP_ROOT/data/models/.hf_home") TORCH_HOME=$(printf %q "$MP_ROOT/data/models/.torch_home")$token_env"
prefix+='; export HOME="$MP_ROOT" PYTHONNOUSERSITE=1 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl TOKENIZERS_PARALLELISM=false; cd "$MP_ROOT"; exec bash "$0" "$@"'
echo "[SLURM-INFO] host $(hostname) | job ${SLURM_JOB_ID:-local} | image $MP_IMAGE | ${nv[*]:-cpu} | $script $*"
rc=0; "$app" exec --cleanenv "${nv[@]}" --bind "$binds" "$MP_IMAGE" bash -c "$prefix" "$MP_ROOT/$script" "$@" || rc=$?
###################################### verify outputs
if [ "$rc" -eq 0 ]; then
  IFS=: read -ra outs <<< "${MP_OUTPUTS:-}"
  for f in "${outs[@]}"; do
    ok=0; [ -s "$f" ] && ok=1
    [ "$ok" = 1 ] && [[ "$f" == *.json ]] && ! python3 -I -c 'import json, sys; json.load(open(sys.argv[1]))' "$f" 2>/dev/null && ok=0
    [ "$ok" = 1 ] || { echo "[SLURM-INFO] output check failed: $f"; rc=1; break; }
  done
  [ "$rc" -eq 0 ] && echo "[SLURM-INFO] outputs verified: ${#outs[@]}"
fi
###################################### performance metadata
if [ -n "${MP_META_TARGET:-}" ]; then
  MP_OUT="$MP_OUT" python3 -I "$MP_ROOT/src/sample_meta.py" --perf "$MP_META_TARGET" "$MP_META_STEP" \
    elapsed_s=$((SECONDS-S0)) exit="$rc" started="$T_START" ended="$(date '+%F %T')" \
    && echo "[SLURM-INFO] perf metadata: meta/$MP_META_TARGET.json steps.$MP_META_STEP.perf" \
    || echo "[SLURM-INFO] !! perf metadata not written"
fi
exit "$rc"
