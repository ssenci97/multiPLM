#!/bin/bash
###################################### dependencies
set -euo pipefail
MP_ROOT="${MP_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$MP_ROOT"
source containers/pins.env
###################################### parameters
registry="${MP_REGISTRY:-ghcr.io}"
owner="${MP_OWNER:-$( (git config --get remote.origin.url 2>/dev/null || true) | sed -n 's#.*[/:]\([^/]*\)/[^/]*$#\1#p')}"
owner="${owner:-local}"; owner="${owner,,}"
tag="${MP_TAG:-1.0}"
engine="${MP_ENGINE:-}"
need_gb="${MP_PULL_GB:-20}"
T0=$SECONDS
image="$registry/$owner/multiplm:$tag"
sif=containers/multiplm.sif
digests=containers/images.env
locks=(containers/mp.lock.txt containers/mp.pip.lock.txt containers/esm-evo.lock.txt containers/dadapy.lock.txt)
###################################### helpers
log() { echo "[SLURM-INFO] $*"; }
footer() { log "$(date '+%F %T') | elapsed $(date -u -d @$((SECONDS-T0)) +%H:%M:%S) | max RSS NA (no job cgroup)"; }
col_of() { awk -F'\t' -v n="$1" -v c="$2" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $1==n{print $h[c]}' configs/tools.tsv; }
pinned() { local c; c=$(col_of "$1" commit); [[ "$c" =~ ^[0-9a-f]{40}$ ]] || { log "$1 is not pinned to a full commit sha in configs/tools.tsv: '$c'"; exit 1; }; echo "$c"; }
###################################### engine
MODE="${1:-build}"; GPU_TEST="${2:-}"
case "$MODE" in orfeo|pull|test) host_only=1 ;; *) host_only=0 ;; esac
if [ "$host_only" = 0 ] && [ -z "$engine" ]; then
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then engine=docker; else engine=podman; fi
fi
[ "$host_only" = 1 ] || command -v "$engine" >/dev/null || { log "$engine not found"; exit 1; }
engine_flags=()
if [ "$host_only" = 0 ] && [ "$engine" = podman ]; then
  engine_flags=(--cgroup-manager=cgroupfs --events-backend=file)
  if [ ! -d "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" ]; then
    export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/podman-$(id -u)"; mkdir -p -m 700 "$XDG_RUNTIME_DIR"
    log "no /run/user/$(id -u); using XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
  fi
fi
[[ "$owner" =~ ^[a-z0-9._-]+$ ]] || { log "MP_OWNER must be lowercase: '$owner'"; exit 1; }
[ "$host_only" = 1 ] || { log "param registry=$registry"; log "param owner=$owner"; log "param tag=$tag"; log "param engine=$engine ${engine_flags[*]}"; }
###################################### build
build_image() {
  log "building $image (base $BASE_TAG_FOR_HUMANS, esm-evo @ $(pinned esm-evo | cut -c1-7), fair-esm @ $(pinned fair-esm | cut -c1-7), locks $([ -s "${locks[0]}" ] && echo yes || echo "no -- solved from envs/"))"
  "$engine" "${engine_flags[@]}" build -f containers/Dockerfile -t "$image" --build-arg "BASE_REF=$BASE_REF" \
    --build-arg "ESM_EVO_REPO=$(col_of esm-evo repo)" --build-arg "ESM_EVO_COMMIT=$(pinned esm-evo)" \
    --build-arg "FAIR_ESM_REPO=$(col_of fair-esm repo)" --build-arg "FAIR_ESM_COMMIT=$(pinned fair-esm)" .
}
write_lock() {
  log "exporting ${locks[*]} from $image"
  "$engine" "${engine_flags[@]}" run --rm "$image" conda list -n mp --explicit --md5 > "${locks[0]}.tmp"
  grep -q "^https\?://" "${locks[0]}.tmp" || { log "conda export produced no package urls"; rm -f "${locks[0]}.tmp"; exit 1; }
  { echo "--extra-index-url https://download.pytorch.org/whl/cu121"; "$engine" "${engine_flags[@]}" run --rm "$image" conda list -n mp | awk '$4=="pypi"{print $1"=="$2}'; } > "${locks[1]}.tmp"
  "$engine" "${engine_flags[@]}" run --rm "$image" /opt/envs/esm-evo/bin/pip freeze --local | grep -v -e '^esm' -e '@' > "${locks[2]}.tmp"
  "$engine" "${engine_flags[@]}" run --rm "$image" /opt/envs/dadapy/bin/pip freeze | grep -v -e '^dadapy' -e '@' > "${locks[3]}.tmp"
  grep -q '^torch==' "${locks[1]}.tmp" && [ -s "${locks[2]}.tmp" ] && grep -q '^jax==' "${locks[3]}.tmp" || { log "pip export incomplete"; rm -f "${locks[@]/%/.tmp}"; exit 1; }
  for f in "${locks[@]}"; do mv "$f.tmp" "$f"; done
  log "$(grep -c "^https\?://" "${locks[0]}") conda, $(grep -c == "${locks[1]}") base pip, $(grep -c == "${locks[2]}") esm-evo pip, $(grep -c == "${locks[3]}") dadapy pip packages pinned; rebuild to use them"
}
###################################### registry
do_login() {
  [ -n "${MP_GITHUB_TOKEN:-}" ] || { log "export MP_GITHUB_TOKEN=<github personal access token with write:packages>"; exit 1; }
  log "logging in to $registry as $owner"
  printf '%s' "$MP_GITHUB_TOKEN" | "$engine" "${engine_flags[@]}" login "$registry" -u "$owner" --password-stdin
}
push_image() {
  [ "$owner" = local ] && { log "set MP_OWNER to your github namespace before pushing"; exit 1; }
  log "pushing $image"
  "$engine" "${engine_flags[@]}" push "$image"
  local d; d="$("$engine" "${engine_flags[@]}" inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null | sed -n 's/.*@//p')"
  [ -n "$d" ] || { log "no digest for $image"; exit 1; }
  printf 'MP_IMAGE_REF=%s@%s\n' "${image%:*}" "$d" > "$digests"
  log "wrote $digests: ${image%:*}@$d -- commit it"
}
###################################### tests
tests_start() {
  tests_out=""; tests_pid=""
  [ "${MP_SKIP_TESTS:-0}" = 1 ] && { log "tests skipped (MP_SKIP_TESTS=1)"; return; }
  if ! python3 -c "import pytest, yaml" 2>/dev/null; then log "!! unit tests skipped: host python3 lacks pytest/pyyaml (python3 -m pip install --user pytest pyyaml)"; return; fi
  tests_out="$(mktemp "${TMPDIR:-/tmp}/mp_tests.XXXXXX")"
  log "unit tests started in the background (host python3, no container needed)"
  (cd "$MP_ROOT" && python3 -m pytest -q -p no:cacheprovider tests > "$tests_out" 2>&1) & tests_pid=$!
}
tests_wait() {
  [ -n "$tests_pid" ] || return 0
  local rc=0; wait "$tests_pid" || rc=$?
  sed 's/^/[SLURM-INFO]   /' "$tests_out"; rm -f "$tests_out"
  [ "$rc" -eq 0 ] && log "unit tests passed" || { log "!! unit tests FAILED (rc $rc)"; return 1; }
}
smoke() {
  local app="$1" nv=()
  [ "$GPU_TEST" = --gpu ] && nv=(--nv)
  log "container smoke test on $sif ${nv[*]}"
  "$app" exec --cleanenv "${nv[@]}" "$MP_ROOT/$sif" bash -c 'set -euo pipefail
    /opt/conda/envs/mp/bin/python -c "import torch, transformers, sentencepiece, h5py; print(\"mp torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpu\", torch.cuda.is_available())"
    /opt/envs/esm-evo/bin/python -c "import esm.pretrained; from esm.models.esmc import ESMC; print(\"esm-evo ok\")"
    /opt/envs/esm-fair/bin/python -c "import esm.pretrained; print(\"esm-fair\", esm.__version__)"
    /opt/conda/envs/mp/bin/mmseqs version | sed "s/^/mmseqs /"
    /opt/envs/dadapy/bin/python -c "import importlib.metadata as m; print(\"dadapy\", m.version(\"dadapy\"))"
    for b in git curl flock; do command -v "$b" >/dev/null || { echo "MISSING $b"; exit 1; }; done' \
    | sed 's/^/[SLURM-INFO]   /' || { log "!! container smoke test FAILED"; return 1; }
  if [ "$GPU_TEST" = --gpu ]; then
    "$app" exec --cleanenv --nv "$MP_ROOT/$sif" /opt/conda/envs/mp/bin/python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())" \
      | sed 's/^/[SLURM-INFO]   /' || { log "!! GPU check FAILED (run on a GPU node: srun -p GPU --gpus=1 ... build.sh test --gpu)"; return 1; }
  fi
  log "DADApy benchmarks of Facco et al. 2017 (about a minute)"
  "$app" exec --cleanenv --bind "$MP_ROOT" "$MP_ROOT/$sif" /opt/envs/dadapy/bin/python "$MP_ROOT/tests/dadapy_benchmarks.py" | sed 's/^/[SLURM-INFO]   /' || { log "!! DADApy benchmarks FAILED"; return 1; }
  log "container smoke test passed"
}
find_app() {
  local app="${MP_APPTAINER:-$(command -v singularity || command -v apptainer || true)}"
  [ -n "$app" ] || app="$(bash -lc 'module load singularity >/dev/null 2>&1; command -v singularity || command -v apptainer' 2>/dev/null || true)"
  echo "$app"
}
test_only() {
  local app; app="$(find_app)"; tests_start
  local bad=0
  if [ -s "$MP_ROOT/$sif" ] && [ -n "$app" ]; then smoke "$app" || bad=1; else log "!! no $sif or no singularity: container smoke test skipped"; fi
  tests_wait || bad=1
  [ "$bad" = 0 ] || exit 1
}
###################################### orfeo
pull() {
  local app ref avail; app="$(find_app)"
  [ -n "$app" ] || { log "no singularity/apptainer, even after 'module load singularity' -- are you on the cluster?"; exit 1; }
  log "using $app"; tests_start
  ref="$image"
  if [ -s "$digests" ]; then ref="$(sed -n 's/^MP_IMAGE_REF=//p' "$digests" | tail -1)"; else log "!! $digests missing: pulling by tag, not by digest"; fi
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$MP_ROOT/.singularity}" APPTAINER_CACHEDIR="${SINGULARITY_CACHEDIR:-$MP_ROOT/.singularity}"
  export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$SINGULARITY_CACHEDIR/tmp}"; export APPTAINER_TMPDIR="$SINGULARITY_TMPDIR"
  mkdir -p "$SINGULARITY_TMPDIR"
  avail=$(df -Pk "$MP_ROOT" | awk 'NR==2{print int($4/1048576)}')
  log "pulling $ref into $MP_ROOT/$sif (cache $SINGULARITY_CACHEDIR, ${avail} GB free here)"
  [ "$avail" -lt "$need_gb" ] && { log "!! only ${avail} GB free; the CUDA image needs about ${need_gb} GB (layer cache + sif). Free space or set MP_PULL_GB"; exit 1; }
  "$app" pull --force "$MP_ROOT/$sif" "docker://$ref" || { log "pull failed; if killed while creating squashfs, give it memory: srun -p THIN --mem=16G -c 4 --time=01:00:00 containers/build.sh pull"; exit 1; }
  [ -s "$MP_ROOT/$sif" ] || { log "pull reported success but $sif is missing or empty"; exit 1; }
  log "ok: $(du -h "$MP_ROOT/$sif" | cut -f1) at $MP_ROOT/$sif"
  local bad=0; smoke "$app" || bad=1; tests_wait || bad=1
  [ "$bad" = 0 ] || { log "!! image is in place but checks failed; rerun only the checks with: containers/build.sh test"; exit 1; }
  log "next: python3 src/pipeline.py <analysis> --dry-run, or the console: python3 src/gui_serve.py"
}
orfeo() {
  local ref="$image"
  [ -s "$digests" ] && ref="$(sed -n 's/^MP_IMAGE_REF=//p' "$digests" | tail -1)"
  log "on the cluster, from the repo root:"
  log "  srun -p THIN --mem=16G -c 4 --time=01:00:00 containers/build.sh pull"
  log "  srun -p GPU --gpus=1 -c 1 --mem=4G --time=00:15:00 containers/build.sh test --gpu"
  log "  manual pull: module load singularity; singularity pull --force $sif docker://$ref"
  log "  then: python3 src/pipeline.py <analysis> --dry-run, or the console: python3 src/gui_serve.py"
}
###################################### dispatch
case "$MODE" in
  build|all) build_image ;;
  pull)      pull ;;
  lock)      write_lock ;;
  login)     do_login ;;
  push)      push_image ;;
  ship)      do_login; push_image; orfeo ;;
  orfeo)     orfeo ;;
  test)      test_only ;;
  *) log "usage: build.sh [build|lock|login|push|ship|pull|test [--gpu]|orfeo]"; exit 1 ;;
esac
footer
