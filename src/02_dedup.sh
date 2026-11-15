#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### parameters
scope="$1"; out_dir="$2"; shift 2
###################################### dedup
"$(mp_python mp)" "$MP_ROOT/src/dedup.py" --scope "$scope" --out_dir "$out_dir" --threads "${SLURM_CPUS_PER_TASK:-1}" --fasta "$@" --yaml-in "$MP_PARAMS"
for f in "$@"; do s=$(basename "$f" .fasta); mp_meta "$s" 02_dedup dedup="@$out_dir/$s.dedup.json"; done
mp_footer
