#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### parameters
sample="$1"; backend="$2"; shards="$3"; root="$4"; out="$5"
###################################### merge
"$(mp_python mp)" "$MP_ROOT/src/merge.py" --sample "$sample" --shards "$shards" --shard_root "$root" --out_dir "$out" --yaml-in "$MP_PARAMS"
mp_meta "$sample" "04_merge_$backend" provenance="@$out/$sample.json"
mp_footer
