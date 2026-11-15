#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### parameters
sample="$1"; backend="$2"; env="$3"; fasta="$4"; shard="$5"; shards="$6"; chunks="$7"; done_marker="$8"
###################################### embed
"$(mp_python "$env")" "$MP_ROOT/src/embed.py" --backend "$backend" --fasta "$fasta" --shard "$shard" --shards "$shards" --chunk_dir "$chunks" --done "$done_marker" --models_dir "$MP_ROOT/data/models" --yaml-in "$MP_PARAMS"
mp_meta "$sample" "03_embed_${backend}_shard$shard" embed="@$done_marker"
mp_footer
