#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### parameters
sample="$1"; source="$2"; fasta="$3"; table="$4"; info="$5"
###################################### prepare
"$(mp_python mp)" "$MP_ROOT/src/prepare.py" --sample "$sample" --source "$source" --out_fasta "$fasta" --out_tsv "$table" --out_json "$info" --yaml-in "$MP_PARAMS"
mp_meta "$sample" 01_prepare prepare="@$info"
mp_footer
