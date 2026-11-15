#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### parameters
backend="$1"
###################################### weights
"$(mp_python mp)" "$MP_ROOT/src/weights.py" --backend "$backend" --models_dir "$MP_ROOT/data/models" --checksums_dir "$MP_ROOT/configs/checksums" --yaml-in "$MP_PARAMS"
mp_meta _analysis "00_weights_$backend" backend="$backend" checksums="@$MP_ROOT/data/models/$backend/CHECKSUMS" source="@$MP_ROOT/data/models/$backend/SOURCE.json"
mp_footer
