#!/bin/bash
###################################### dependencies
source "$(dirname "$0")/common.sh"
###################################### summary
"$(mp_python mp)" "$MP_ROOT/src/summarize.py" --analysis_dir "$MP_OUT" --yaml-in "$MP_PARAMS"
mp_meta _analysis 05_summary summary=summary.tsv aggregates=aggregates.json
mp_footer
