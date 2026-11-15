#!/bin/bash
###################################### helpers
find_app() {
  local app="${MP_APPTAINER:-$(command -v singularity || command -v apptainer || true)}"
  [ -n "$app" ] || app="$(bash -lc 'module load singularity >/dev/null 2>&1; command -v singularity || command -v apptainer' 2>/dev/null || true)"
  [ -n "$app" ] || [ ! -x /opt/programs/singularity/4.3.1/bin/singularity ] || app=/opt/programs/singularity/4.3.1/bin/singularity
  [ -n "$app" ] || { echo "[SLURM-INFO] no singularity/apptainer found on $(hostname): set MP_APPTAINER=/path/to/singularity" >&2; return 1; }
  echo "$app"
}
