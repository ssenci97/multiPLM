"""Shared fixtures: module loading by path, a throw-away repo copy and fake SLURM/singularity commands on PATH."""
import importlib.util, os, shutil, stat, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
FAKES = {
    "sbatch": '#!/bin/bash\nd="$(dirname "$0")"; n=$(( $(cat "$d/count" 2>/dev/null || echo 1000) + 1 )); echo "$n" > "$d/count"; echo "$n $*" >> "$d/sbatch.log"; echo "$n" >> "$d/live"; echo "$n"\n',
    "squeue": '#!/bin/bash\nd="$(dirname "$0")"; [ -f "$d/live" ] || exit 0; case "$*" in *%T*) while read -r i; do echo "$i|PENDING"; done < "$d/live" ;; *%k*) ;; *) cat "$d/live" ;; esac\n',
    "sacct": '#!/bin/bash\nd="$(dirname "$0")"; [ -f "$d/sacct.txt" ] && cat "$d/sacct.txt"; exit 0\n',
    "scancel": '#!/bin/bash\nd="$(dirname "$0")"; echo "$*" >> "$d/scancel.log"; for i in "$@"; do sed -i "/^$i\\$/d" "$d/live" 2>/dev/null; done; exit 0\n',
    "sinfo": "#!/bin/bash\nexit 0\n",
    "singularity": "#!/bin/bash\nexit 0\n",
}

def load(name):
    """Import src/<name>.py by path."""
    spec = importlib.util.spec_from_file_location(name, f"{ROOT}/src/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

@pytest.fixture
def repo(tmp_path):
    """A copy of src/, configs/, gui/ and containers/ in a temporary folder."""
    r = tmp_path / "repo"
    for d in ("src", "configs", "gui", "containers"): shutil.copytree(f"{ROOT}/{d}", r / d, ignore=shutil.ignore_patterns("__pycache__", "*.sif"))
    (r / "containers" / "images.env").write_text("MP_IMAGE_REF=ghcr.io/test/multiplm@sha256:" + "0" * 64 + "\n")
    return r

@pytest.fixture
def fakebin(tmp_path, monkeypatch):
    """Fake sbatch/squeue/sacct/scancel/sinfo/singularity first on PATH."""
    b = tmp_path / "bin"; b.mkdir()
    for n, body in FAKES.items():
        (b / n).write_text(body); (b / n).chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    monkeypatch.setenv("MP_SIF", str(tmp_path / "fake.sif"))
    monkeypatch.setenv("NODE_PICK", "0")
    return b
