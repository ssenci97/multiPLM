"""Workflow wiring: shell syntax of every script and the container guard of the step scripts."""
import glob, os, subprocess
import pytest
from conftest import ROOT

@pytest.mark.parametrize("script", sorted(glob.glob(f"{ROOT}/src/*.sh") + glob.glob(f"{ROOT}/containers/*.sh")))
def test_bash_syntax(script):
    subprocess.run(["bash", "-n", script], check=True)

@pytest.mark.parametrize("script", ["00_weights.sh", "01_prepare.sh", "02_dedup.sh", "03_embed.sh", "04_merge.sh", "05_summary.sh"])
def test_steps_refuse_outside_the_container(script, tmp_path):
    r = subprocess.run(["bash", f"{ROOT}/src/{script}", "x", "y"], capture_output=True, text=True, env=dict(os.environ, MP_OUT=str(tmp_path)))
    assert r.returncode != 0 and "refusing to run outside the container" in r.stdout + r.stderr

def test_python_compiles():
    subprocess.run(["python3", "-m", "py_compile"] + glob.glob(f"{ROOT}/src/*.py") + glob.glob(f"{ROOT}/src/backends/*.py"), check=True)
