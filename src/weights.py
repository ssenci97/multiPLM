#!/usr/bin/env python3
"""Download the pinned weights of one backend into data/models/<backend>/ and record and check the sha256 of every file (any mismatch stops the job)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import urllib.request
from huggingface_hub import __version__ as hub_version
from huggingface_hub import snapshot_download
from tqdm import tqdm
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import compare, model_dir, read_checksums, registry, sha256, weight_files, write_checksums
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "weights"
MODELS_DIR = "data/models"
CHECKSUMS_DIR = "configs/checksums"
HF_PREFIX = "https://huggingface.co/"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Download and checksum the pinned weights of one backend")
parser.add_argument("--backend", type=str, default=None, help="Backend name of configs/tools.tsv")
parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
parser.add_argument("--checksums_dir", type=str, default=CHECKSUMS_DIR, help="Tracked reference checksums (<backend>.sha256), optional")
parser.add_argument("--token", type=str, default=None, help="Hugging Face token (prefer MP_HF_TOKEN_FILE: a command-line token is visible in ps)")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(models_dir=MODELS_DIR, checksums_dir=CHECKSUMS_DIR)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    params.update({k: v for k, v in (config_data.get(SCRIPT_KEY) or {}).items() if k in params})
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = (yaml.safe_load(Path(args.yaml_out).read_text()) or {}) if Path(args.yaml_out).exists() else {}
    config_data[SCRIPT_KEY] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).write_text(yaml.safe_dump(config_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def log(msg: str) -> None:
    """Print a tagged log line."""
    print(f"[PYTHON-INFO] {msg}", flush=True)
def fail(problems: list[str], what: str) -> None:
    """Stop with every checksum problem listed."""
    sys.exit(f"[PYTHON-INFO] !! {what}:\n  " + "\n  ".join(problems))
def fetch_hf(spec: dict, files: Path) -> None:
    """Download the pinned Hugging Face revision into files/ (resumable)."""
    patterns = [p for p in spec["files"] if p != "*"] or None
    snapshot_download(repo_id=spec["repo"].removeprefix(HF_PREFIX), revision=spec["commit"], local_dir=str(files), allow_patterns=patterns, token=args.token or None)
def fetch_url(spec: dict, files: Path) -> None:
    """Download the fair-esm model and contact-regression files (resumable by file)."""
    for name in tqdm(spec["files"], desc=f"{spec['name']} files"):
        dst = files / name
        if dst.exists(): continue
        sub = "regression" if name.endswith("-contact-regression.pt") else "models"
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(f"{spec['repo']}/{sub}/{name}", timeout=600) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f, 1 << 24)
        tmp.replace(dst)
###################################### MAIN
def main() -> None:
    """Download if needed, checksum, compare and record."""
    t0 = time.time()
    if not args.backend:
        if args.yaml_out: return
        parser.error("--backend is required")
    spec = registry()[args.backend]
    d = model_dir(Path(args.models_dir), spec["name"]); files = d / "files"; src = d / "SOURCE.json"
    old = json.loads(src.read_text()) if src.is_file() else {}
    if old and old.get("revision") != spec["commit"]: fail([f"{d} holds revision {old.get('revision')}, configs/tools.tsv pins {spec['commit']}: move the folder aside first"], "pinned revision changed")
    files.mkdir(parents=True, exist_ok=True)
    log(f"{spec['name']}: {spec['repo']} @ {spec['commit']} (~{spec['size_gb']} GB) -> {files}")
    fetch_hf(spec, files) if spec["repo"].startswith(HF_PREFIX) else fetch_url(spec, files)
    got = {str(p.relative_to(d)): sha256(p) for p in tqdm(weight_files(d), desc="sha256")}
    rec = read_checksums(d / "CHECKSUMS")
    if rec and (bad := compare(got, rec, f"{d}/CHECKSUMS")): fail(bad, f"{spec['name']}: files differ from the recorded CHECKSUMS")
    ref_path = Path(args.checksums_dir) / f"{spec['name']}.sha256"
    ref = read_checksums(ref_path)
    if ref and (bad := compare(got, ref, str(ref_path))): fail(bad, f"{spec['name']}: download differs from the tracked reference checksums")
    write_checksums(d, got)
    src.write_text(json.dumps(dict(name=spec["name"], repo=spec["repo"], revision=spec["commit"], licence_in_tools_tsv=spec["licence"], files={k: os.path.getsize(d / k) for k in got},
                                   downloaded_at=old.get("downloaded_at") or f"{datetime.datetime.now():%F %T}", verified_at=f"{datetime.datetime.now():%F %T}", huggingface_hub=hub_version), indent=1))
    log(f"{spec['name']}: {len(got)} file(s), {sum(os.path.getsize(d / k) for k in got) / 1e9:.2f} GB, sha256 recorded in {d}/CHECKSUMS" + ("" if ref else f"; no {ref_path} yet: copy CHECKSUMS there and commit it to pin these bytes"))
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
