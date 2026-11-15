#!/usr/bin/env python3
"""Pre-download UniProtKB FASTA (queries or accession lists) on a login node, with provenance (query, release, date, sha256)."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import csv
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis
ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = "configs/param_configs.yaml"
SCRIPT_KEY = "prefetch"
SAMPLES_TSV = "configs/samples.tsv"
REVIEWED_ONLY = False
SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PAGE_SIZE = 500
ACCESSIONS_PER_QUERY = 100
HTTP_RETRIES = 5
HTTP_BACKOFF = 5
HTTP_TIMEOUT = 300
ACC_PREFIX = "accessions:"
ACC_RE = re.compile(r"^[A-Z0-9]{6,10}(-\d+)?$")
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Pre-download UniProtKB FASTA for the samples that use a query")
parser.add_argument("--samples", type=str, default=SAMPLES_TSV, help="Sample sheet (configs/samples.tsv or an example's samples.tsv)")
parser.add_argument("--reviewed_only", action=argparse.BooleanOptionalAction, default=REVIEWED_ONLY, help="Add reviewed:true (Swiss-Prot) to every query")
parser.add_argument("--force", action="store_true", help="Download again even when a file for the same query exists")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
params = dict(reviewed_only=REVIEWED_ONLY)
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
def queries(text: str, reviewed: bool) -> list[str]:
    """UniProtKB queries of one sample: the query itself, or OR-groups of at most 100 accessions; reviewed:true added on request."""
    text = text.strip()
    if text.startswith(ACC_PREFIX):
        acc = [a for a in re.split(r"[\s,;]+", text[len(ACC_PREFIX):]) if a]
        bad = [a for a in acc if not ACC_RE.match(a)]
        if bad or not acc: raise ValueError(f"not UniProt accessions: {bad[:5] or 'none given'}")
        out = ["(" + " OR ".join(f"accession:{a}" for a in acc[i:i + ACCESSIONS_PER_QUERY]) + ")" for i in range(0, len(acc), ACCESSIONS_PER_QUERY)]
    else: out = [text]
    return [f"({q}) AND (reviewed:true)" if reviewed else q for q in out]
def fetch(url: str) -> tuple[str, dict]:
    """GET one UniProt URL, retrying rate limits, server errors and dropped connections with exponential backoff (after SCambrian dw_uniprot_seqs.py)."""
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r: return r.read().decode("utf-8"), {k.lower(): v for k, v in r.headers.items()}
        except urllib.error.HTTPError as e:
            reason = f"HTTP {e.code}"
            if (e.code != 429 and e.code < 500) or attempt == HTTP_RETRIES: raise RuntimeError(f"UniProt {reason} for {url}: {e.read().decode('utf-8', 'replace')[:300]}") from None
        except (OSError, http.client.HTTPException) as e:
            reason = f"connection error ({e})"
            if attempt == HTTP_RETRIES: raise RuntimeError(f"UniProt {reason} after {attempt} attempts for {url}") from None
        log(f"UniProt {reason}, retry {attempt}/{HTTP_RETRIES - 1} in {HTTP_BACKOFF * 2 ** (attempt - 1)} s")
        time.sleep(HTTP_BACKOFF * 2 ** (attempt - 1))
    raise RuntimeError("unreachable")
def download(qs: list[str]) -> tuple[str, list[dict]]:
    """Page through every query; each must return its announced x-total-results entries."""
    body, info = [], []
    for q in qs:
        url = first = f"{SEARCH_URL}?query={urllib.parse.quote(q, safe='')}&format=fasta&size={PAGE_SIZE}"
        n, total = 0, None
        while url:
            text, h = fetch(url)
            total, release, date = total or h.get("x-total-results"), h.get("x-uniprot-release"), h.get("x-uniprot-release-date")
            body.append(text); n += text.count("\n>") + text.startswith(">")
            m = re.search(r'<([^>]+)>;\s*rel="next"', h.get("link", ""))
            url = m.group(1) if m else None
            log(f"  {n:,}/{total} entries")
        if not n or total is None or n != int(total): raise RuntimeError(f"incomplete UniProt download for {q!r}: {n} entries, {total} announced")
        info.append(dict(query=q, url=first, entries=n, uniprot_release=release, uniprot_release_date=date))
    return "".join(body), info
###################################### MAIN
def main() -> None:
    """Download every missing UniProt sample of the sheet into data/uniprot/<sample>/<query key>.fasta (+ .json, see analysis.uniprot_file)."""
    t0 = time.time()
    sheet = Path(args.samples) if Path(args.samples).is_absolute() else ROOT / args.samples
    todo = [r for r in csv.DictReader(open(sheet), delimiter="\t") if (r.get("uniprot_query") or "").strip()]
    if not todo: log(f"no sample of {sheet} uses a uniprot_query: nothing to download")
    for r in todo:
        sid, text = r["sample_id"], r["uniprot_query"]
        out = Path(analysis.uniprot_file(str(ROOT), sid, text, args.reviewed_only))
        if out.is_file() and out.with_suffix(".json").is_file() and not args.force: log(f"{sid}: already downloaded ({out.relative_to(ROOT)})"); continue
        log(f"{sid}: {text.strip()[:120]}")
        try: body, info = download(queries(text, args.reviewed_only))
        except (ValueError, RuntimeError) as e: sys.exit(f"[PYTHON-INFO] !! {sid}: {e}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp"); tmp.write_text(body); os.replace(tmp, out)
        meta = dict(sample=sid, uniprot_query=text.strip(), reviewed_only=args.reviewed_only, requests=info, entries=sum(i["entries"] for i in info), sha256=hashlib.sha256(out.read_bytes()).hexdigest(), downloaded_at=f"{datetime.datetime.now():%F %T}")
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        log(f"{sid}: {meta['entries']:,} entries, UniProt {info[0]['uniprot_release']} ({info[0]['uniprot_release_date']}) -> {out.relative_to(ROOT)}")
    log(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | elapsed {time.time() - t0:.1f} s | max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB")
if __name__ == "__main__":
    main()
