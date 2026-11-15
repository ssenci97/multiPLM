"""prefetch.py: the per-sample protein hard cap stops the download before any sequence is fetched."""
import sys
import pytest
from conftest import load

sys.argv = ["prefetch.py", "--yaml-in", "/nonexistent.yaml"]
prefetch = load("prefetch")

def fake_fetch(total, calls, entries=1):
    def fetch(url):
        calls.append(url)
        return "".join(f">sp|P{i:05d}|A_HUMAN a\nMKV\n" for i in range(entries)), {"x-total-results": str(total), "x-uniprot-release": "2026_04", "x-uniprot-release-date": "2026-10-01"}
    return fetch

def test_query_above_the_sample_cap_is_refused_after_the_first_page(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch, "fetch", fake_fetch(prefetch.analysis.MAX_PROTEINS_SAMPLE + 1, calls))
    with pytest.raises(ValueError, match="hard cap"): prefetch.download(["keyword:KW-0407"])
    assert len(calls) == 1

def test_accession_chunks_count_together_towards_the_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch.analysis, "MAX_PROTEINS_SAMPLE", 150)
    monkeypatch.setattr(prefetch, "fetch", fake_fetch(100, calls, 100))
    with pytest.raises(ValueError, match="hard cap"): prefetch.download(["accession:A", "accession:B"])
    assert len(calls) == 2

def test_max_entries_refuses_below_the_hard_cap_and_never_raises_it(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch, "fetch", fake_fetch(501, calls))
    with pytest.raises(ValueError, match=r"prefetch.max_entries of 500 .*nothing was saved"): prefetch.download(["xref:pfam-PF00011"], 500)
    monkeypatch.setattr(prefetch, "fetch", fake_fetch(prefetch.analysis.MAX_PROTEINS_SAMPLE + 1, calls))
    with pytest.raises(ValueError, match="hard cap"): prefetch.download(["xref:pfam-PF00011"], 10**9)
    assert len(calls) == 2

def test_missing_total_header_is_refused_on_the_first_page(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch, "fetch", lambda url: (calls.append(url), (">sp|P00001|A_HUMAN a\nMKV\n", {"x-uniprot-release": "2026_04"}))[1])
    with pytest.raises(RuntimeError, match="x-total-results"): prefetch.download(["keyword:KW-0407"])
    assert len(calls) == 1

def test_query_under_the_cap_downloads_every_announced_entry(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch, "fetch", fake_fetch(3, calls, 3))
    body, info = prefetch.download(["keyword:KW-0407"])
    assert body.count(">") == 3 and info[0]["entries"] == 3 and info[0]["uniprot_release"] == "2026_04"
