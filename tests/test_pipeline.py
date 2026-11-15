"""Driver: plan, done rule, frozen inputs, preflight, path safety, the GPU lane and a full launch with fake SLURM commands."""
import json, os
import pytest
import yaml
from conftest import load

analysis, pipeline = load("analysis"), load("pipeline")
EXAMPLE = "example_toy"

def make(repo, name="toy", backends=None, slurm=None):
    ex = repo / "configs" / "examples" / EXAMPLE
    if backends or slurm:
        P = yaml.safe_load((ex / "param_configs.yaml").read_text())
        if backends: P["embed"]["backends"] = backends
        if slurm: P["slurm"].update(slurm)
        (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    d, new, errs = analysis.create(name, str(repo), str(ex))
    assert new and not errs, errs
    return d

def footer(job):
    os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
    open(job["log"], "w").write("[SLURM-INFO] 2026-11-14 10:00:00 | elapsed 00:00:05 | max RSS 0.10 GB\n")
    for o in job["outputs"]:
        os.makedirs(os.path.dirname(o), exist_ok=True); open(o, "w").write("x")

def test_example_freezes_absolute_fasta_paths_and_tools(repo):
    d = make(repo)
    rows = analysis.read_sheet(f"{d}/inputs/samples.tsv")
    assert all(os.path.isabs(r["fasta_path"]) and os.path.isfile(r["fasta_path"]) for r in rows)
    assert os.path.isfile(f"{d}/inputs/tools.tsv") and "tools.tsv" in json.load(open(f"{d}/inputs/analysis.json"))["checksums"]

def test_plan_has_one_job_per_step_unit(repo):
    make(repo, backends=["esmc_300m", "esm2_t6_8M"])
    jobs = pipeline.plan("toy", str(repo)); keys = [j["key"] for j in jobs]
    assert keys == ["weights.esmc_300m", "weights.esm2_t6_8M", "prepare.globins", "prepare.small_proteins", "embed.esmc_300m.globins.1", "merge.esmc_300m.globins", "embed.esmc_300m.small_proteins.1", "merge.esmc_300m.small_proteins",
                    "embed.esm2_t6_8M.globins.1", "merge.esm2_t6_8M.globins", "embed.esm2_t6_8M.small_proteins.1", "merge.esm2_t6_8M.small_proteins", "summary"]
    by = {j["key"]: j for j in jobs}
    assert by["embed.esmc_300m.globins.1"]["gpu"] and by["embed.esmc_300m.globins.1"]["deps"] == ["prepare.globins", "weights.esmc_300m"] and not by["merge.esmc_300m.globins"]["gpu"]
    assert by["merge.esmc_300m.globins"]["deps"] == ["embed.esmc_300m.globins.1"] and (by["merge.esmc_300m.globins"]["cpus"], by["merge.esmc_300m.globins"]["mem_mb"]) == (1, 1000)
    assert by["weights.esmc_300m"]["shared"] and by["weights.esmc_300m"]["online"] and not by["prepare.globins"]["online"]
    assert all(not j["gpu"] and (j["cpus"], j["mem_mb"]) == (1, 1000) for j in jobs if pipeline.kind(j["key"]) in ("weights", "prepare", "summary"))
    assert by["embed.esm2_t6_8M.globins.1"]["args"][2] == "esm-fair" and by["embed.esmc_300m.globins.1"]["args"][2] == "esm-evo"

def test_done_rule_footer_mtime_and_weights_revision(repo):
    make(repo)
    jobs, run = pipeline.todo("toy", str(repo)); by = {j["key"]: j for j in jobs}
    assert run == set(by)
    for j in jobs:
        if j["key"] != "weights.esmc_300m": footer(j)
    w = by["weights.esmc_300m"]; footer(w)
    json.dump(dict(revision="0" * 40), open(os.path.join(os.path.dirname(w["outputs"][0]), "SOURCE.json"), "w"))
    assert "weights.esmc_300m" in pipeline.todo("toy", str(repo))[1]
    json.dump(dict(revision=w["revision"]), open(os.path.join(os.path.dirname(w["outputs"][0]), "SOURCE.json"), "w"))
    assert pipeline.todo("toy", str(repo))[1] == set()
    p = by["prepare.globins"]["outputs"][0]; os.utime(p, (os.path.getmtime(p) + 100,) * 2)
    assert {"embed.esmc_300m.globins.1", "merge.esmc_300m.globins", "summary"} <= pipeline.todo("toy", str(repo))[1]

def test_frozen_inputs_and_pins_are_verified(repo):
    d = make(repo)
    pipeline.verify_inputs("toy", str(repo))
    open(f"{repo}/configs/tools.tsv", "a").write("\n")
    with pytest.raises(ValueError, match="pins"): pipeline.verify_inputs("toy", str(repo))
    open(f"{d}/inputs/params.yaml", "a").write("\n")
    with pytest.raises(ValueError, match="frozen input"): pipeline.verify_inputs("toy", str(repo))

def test_preflight_and_path_safety(repo):
    make(repo)
    jobs = pipeline.plan("toy", str(repo))
    pipeline.preflight(jobs, {j["key"] for j in jobs})
    with pytest.raises(ValueError, match="missing or empty input"): pipeline.preflight(jobs, {"embed.esmc_300m.globins.1"})
    bad = [dict(jobs[2], outputs=[f"{repo}/results/other/x"])]
    with pytest.raises(ValueError, match="unsafe"): pipeline.check_paths(bad, "toy", str(repo))

def test_gpu_rule_limits(repo):
    make(repo, "a", slurm=dict(gpus_max=3))
    with pytest.raises(ValueError, match="GPU rule"): pipeline.plan("a", str(repo))

def test_node_fraction_above_half_is_refused(repo):
    make(repo, "b", slurm=dict(node_fraction=0.75))
    with pytest.raises(ValueError, match="explicit approval"): pipeline.plan("b", str(repo))

def gpu_schedule(repo, name, backends, gpus_max):
    make(repo, name, backends=backends, slurm=dict(gpus_max=gpus_max))
    jobs = pipeline.plan(name, str(repo)); S = pipeline.slurm_cfg(analysis.load_yaml(f"{repo}/results/{name}/inputs/params.yaml"))
    for j in jobs: j["est"] = 1
    return jobs, pipeline.schedule(jobs, dict(node="", part="GENOA", cpus=4, mem=4000), S)

def test_gpu_lane_one_gpu_at_a_time_by_default(repo):
    jobs, sch = gpu_schedule(repo, "c", ["esmc_300m", "prott5_xl_half"], 1)
    gpu = [k for k in sch["order"] if k.startswith("embed.")]
    assert all(set(sch["prev"][k]) & set(gpu) for k in gpu[1:])
    spans = sorted((sch["start"][k], sch["end"][k]) for k in gpu)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))

def test_all_gpu_jobs_on_dgx_one_at_a_time(repo):
    jobs, sch = gpu_schedule(repo, "d", ["esmc_6b", "esmc_300m"], 2)
    assert {x["partition"] for x in jobs if x["key"].startswith("embed.")} == {"DGX"}
    iv = sorted((sch["start"][k], sch["end"][k]) for k in sch["order"] if k.startswith("embed."))
    assert len(iv) >= 2 and all(a[1] <= b[0] for a, b in zip(iv, iv[1:]))

def test_launch_with_fake_slurm(repo, fakebin):
    make(repo, backends=["esmc_300m"])
    out = pipeline.submit("toy", str(repo))
    assert sum(j["state"] == "submitted" for j in out) == 8
    log = (fakebin / "sbatch.log").read_text().splitlines()
    embed = [l for l in log if "embed.esmc_300m" in l]
    assert all(" -p DGX " in l and " --gres=gpu:A100:1 " in l and "--dependency=afterok:" in l for l in embed)
    assert all("--gres" not in l and " -c 1 --mem=1000 " in l for l in log if "embed." not in l) and sum("merge." in l for l in log) == 2
    assert "afterany:" in embed[1]
    rows = pipeline.read_jobs("toy", str(repo))
    assert [r["gpu"] for r in rows if r["key"].startswith("embed.")] == ["1", "1"]
    with pytest.raises(RuntimeError, match="live jobs"): pipeline.submit("toy", str(repo))
    r = pipeline.stop("toy", str(repo))
    assert len(r["cancelled"]) == 8 and not r["remaining"]

def test_dry_run_submits_nothing(repo, fakebin):
    make(repo)
    out = pipeline.submit("toy", str(repo), dry_run=True)
    assert {j["state"] for j in out} == {"would-run"} and not (fakebin / "sbatch.log").exists()

def test_shards_follow_the_protein_count(repo):
    ex = repo / "configs" / "examples" / EXAMPLE
    P = yaml.safe_load((ex / "param_configs.yaml").read_text()); P["embed"]["shard_proteins"] = 2; (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    analysis.create("sh", str(repo), str(ex))
    keys = [j["key"] for j in pipeline.plan("sh", str(repo))]
    assert [k for k in keys if k.startswith("embed.esmc_300m.globins.")] == [f"embed.esmc_300m.globins.{k}" for k in (1, 2, 3)]
    merge = next(j for j in pipeline.plan("sh", str(repo)) if j["key"] == "merge.esmc_300m.globins")
    assert merge["args"][2] == "3" and len(merge["deps"]) == 3

def test_layers_are_validated_against_each_backend(repo):
    ex = repo / "configs" / "examples" / EXAMPLE
    P = yaml.safe_load((ex / "param_configs.yaml").read_text()); P["embed"]["backends"] = ["esm3_sm_open_v1"]; P["embed"]["layers"] = [10]; (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    _, new, errs = analysis.create("ly", str(repo), str(ex))
    assert not new and any("only the last layer" in e for e in errs)
    P["embed"]["backends"], P["embed"]["layers"] = ["esm2_t6_8M"], [7]; (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    assert any("outside" in e for e in analysis.create("ly", str(repo), str(ex))[2])

def test_size_estimate_warns_refuses_and_can_be_overridden(repo, fakebin):
    make(repo, backends=["esmc_300m"])
    est = pipeline.estimate("toy", str(repo))
    assert est["level"] == "ok" and [x["proteins"] for x in est["rows"]] == [5, 5] and est["rows"][0]["gb"] == round(5 * 960 * 4 / 1e9, 3)
    ex = repo / "configs" / "examples" / EXAMPLE
    P = yaml.safe_load((ex / "param_configs.yaml").read_text()); P["embed"].update(layers="all", pooling=["mean", "tokens"], size_warn_gb=0.0001, size_cap_gb=0.001); (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    analysis.create("big", str(repo), str(ex))
    big = pipeline.estimate("big", str(repo))
    assert big["level"] == "refuse" and big["rows"][0]["residues"] > 5
    with pytest.raises(ValueError, match="size_cap_gb"): pipeline.submit("big", str(repo), dry_run=True)
    assert pipeline.submit("big", str(repo), dry_run=True, allow_large=True)

def test_dedup_jobs_sit_between_prepare_and_embed(repo):
    ex = repo / "configs" / "examples" / EXAMPLE
    for scope, expect in (("sample", ["dedup.globins", "dedup.small_proteins"]), ("joint", ["dedup"])):
        P = yaml.safe_load((ex / "param_configs.yaml").read_text()); P["dedup"].update(enabled=True, scope=scope); (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
        analysis.create(f"dd_{scope}", str(repo), str(ex))
        by = {j["key"]: j for j in pipeline.plan(f"dd_{scope}", str(repo))}
        assert [k for k in by if k.startswith("dedup")] == expect
        e = by["embed.esmc_300m.globins.1"]
        assert e["deps"][0] == expect[0] and e["args"][3].endswith("/dedup/globins.fasta") and (by[expect[0]]["cpus"], by[expect[0]]["mem_mb"]) == (1, 1000)

def test_uniprot_samples_need_a_prefetch_and_are_frozen(repo):
    ex = repo / "configs" / "examples" / EXAMPLE
    (ex / "samples.tsv").write_text("sample_id\tfasta_path\tuniprot_query\tdescription\nup\t\taccessions:P69905,P68871\t\n")
    _, new, errs = analysis.create("up1", str(repo), str(ex))
    assert not new and "prefetch.py" in errs[0]
    f = analysis.uniprot_file(str(repo), "up", "accessions:P69905,P68871", False)
    os.makedirs(os.path.dirname(f)); open(f, "w").write(">sp|P69905|HBA\nMVLS\n"); open(f[:-6] + ".json", "w").write(json.dumps(dict(uniprot_query="accessions:P69905,P68871", requests=[dict(uniprot_release="2026_04")])))
    d, new, errs = analysis.create("up2", str(repo), str(ex))
    assert new and not errs and open(f"{d}/inputs/uniprot/up.fasta").read().startswith(">sp|P69905") and "uniprot/up.fasta" in json.load(open(f"{d}/inputs/analysis.json"))["checksums"]
    assert next(j for j in pipeline.plan("up2", str(repo)) if j["key"] == "prepare.up")["args"][1].endswith("inputs/uniprot/up.fasta")

def test_summary_memory_follows_reps_and_twonn(repo):
    make(repo, "off")
    off = next(j for j in pipeline.plan("off", str(repo)) if j["key"] == "summary")
    assert (off["cpus"], off["mem_mb"]) == (1, 1000) and "jax" not in off["why"]
    ex = repo / "configs" / "examples" / EXAMPLE
    P = yaml.safe_load((ex / "param_configs.yaml").read_text()); P["summarize"]["twonn"]["enabled"] = True; (ex / "param_configs.yaml").write_text(yaml.safe_dump(P))
    analysis.create("on", str(repo), str(ex))
    on = next(j for j in pipeline.plan("on", str(repo)) if j["key"] == "summary")
    assert on["mem_mb"] == 1000 + pipeline.TWONN_RUNTIME_MB and "jax/DADApy" in on["why"]
    pipeline.size_job(on, [], {})
    assert on["mem_mb"] == 6000 and on["sized"].startswith("default: total size of the reps")
